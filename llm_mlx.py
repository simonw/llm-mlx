import click
import json
import llm
import re
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from pydantic import Field
from typing import Optional
import os
from pathlib import Path
import sys

disable_progress_bars()

# These defaults copied from llama.cpp
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_MIN_P = 0.1
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_MAX_TOKENS = 1024

DEBUG = False

def _ensure_models_file():
    filepath = llm.user_dir() / "llm-mlx.json"
    if not filepath.exists():
        filepath.write_text("{}")
    return filepath


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def mlx():
        "Commands for working with MLX models"

    @mlx.command()
    def models_file():
        "Display the path to the llm-mlx.json file"
        click.echo(_ensure_models_file())

    @mlx.command()
    @click.argument("model_path")
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def download_model(model_path, aliases):
        "Download and register a MLX model"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        models[model_path] = {"aliases": aliases}
        enable_progress_bars()
        MlxModel(model_path).prompt("hi").text()
        disable_progress_bars()
        models_file.write_text(json.dumps(models, indent=2))

    @mlx.command()
    def models():
        "List registered MLX models"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        click.echo(json.dumps(models, indent=2))

    @mlx.command()
    def manage_models():
        "Register MLX models from the Hugging Face cache on disk"
        cache_dir = Path(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        )
        found_models = set()
        for root, dirs, _ in os.walk(cache_dir):
            for d in dirs:
                if "mlx-community" in d:
                    model_dir = Path(root) / d
                    snapshots_dir = model_dir / "snapshots"
                    if not snapshots_dir.exists():
                        continue

                    # Search for config.json in any subfolder under snapshots
                    config_found = False
                    for config_root, _, config_files in os.walk(snapshots_dir):
                        if "config.json" in config_files:
                            config_path = Path(config_root) / "config.json"
                            try:
                                with open(config_path) as f:
                                    config = json.load(f)
                                    model_type = config.get("model_type", "").lower()
                                    if model_type in [
                                        "whisper",
                                        "llava",
                                        "paligemma",
                                        "qwen2_vl",
                                        "qwen2_5_vl",
                                        "florence2",
                                        "florence",
                                    ]:
                                        continue
                                    config_found = True
                                    break
                            except (json.JSONDecodeError, FileNotFoundError):
                                continue

                    if not config_found:
                        continue
                    parts = d.split("--")
                    model_name = "/".join(parts[1:])
                    if model_name:
                        # Store model_type along with model_name
                        found_models.add((model_type, model_name))

        if not found_models:
            click.echo("No MLX models found in Hugging Face cache")
            return

        models_file = _ensure_models_file()
        existing_models = json.loads(models_file.read_text())

        # Create list of models with their current registration status
        model_choices = []
        for model_type, model in sorted(found_models):
            is_registered = model in existing_models
            action = "Unregister" if is_registered else "Register"
            # Include model_type in display name
            display_name = f"{action} {model} ({model_type})"
            model_choices.append((display_name, model, is_registered))

        # Show interactive selection menu
        selected = select_models(model_choices)
        print("\nUpdating {}".format(models_file))
        for display_name, model_name, is_registered in selected:
            if is_registered:
                # Unregister model
                del existing_models[model_name]
                models_file.write_text(json.dumps(existing_models, indent=2))
                click.echo(f"  Unregistered {model_name}")
            else:
                # Register new model
                existing_models[model_name] = {"aliases": []}
                models_file.write_text(json.dumps(existing_models, indent=2))
                click.echo(f"  Registered {model_name}")


@llm.hookimpl
def register_models(register):
    for model_path, config in json.loads(_ensure_models_file().read_text()).items():
        aliases = config.get("aliases", [])
        register(MlxModel(model_path), aliases=aliases)


class MlxModel(llm.Model):
    can_stream = True
    supports_tools = True

    class Options(llm.Options):
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate",
            ge=0,
            default=None,
        )
        unlimited: Optional[bool] = Field(
            description="Unlimited output tokens",
            default=None,
        )
        temperature: Optional[float] = Field(
            description="Sampling temperature",
            ge=0,
            default=None,
        )
        top_p: Optional[float] = Field(
            description="Sampling top-p",
            ge=0,
            le=1,
            default=None,
        )
        min_p: Optional[float] = Field(
            description="Sampling min-p",
            ge=0,
            le=1,
            default=None,
        )
        min_tokens_to_keep: Optional[int] = Field(
            description="Minimum tokens to keep for min-p sampling",
            ge=1,
            default=None,
        )
        seed: Optional[int] = Field(
            description="Random number seed",
            default=None,
        )

    def __init__(self, model_path):
        self.model_id = model_path
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._tool_format = None

    def _load(self):
        from mlx_lm import load

        if self._model is None:
            self._model, self._tokenizer = load(self.model_path)
        if self._tool_format is None:
            # Detect tool call format from tokenizer's chat template
            self._tool_format = self._detect_tool_format(self._tokenizer)
            if DEBUG:
                print(f"Detected tool format: {self._tool_format}")

        if DEBUG:
            print("Template is", self._tokenizer.chat_template)
        return self._model, self._tokenizer

    def build_messages(self, prompt, conversation):
        """Build messages for the chat template with tool support"""
        messages = []
        current_system = None
        
        # Add conversation history
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                messages.append({"role": "assistant", "content": prev_response.text()})
        # Add current system message if needed
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        
        # Add tool results from current prompt if any
        if prompt.tool_results:
            for tool_result in prompt.tool_results:
                messages.append({
                    "role": "tool",  # Template should convert to ipython for llama3.x
                    "name": tool_result.name,
                    "tool_call_id": tool_result.tool_call_id,
                    "content": json.loads(tool_result.output),
                })
        
        # Add current user message (unless we're just adding tool results)
        if not prompt.tool_results:
            messages.append({"role": "user", "content": prompt.prompt})
        
        if DEBUG:
            print(messages)  # Debug output
        return messages

    def _detect_tool_format(self, tokenizer):
        """Analyze the model's chat template to predict tool call format"""
        try:
            template_str = str(tokenizer.chat_template)
            if not template_str:
                return "generic"
            
            # Analyze template content for format markers (like llama.cpp does)
            if "<｜tool▁calls▁begin｜>" in template_str:
                return "deepseek_r1"
            elif "<tool_call>" in template_str:
                return "hermes"
            elif "<|start_header_id|>ipython<|end_header_id|>" in template_str:
                return "llama3x"
            elif "[TOOL_CALLS]" in template_str:
                return "mistral_nemo"
            elif "<function=" in template_str:
                return "functionary"
            elif "functools[" in template_str:
                return "firefunction"
            # Additional patterns for better detection
            elif "llama" in self.model_path.lower() and ("3.2" in self.model_path or "3.1" in self.model_path):
                return "llama3x"
            elif "qwen" in self.model_path.lower():
                return "hermes"  # Qwen often uses <tool_call> format
            elif "mistral" in self.model_path.lower() and "nemo" in self.model_path.lower():
                return "mistral_nemo"
            else:
                return "generic"
        except Exception as e:
            print(f"Warning: Could not analyze template: {e}")
            return "generic"
    
    def _get_format_patterns(self, format_type):
        """Get regex patterns based on detected format"""
        patterns = {
            "llama3x": [
                # Llama 3.x simple format: {"name": "func", "parameters": {...}}
                (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}', 'llama3x'),
                # # Also handle "arguments" variant
                # (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}', 'llama3x_args'),
            ],
            "hermes": [
                # Hermes XML format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
                (r'<tool_call>\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}\s*</tool_call>', 'hermes'),
                # Also support bare JSON in case template doesn't wrap it
                (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}', 'hermes_bare'),
                # Handle parameters variant for Qwen models
                (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}', 'hermes_params'),
            ],
            "deepseek_r1": [
                # DeepSeek R1 format: function<｜tool▁sep｜>name\n```json\n{...}```
                (r'function<｜tool▁sep｜>([^\n]+)\n```json\n(\{[^}]*\})```', 'deepseek_r1'),
            ],
            "mistral_nemo": [
                # Mistral Nemo format: [TOOL_CALLS] [{"name": "func", "arguments": {...}}]
                (r'\[TOOL_CALLS\]\s*\[\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}\s*\]', 'mistral_nemo'),
            ],
            "functionary": [
                # Functionary format: <function=name>{...}</function>
                (r'<function=([^>]+)>(\{[^}]*\})</function>', 'functionary'),
            ],
            "firefunction": [
                # FireFunction format:  functools[{"name": "func", "arguments": {...}}]
                (r'functools\[\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}\s*\]', 'firefunction'),
            ],
            "generic": [
                # OpenAI format: {"tool_calls": [{"type": "function", "function": {"name": "func", "arguments": {...}}}]}
                (r'\{\s*"tool_calls"\s*:\s*\[\s*\{\s*"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}\s*\}\s*\]\s*\}', 'openai'),
                # Simple format fallback: {"name": "func", "parameters": {...}}
                (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}', 'simple'),
                # # Simple format with arguments: {"name": "func", "arguments": {...}}
                (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}', 'simple_args'),
                # Function call format: function_name({...})
                (r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(\{[^}]*\})\s*\)', 'func_call'),
            ]
        }
        return patterns.get(format_type, patterns["generic"])
    
    def _parse_tool_calls(self, text, available_tools, format_type="generic"):
        """Parse tool calls using format-aware patterns"""
        tool_calls = []
        tool_names = [tool.name for tool in available_tools]
        seen_calls = set()  # Track duplicates
        
        # Get patterns for the detected format
        patterns = self._get_format_patterns(format_type)
        
        for pattern, fmt in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                if len(match) >= 2:
                    func_name, args_str = match[0], match[1]
                    
                    # Create a signature to avoid duplicates
                    call_signature = f"{func_name}:{args_str.strip()}"
                    if call_signature in seen_calls:
                        continue
                    
                    if func_name in tool_names:
                        try:
                            # Parse arguments with better error handling
                            if args_str.strip() in ['{}', '']:
                                arguments = {}
                            else:
                                # Handle common JSON parsing issues
                                args_str = args_str.strip()
                                if not args_str.startswith('{'):
                                    args_str = '{' + args_str
                                if not args_str.endswith('}'):
                                    args_str = args_str + '}'
                                arguments = json.loads(args_str)
                            
                            tool_call = llm.ToolCall(
                                name=func_name,
                                arguments=arguments,
                                tool_call_id=f"call_{len(tool_calls)}_{fmt}"
                            )
                            tool_calls.append(tool_call)
                            seen_calls.add(call_signature)
                            
                        except (json.JSONDecodeError, ValueError):
                            continue
        
        return tool_calls

    def execute(self, prompt, stream, response, conversation):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._load()
        messages = self.build_messages(prompt, conversation)
        
        # Convert tools to the format expected by apply_chat_template
        tools = None
        if prompt.tools:
            tools = []
            for tool in prompt.tools:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema or {"type": "object", "properties": {}}
                    }
                }
                tools.append(tool_def)
        
        # Use the tokenizer's built-in chat template with tool support
        try:
            template_args = dict(
                conversation=messages,
                tools=tools,
                add_generation_prompt=True,
                tools_in_user_message=False,  # Required for llama3x
                tokenize=False,  # Let it tokenize directly
            )
            if DEBUG:
                # Print formatted messages for debugging before tokenization
                print("Messages before tokenization:")
                print(tokenizer.apply_chat_template(
                    **template_args
                ))

            chat_prompt = tokenizer.apply_chat_template(
                **template_args,
            )
        except Exception as e:
            # Fallback to basic template without tools if template doesn't support them
            print(f"Warning: Template doesn't support tools, falling back: {e}")
            chat_prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=True
            )

        sampler = make_sampler(
            (
                DEFAULT_TEMPERATURE
                if prompt.options.temperature is None
                else prompt.options.temperature
            ),
            DEFAULT_TOP_P if prompt.options.top_p is None else prompt.options.top_p,
            DEFAULT_MIN_P if prompt.options.min_p is None else prompt.options.min_p,
            (
                DEFAULT_MIN_TOKENS_TO_KEEP
                if prompt.options.min_tokens_to_keep is None
                else prompt.options.min_tokens_to_keep
            ),
        )
        if prompt.options.seed:
            mx.random.seed(prompt.options.seed)

        max_tokens = DEFAULT_MAX_TOKENS
        if prompt.options.max_tokens is not None:
            max_tokens = prompt.options.max_tokens
        if prompt.options.unlimited:
            max_tokens = -1

        # Generate text and collect for tool call parsing
        generated_text = ""
        
        for chunk in stream_generate(
            model,
            tokenizer,
            chat_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
        ):
            generated_text += chunk.text
            if not prompt.tools:
                yield chunk.text
        
        # Set usage info
        response.set_usage(input=chunk.prompt_tokens, output=chunk.generation_tokens)
        response.response_json = {
            "prompt_tps": chunk.prompt_tps,
            "generation_tps": chunk.generation_tps,
            "peak_memory": chunk.peak_memory,
            "finish_reason": chunk.finish_reason,
        }
        
        # Parse tool calls if tools are available
        if prompt.tools:
            tool_calls = self._parse_tool_calls(generated_text, prompt.tools, self._tool_format)
            if tool_calls:
                for tool_call in tool_calls:
                    if DEBUG:
                        print(f"Found tool call: {tool_call.name} with args {tool_call.arguments}")  # Debug output
                    response.add_tool_call(tool_call)
                # Add some new lines to separate tool calls from the main text
                generated_text += "\n\n"
        
        if prompt.tools:
            yield generated_text


def select_models(model_choices):
    selected = [False] * len(model_choices)
    idx = 0
    window_size = os.get_terminal_size().lines - 5

    while True:
        print("\033[H\033[J", end="")
        print(
            "❯ llm mlx manage-models\nAvailable model files (↑/↓ to navigate, SPACE to select, ENTER to confirm, Ctrl+C to quit):"
        )

        window_start = max(
            0, min(idx - window_size + 3, len(model_choices) - window_size)
        )
        window_end = min(window_start + window_size, len(model_choices))

        for i in range(window_start, window_end):
            display_name, _, _ = model_choices[i]
            print(
                f"{'>' if i == idx else ' '} {'◉' if selected[i] else '○'} {display_name}"
            )

        key = get_key()
        if key == "\x1b[A":  # Up arrow
            idx = max(0, idx - 1)
        elif key == "\x1b[B":  # Down arrow
            idx = min(len(model_choices) - 1, idx + 1)
        elif key == " ":
            selected[idx] = not selected[idx]
        elif key == "\r":  # Enter key
            break
        elif key == "\x03":  # Ctrl+C
            print("\nNo changes made.")
            sys.exit(0)

    return [
        choice for choice, is_selected in zip(model_choices, selected) if is_selected
    ]


def get_key():
    """Get a single keypress from the user."""
    import tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
