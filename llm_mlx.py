import os
import sys
import click
import json
import llm
import uuid
import time
import uvicorn
import logging
import mlx.core as mx
from pathlib import Path
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from typing import Union, Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator

disable_progress_bars()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Default generation settings
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_MIN_P = 0.1
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_MAX_TOKENS = 1024

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
    def import_models():
        "Import existing MLX models from the Hugging Face cache"
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

        # Create list of models with their current import status
        model_choices = []
        for model_type, model in sorted(found_models):
            is_imported = model in existing_models
            status = " (already imported)" if is_imported else ""
            # Include model_type in display name
            display_name = f"({model_type}) {model}{status}"
            model_choices.append((display_name, model, is_imported))

        # Show interactive selection menu
        selected = select_models(model_choices)
        print("\nImporting models...\n")
        for display_name, model_name, is_imported in selected:
            if is_imported:
                # Remove model if it was already imported
                del existing_models[model_name]
                models_file.write_text(json.dumps(existing_models, indent=2))
                click.echo(f"Removed {model_name}")
            else:
                # Import new model
                existing_models[model_name] = {"aliases": []}
                models_file.write_text(json.dumps(existing_models, indent=2))
                click.echo(f"Imported {model_name}")
    
    @mlx.command()
    @click.argument("model_identifier")
    @click.option("--port", default=8000, help="Port to run the API server on")
    @click.option("-m", "--model", help="Path to the model")
    @click.option("--alias", help="Register the model with this alias")
    @click.option("-o", "--option", multiple=True, nargs=2, help="Set model options (can be used multiple times)")
    def serve(model_identifier, port, model, alias, option):
        """
        Start an OpenAI-compatible API server for the specified model
        """
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())

        # figure out which model path to load
        if model:
            # register alias if provided
            if alias:
                models[model] = {"aliases": [alias]}
                models_file.write_text(json.dumps(models, indent=2))
                click.echo(f"Registered '{model}' as alias '{alias}'")
            model_to_load = model
            server_model_name = alias or model_identifier
        else:
            model_to_load = None
            for path, config in models.items():
                if model_identifier in config.get("aliases", []):
                    model_to_load = path
                    break
            if not model_to_load:
                raise click.ClickException(f"Model '{model_identifier}' not found")
            server_model_name = model_identifier

        # Convert options to a dictionary
        options = {}
        if option:
            for key, value in option:
                try:
                    # Try to convert to appropriate type
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string if conversion fails
                    pass
                options[key] = value

        # Load the model with options
        try:
            mlx_model = MlxModel(model_to_load)
            # Set options before loading
            mlx_model.default_options = mlx_model.Options(**options)
            mlx_model._load()
        except Exception as e:
            raise click.ClickException(f"Failed to load model: {str(e)}")

        click.echo(f"Starting API server for {server_model_name} on port {port}")
        mlx_model.serve(server_model_name, port)

@llm.hookimpl
def register_models(register):
    for model_path, config in json.loads(_ensure_models_file().read_text()).items():
        aliases = config.get("aliases", [])
        register(MlxModel(model_path), aliases=aliases)
        
class MlxModel(llm.Model):
    can_stream = True

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

    def _load(self):
        if self._model is None:
            self._model, self._tokenizer = load(self.model_path)
        return self._model, self._tokenizer

    def execute(self, prompt, stream, response, conversation):
        model, tokenizer = self._load()

        messages = []
        current_system = None

        # Reconstruct chat conversation from conversation object
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append({"role": "system", "content": prev_response.prompt.system})
                    current_system = prev_response.prompt.system
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})

        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})

        # Convert the conversation messages to model format
        chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Build a sampler
        sampler = make_sampler(
            DEFAULT_TEMPERATURE if prompt.options.temperature is None else prompt.options.temperature,
            DEFAULT_TOP_P if prompt.options.top_p is None else prompt.options.top_p,
            DEFAULT_MIN_P if prompt.options.min_p is None else prompt.options.min_p,
            DEFAULT_MIN_TOKENS_TO_KEEP if prompt.options.min_tokens_to_keep is None else prompt.options.min_tokens_to_keep
        )

        if prompt.options.seed:
            mx.random.seed(prompt.options.seed)

        max_tokens = DEFAULT_MAX_TOKENS
        if prompt.options.max_tokens is not None:
            max_tokens = prompt.options.max_tokens
        if prompt.options.unlimited:
            max_tokens = -1  # unlimited in this system

        # Stream out chunks
        for chunk in stream_generate(
            model,
            tokenizer,
            chat_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
        ):
            yield chunk.text
        # Record usage if desired
        response.set_usage(input=chunk.prompt_tokens, output=chunk.generation_tokens)
        response.response_json = {
            "prompt_tps": chunk.prompt_tps,
            "generation_tps": chunk.generation_tps,
            "peak_memory": chunk.peak_memory,
            "finish_reason": chunk.finish_reason,
        }

    def get_generation_params(self, options=None):
        """Get generation parameters from options or defaults"""
        opts = options or self.default_options
        return {
            "temperature": opts.temperature if opts.temperature is not None else DEFAULT_TEMPERATURE,
            "top_p": opts.top_p if opts.top_p is not None else DEFAULT_TOP_P,
            "min_p": opts.min_p if opts.min_p is not None else DEFAULT_MIN_P,
            "min_tokens_to_keep": opts.min_tokens_to_keep if opts.min_tokens_to_keep is not None else DEFAULT_MIN_TOKENS_TO_KEEP,
            "max_tokens": opts.max_tokens if opts.max_tokens is not None else DEFAULT_MAX_TOKENS
        }

    async def _generate_completion(self, chat_prompt, sampler, max_tokens, model_name):
        """Helper method to generate non-streaming responses"""
        generated_text = ""
        for chunk in stream_generate(
            self._model,
            self._tokenizer,
            chat_prompt,
            sampler=sampler,
            max_tokens=max_tokens
        ):
            generated_text += chunk.text

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    def _normalize_messages(self, messages: Any) -> List[Dict[str, str]]:
        """Helper method to normalize message format"""
        if not isinstance(messages, list):
            raise ValueError(f"messages must be a list, got {type(messages)}")
        normalized = []
        for m in messages:
            if not isinstance(m, dict):
                raise ValueError(f"Each message must be a dict, got {type(m)}")
            if "role" not in m or "content" not in m:
                raise ValueError(f"Each message needs 'role' and 'content' fields: {m}")
            content = m["content"]
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            normalized.append({"role": m["role"], "content": content})
        return normalized

    def _stream_response(self, chat_prompt, sampler, max_tokens, model_name):
        """Helper method to generate streaming responses"""
        for chunk in stream_generate(
            self._model,
            self._tokenizer,
            chat_prompt,
            sampler=sampler,
            max_tokens=max_tokens
        ):
            data = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "delta": {"content": chunk.text},
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    def create_api_app(self, server_model_name):
        """Create FastAPI app with model settings"""
        app = FastAPI(title="MLX OpenAI-Compatible API")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class ChatCompletionRequest(BaseModel):
            model: Optional[str] = None
            messages: Optional[List[Dict[str, Any]]] = None
            temperature: Optional[float] = Field(default=self.default_options.temperature or DEFAULT_TEMPERATURE, ge=0, le=2)
            max_tokens: Optional[int] = Field(default=self.default_options.max_tokens or DEFAULT_MAX_TOKENS, ge=-1)
            top_p: Optional[float] = Field(default=self.default_options.top_p or DEFAULT_TOP_P, ge=0, le=1)
            stream: Optional[bool] = False

            class Config:
                extra = "allow"

            @root_validator(pre=True)
            def fill_defaults(cls, values):
                if not values.get("model"):
                    values["model"] = server_model_name
                if "messages" not in values or values["messages"] is None:
                    values["messages"] = []
                return values

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                messages = self._normalize_messages(request.messages)
                chat_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )

                # Use model's generation parameters
                gen_params = self.get_generation_params()
                gen_params.update({
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens
                })

                sampler = make_sampler(
                    gen_params["temperature"],
                    gen_params["top_p"],
                    gen_params["min_p"],
                    gen_params["min_tokens_to_keep"]
                )

                if request.stream:
                    return StreamingResponse(
                        self._stream_response(chat_prompt, sampler, gen_params["max_tokens"], request.model),
                        media_type="text/event-stream"
                    )
                else:
                    return await self._generate_completion(chat_prompt, sampler, gen_params["max_tokens"], request.model)

            except Exception as e:
                logger.error(f"Error in chat completion: {str(e)}")
                logger.exception("Full traceback:")
                raise HTTPException(400, detail=str(e))

        @app.get("/v1/models")
        async def list_models():
            return {
                "data": [{
                    "id": server_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx-org",
                    "permission": [],
                    "root": server_model_name,
                    "parent": None
                }]
            }

        return app

    def serve(self, model_identifier, port=8000):
        """Serve the model via API"""
        app = self.create_api_app(model_identifier)
        uvicorn.run(app, host="0.0.0.0", port=port)

def select_models(model_choices):
    selected = [False] * len(model_choices)
    idx = 0
    window_size = os.get_terminal_size().lines - 5

    while True:
        print("\033[H\033[J", end="")
        print(
            "❯ llm mlx import-models\nAvailable models (↑/↓ to navigate, SPACE to select, ENTER to confirm, Ctrl+C to quit):"
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
            print("\nImport is cancelled. Do nothing.")
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
