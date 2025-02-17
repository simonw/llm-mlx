import click
import json
import llm
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from pydantic import Field, field_validator
from typing import Optional
import os
from pathlib import Path  # local import to avoid adding to global namespace
import sys

disable_progress_bars()

# These defaults copied from llama.cpp
# https://github.com/ggml-org/llama.cpp/blob/68ff663a04ed92044a9937bcae353e9d9733f9cd/examples/main/README.md#generation-flags
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
        "Import MLX models from the Hugging Face cache."
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
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
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

        # Always use stream_generate() because generate() in mlx_lm calls it under the hood
        for chunk in stream_generate(
            model,
            tokenizer,
            chat_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
        ):
            yield chunk.text
        response.set_usage(input=chunk.prompt_tokens, output=chunk.generation_tokens)
        response.response_json = {
            "prompt_tps": chunk.prompt_tps,
            "generation_tps": chunk.generation_tps,
            "peak_memory": chunk.peak_memory,
            "finish_reason": chunk.finish_reason,
        }


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
