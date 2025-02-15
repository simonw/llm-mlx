import click
import json
import llm
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from pydantic import Field, field_validator
from typing import Optional

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
