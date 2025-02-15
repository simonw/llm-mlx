import click
import json
import llm
from mlx_lm import load, stream_generate
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

disable_progress_bars()

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
        if model_path not in models:
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
    for model_path in json.loads(_ensure_models_file().read_text()):
        register(MlxModel(model_path))


class MlxModel(llm.Model):
    can_stream = True

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
        
        # Always use stream_generate() because generate() in mlx_lm calls it under the hood
        for chunk in stream_generate(model, tokenizer, chat_prompt):
            yield chunk.text
        response.set_usage(input=chunk.prompt_tokens, output=chunk.generation_tokens)
        response.response_json = {
            "prompt_tps": chunk.prompt_tps,
            "generation_tps": chunk.generation_tps,
            "peak_memory": chunk.peak_memory,
            "finish_reason": chunk.finish_reason,
        }

