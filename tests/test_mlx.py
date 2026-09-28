import json

import click
from click.testing import CliRunner

import llm_mlx

TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


def test_basic_prompt():
    # This model is just 75MB
    # https://huggingface.co/mlx-community/SmolLM-135M-Instruct-4bit/tree/main
    model = llm_mlx.MlxModel(TINY_MODEL)
    response = model.prompt("hi")
    assert response.text()
    # Should have expected detail keys:
    details = response.json()
    assert {"prompt_tps", "generation_tps", "peak_memory", "finish_reason"} == set(
        details.keys()
    )


def test_model_options():
    model = llm_mlx.MlxModel(TINY_MODEL)
    response = model.prompt("hi", temperature=0.5, top_p=0.9, min_p=0.1, max_tokens=5)
    output = response.text()
    assert len(output) < 200
    options = response.prompt.options
    assert options.max_tokens == 5
    assert options.temperature == 0.5
    assert options.top_p == 0.9
    assert options.min_p == 0.1
    assert response.json()["finish_reason"] == "length"


def test_manage_models_respects_hf_hub_cache(monkeypatch, tmp_path):
    fake_cache = tmp_path / "fake-hub-cache"
    model_dir = (
        fake_cache
        / "models--mlx-community--SmolLM-135M-Instruct-4bit"
        / "snapshots"
        / "abc123"
    )
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

    # HF_HUB_CACHE is computed at import time, so patch the attribute
    # llm_mlx actually reads rather than the environment variable.
    monkeypatch.setattr(llm_mlx, "HF_HUB_CACHE", str(fake_cache))

    llm_user_dir = tmp_path / "llm-user-dir"
    monkeypatch.setenv("LLM_USER_PATH", str(llm_user_dir))

    # Auto-select every discovered model instead of driving the
    # interactive terminal picker.
    monkeypatch.setattr(
        llm_mlx, "select_models", lambda model_choices: model_choices
    )

    cli = click.Group()
    llm_mlx.register_commands(cli)

    runner = CliRunner()
    result = runner.invoke(cli, ["mlx", "manage-models"])
    assert result.exit_code == 0, result.output

    models_file = llm_user_dir / "llm-mlx.json"
    registered = json.loads(models_file.read_text())
    assert "mlx-community/SmolLM-135M-Instruct-4bit" in registered
