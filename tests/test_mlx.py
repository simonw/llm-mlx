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
