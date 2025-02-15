import llm_mlx


def test_basic_prompt():
    # This model is just 75MB
    # https://huggingface.co/mlx-community/SmolLM-135M-Instruct-4bit/tree/main
    tiny_model = "mlx-community/SmolLM-135M-Instruct-4bit"
    model = llm_mlx.MlxModel(tiny_model)
    assert model.prompt("hi").text()
