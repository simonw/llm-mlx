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


def test_harmony_detection():
    """Test that Harmony template detection works correctly."""

    # Mock tokenizer with Harmony template
    class MockHarmonyTokenizer:
        chat_template = """
        {% if message.thinking %}
        <|channel|>analysis<|message|>{{ message.thinking }}<|end|>
        {% endif %}
        <|channel|>final<|message|>{{ message.content }}<|end|>
        """

    # Mock tokenizer without Harmony template
    class MockNormalTokenizer:
        chat_template = "{{ message.content }}"

    # Mock tokenizer with no template
    class MockNoTemplate:
        pass

    assert llm_mlx._tokenizer_uses_harmony(MockHarmonyTokenizer()) is True
    assert llm_mlx._tokenizer_uses_harmony(MockNormalTokenizer()) is False
    assert llm_mlx._tokenizer_uses_harmony(MockNoTemplate()) is False


def test_split_harmony_message():
    """Test that Harmony message splitting works correctly."""

    harmony_text = """<|channel|>analysis<|message|>
Let me think about this problem step by step.
First, I need to understand what the user is asking.
<|end|>
<|channel|>final<|message|>
Here is my final answer to your question.
<|end|>"""

    analysis, final = llm_mlx._split_harmony_message(harmony_text)
    assert "Let me think about this problem step by step" in analysis
    assert "Here is my final answer" in final
    assert "<|channel|>" not in analysis
    assert "<|channel|>" not in final
