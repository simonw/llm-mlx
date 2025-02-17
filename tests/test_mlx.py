import json
import llm_mlx
import pytest
from fastapi.testclient import TestClient


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

@pytest.fixture
def server():
    model = llm_mlx.MlxModel(TINY_MODEL)
    model._load()
    app = model.create_api_app("test-model")
    return TestClient(app)

def test_chat_completion(server):
    response = server.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 100
        },
    )
    assert response.status_code == 200
    data = response.json()
    
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"

def test_streaming_response(server):
    response = server.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 100
        },
    )
    assert response.status_code == 200
    
    lines = [
        line for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    
    for line in lines:
        event = json.loads(line.split("data: ")[1])
        assert event["object"] == "chat.completion.chunk"
        assert len(event["choices"]) == 1
        assert "delta" in event["choices"][0]

def test_error_handling(server):
    # Test invalid messages format
    response = server.post(
        "/v1/chat/completions",
        json={"messages": "invalid"},
    )
    assert response.status_code == 422  # FastAPI validation error
    
    # Test invalid temperature
    response = server.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 3.0,
        },
    )
    assert response.status_code == 422  # FastAPI validation error

def test_model_listing(server):
    response = server.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"