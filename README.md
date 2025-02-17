# llm-mlx

[![PyPI](https://img.shields.io/pypi/v/llm-mlx.svg)](https://pypi.org/project/llm-mlx/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-mlx?include_prereleases&label=changelog)](https://github.com/simonw/llm-mlx/releases)
[![Tests](https://github.com/simonw/llm-mlx/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-mlx/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-mlx/blob/main/LICENSE)

Support for [MLX](https://github.com/ml-explore/mlx) models in [LLM](https://llm.datasette.io/).

Read my blog for [background on this project](https://simonwillison.net/2025/Feb/15/llm-mlx/).

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/). This plugin likely only works on macOS.
```bash
llm install llm-mlx
```
This plugin depends on [sentencepiece](https://pypi.org/project/sentencepiece/) which does not yet publish a binary wheel for Python 3.13. You will find this plugin easier to run on Python 3.12 or lower. One way to install a version of LLM that uses Python 3.12 is like this, using [uv](https://github.com/astral-sh/uv):

```bash
uv tool install llm --python 3.12
```
See [issue #7](https://github.com/simonw/llm-mlx/issues/7) for more on this.

## Usage

To install an MLX model from Hugging Face, use the `llm mlx download-model` command. This example downloads 1.8GB of model weights from [mlx-community/Llama-3.2-3B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit):

```bash
llm mlx download-model mlx-community/Llama-3.2-3B-Instruct-4bit
```
Then run prompts like this:
```bash
llm -m mlx-community/Llama-3.2-3B-Instruct-4bit 'Capital of France?' -s 'you are a pelican'
```
The [mlx-community](https://huggingface.co/mlx-community) organization is a useful source for compatible models.

### Models to try

The following models all work well with this plugin:

- `mlx-community/Qwen2.5-0.5B-Instruct-4bit` - [278MB](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit)
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` - [4.08GB](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit)
-  `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` — [13.26 GB](https://huggingface.co/mlx-community/Mistral-Small-24B-Instruct-2501-4bit)
- `mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit` - [18.5GB](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit)
- `mlx-community/Llama-3.3-70B-Instruct-4bit` - [40GB](https://huggingface.co/mlx-community/Llama-3.3-70B-Instruct-4bit)

### Model options

MLX models can use the following model options:

- `-o max_tokens INTEGER`: Maximum number of tokens to generate in the completion (defaults to 1024)
- `-o unlimited 1`: Generate an unlimited number of tokens in the completion
- `-o temperature FLOAT`: Sampling temperature (defaults to 0.8)
- `-o top_p FLOAT`: Sampling top-p (defaults to 0.9)
- `-o min_p FLOAT`: Sampling min-p (defaults to 0.1)
- `-o min_tokens_to_keep INT`: Minimum tokens to keep for min-p sampling (defaults to 1)
- `-o seed INT`: Random number seed to use

For example:
```bash
llm -m mlx-community/Llama-3.2-3B-Instruct-4bit 'Joke about pelicans' -o max_tokens 60 -o temperature 1.0
```

## Using models from Python

You can use this plugin in Python like this:

```python
from llm_mlx import MlxModel
model = MlxModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
print(model.prompt("hi").text())
# Outputs: How can I assist you today?
```
Using `MlxModel` directly in this way avoids needing to first use the `download-model` command.

If you have already registered models with that command you can use them like this instead:
```python
import llm
model = llm.get_model("mlx-community/Llama-3.2-3B-Instruct-4bit")
print(model.prompt("hi").text())
```
The [LLM Python API documentation](https://llm.datasette.io/en/stable/python-api.html) has more details on how to use LLM models.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-mlx
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
