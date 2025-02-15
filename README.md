# llm-mlx

[![PyPI](https://img.shields.io/pypi/v/llm-mlx.svg)](https://pypi.org/project/llm-mlx/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-mlx?include_prereleases&label=changelog)](https://github.com/simonw/llm-mlx/releases)
[![Tests](https://github.com/simonw/llm-mlx/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-mlx/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-mlx/blob/main/LICENSE)

Support for [MLX](https://github.com/ml-explore/mlx) models in [LLM](https://llm.datasette.io/)

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/). This plugin likely only works on macOS.
```bash
llm install llm-mlx
```
## Usage

To install an MLX model from Hugging Face, use the `llm mlx download-model` command:

```bash
llm mlx download-model mlx-community/Llama-3.2-3B-Instruct-4bit
```
Then run prompts like this:
```bash
llm -m mlx-community/Llama-3.2-3B-Instruct-4bit 'Capital of France?' -s 'you are a pelican'
```
The [mlx-community](https://huggingface.co/mlx-community) organization is a useful source for compatible models.

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
