# Qwen3 Coder Deployment

This is a guide to deploy `mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit` with `https://github.com/QwenLM/qwen-code` for local vibe coding on a single Mac. The Python files are based on the `server.py` implementation from `mlx-lm@08e4dd2fc5dc31bc78fe1eb65cc4d81890595040`, and the `qwen3_xml_tool_parser.py` from the official weights repository. They are modified to support Qwen3-specific tool calls.

## Software Environment

- `mlx==0.26.5` and `mlx-lm` in the master branch
- Follow the installation guide at https://github.com/QwenLM/qwen-code

## Usage

```bash
python qwen3_coder_server.py --model /<some_path>/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-480B-A35B-Instruct-4bit/snapshots/ca8dbf41071f579fbe3260f20bbe1ab896f79031/ --host 0.0.0.0 --port 8080
```

Then assign the environment variables for Qwen Code. You can create a file and source it for convenience:

```bash
export OPENAI_API_KEY="1"
export OPENAI_BASE_URL="http://0.0.0.0:8080/v1"
export OPENAI_MODEL="mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"
```

```bash
source env.sh
```

Then start Qwen Coder with:

```bash
qwen
```