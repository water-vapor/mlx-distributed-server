# Multi-node LLM Inference with Macs

This repository contains code and guides for setting up multi-node inference (tested with two Macs) using only Apple's official `mlx` and `mlx-lm` libraries.

## Comparison with [`exo`](https://github.com/water-vapor/exo)

### Advantages
- Achieves 14 tokens/s on Deepseek-R1-0528 (8bit) and 24 tokens/s on Kimi-K2 (4bit), compared to `exo`'s 4 tokens/s on the same hardware and connection
- The [original exo repository](https://github.com/exo-explore/exo) is no longer maintained

### Disadvantages
- Relies on `mlx` model's `pipeline` implementation, currently only available for deepseek architecture (supports deepseek models and kimi-k2)
- Specialized for distributed inference on Apple Silicon Macs with no cross-platform compatibility
- Current codebase is script-based, requiring manual operations and lacking a GUI

## Hardware Setup

You need two Macs connected via Thunderbolt. Additional requirements are detailed [here](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html#setting-up-remote-hosts). Key requirements:
- Bidirectional password-less SSH connections
- Static IP addresses for the Thunderbolt interface (check with `ifconfig bridge0`)

## Weight Preparation

Note: Third-party download tools may filter out important files. Ensure all components are downloaded.

### DeepSeek
Either download the 8bit version from `mlx-community` or convert from the official DeepSeek repository using `mlx-lm`. For models like `R1-0528` without pre-converted weights:
```bash
mlx_lm.convert --hf-path /<path_to>/deepseek-ai--DeepSeek-R1-0528 -q --q-bits 8 --mlx-path <your_desired_output_dir>
```

### Kimi K2
Download the 4bit version from `mlx-community`. Watch for missing files like `tiktoken.model` and `chat_template.jinja`. The `mlx-community` version truncates tool-use tokens from the tokenizer config - restore these manually if you need tool calling:
1. Add the 4 tool tokens in `tokenizer_config.json` from the original K2 repository to the downloaded mlx-community 4bit version
2. Replace `chat_template.jinja` with the `chat_template` content from the original `tokenizer_config.json`

## Software Environment Setup

- Install the latest `mlx` (0.26.0 or newer) from PyPI and the master branch of `mlx-lm`
- Install `tiktoken` and `blobfile` for Kimi-K2 support
- Replicate all scripts, Python environment, and model weights on both machines with identical paths
- Skip `mlx.distributed_config`, it has bugs. Use the configuration below instead

## Connection Configuration

Create a `tb-ring.json` file with your Thunderbolt interface IPs:
```json
[
    {
        "ssh": "192.168.100.1", 
        "ips": ["192.168.100.1"]
    },
    {
        "ssh": "192.168.100.2", 
        "ips": ["192.168.100.2"]
    }
]
```

## One-Time Generation Script

The `pipeline_generate_local.py` script extends `mlx-lm` examples with local model path support and random state synchronization (preventing node divergence). Usage:
```bash
mlx.launch --hostfile tb-ring.json \
    --env "PATH=/<path_to_miniconda>/miniconda/base/envs/mlxdist/bin:$PATH" \
    --env "PYTHONPATH=/<path_to_miniconda>/miniconda/base/envs/mlxdist/lib/python3.11/site-packages" \
    pipeline_generate_local.py \
    --model /<path_to_local_model>/mlx-community--Kimi-K2-Instruct-4bit \
    -m 16384 \
    --temp 0.6 \
    --prompt "tell me about the hardest math problem you know"
```
This script reloads the model each time (slow) and lacks multi-turn conversation support - use it for debugging only.

## Server Script

The `distributed_server.py` provides OpenAI-compatible APIs, derived from `mlx-lm`'s `server.py`. Features include multi-turn chat, random state synchronization, distributed KV cache, and tool-use support. Start with `run_distributed_server.sh` after updating paths:
```bash
sh run_distributed_server.sh <path/to/model> --temp 0.0 --max-tokens 16384
```

## Frontend Options

### Simple CLI Chat Client
`interactive_chat.py` provides fast command-line server access.

### Web-Based Chat Client
Install `open-webui` and follow step 2 in [this guide](https://docs.openwebui.com/getting-started/quick-start/starting-with-openai/). Use the URL format `http://xxx.xxx.xxx:8080/v1` to avoid network errors. Many bloated LLM chat frontends don't work with custom endpoints. While `open-webui` renders thinking traces, math formulas, markdown, and code blocks properly, it shows notably slower tokens/s compared to the CLI interface.

## Additional Features

`test_tool_calling.py` demonstrates Kimi K2's function calling capabilities

For convenience, sync Python files between machines with:
```bash
SOURCE_DIR="$(pwd)"
TARGET_HOST="<other_machine_host_name>"
rsync -av --no-dirs --include="*.py" --exclude="*" "$SOURCE_DIR"/ "$TARGET_HOST:$SOURCE_DIR"/
```

## Acknowledgment

Code development was assisted by Claude Code. This README was not.