#!/bin/bash

# Run distributed MLX server with KV Cache and Tool Calling support V5
# Usage: ./run_distributed_server_v5.sh [model_path] [additional_args...]
# 
# Example with custom sampling parameters:
# ./run_distributed_server_v5.sh /path/to/model --temp 0.7 --max-tokens 1024 --top-p 0.9

set -e  # Exit on error

# Parse arguments
MODEL_PATH="${1:-/<path_to_local_model>/mlx-community--Kimi-K2-Instruct-4bit}"
shift || true  # Remove first argument (model path)
EXTRA_ARGS="$@"  # Capture remaining arguments

# Conda environment
CONDA_ENV="<name_of_your_conda_env>"
CONDA_BASE="/<path_to_miniconda>/miniconda/base"

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Get Python version dynamically
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python $PYTHON_VERSION"

# Build the Python path
PYTHON_SITE_PACKAGES="$CONDA_BASE/envs/$CONDA_ENV/lib/python$PYTHON_VERSION/site-packages"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Usage: $0 [model_path]"
    exit 1
fi

# Check if hostfile exists
if [ ! -f "tb-ring.json" ]; then
    echo "Error: Hostfile tb-ring.json not found"
    echo "Please create a hostfile with the format:"
    echo '[{"ssh": "hostname1", "ips": ["ip1"]}, {"ssh": "hostname2", "ips": ["ip2"]}]'
    exit 1
fi

echo "=========================================="
echo "Distributed Streaming Server V5 with Tool Calling"
echo "Features:"
echo " - KV caching for multi-turn conversations"
echo " - Cache synchronization between ranks"
echo " - Smart cache management (trim/reset)"
echo " - Queue-based token streaming"
echo " - OpenAI-compatible API endpoints"
echo " - Configurable sampling parameters via CLI"
echo " - Tool calling support for function models"
echo "Model: $MODEL_PATH"
echo "Extra args: $EXTRA_ARGS"
echo "Using hostfile: tb-ring.json"
echo "=========================================="

# Run the distributed streaming server v5
mlx.launch --hostfile tb-ring.json \
    --env "PATH=$CONDA_BASE/envs/$CONDA_ENV/bin:$PATH" \
    --env "PYTHONPATH=$PYTHON_SITE_PACKAGES" \
    --verbose \
    distributed_server.py \
    --model "$MODEL_PATH" \
    --server-host 0.0.0.0 \
    --server-port 8080 \
    $EXTRA_ARGS