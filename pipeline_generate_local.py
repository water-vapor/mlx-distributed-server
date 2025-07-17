"""
Pipeline generation with local model path support and fixed random synchronization.

Run with:
```
mlx.launch \
  --hostfile /path/to/hosts.json \
  pipeline_generate_local_fixed.py \
  --model /path/to/local/model \
  --prompt "hello world" \
  --temp 0.7
```
"""

import argparse
import json
import os
import resource
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from mlx_lm import load, stream_generate
from mlx_lm.utils import load_model, load_tokenizer
from mlx_lm.sample_utils import make_sampler

# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))


def download(repo: str, allow_patterns: list[str]) -> Path:
    """Download model from HuggingFace hub."""
    return Path(
        snapshot_download(
            repo,
            allow_patterns=allow_patterns,
        )
    )


def sync_random_seed(group, seed=None):
    """Synchronize random seed across all ranks."""
    rank = group.rank()
    
    # Generate seed on rank 0
    if rank == 0:
        if seed is None:
            import random
            seed = random.randint(0, 2**31 - 1)
    
    # Create array - rank 0 has the seed, others have 0
    if rank == 0:
        seed_array = mx.array([seed], dtype=mx.int32)
    else:
        seed_array = mx.array([0], dtype=mx.int32)
    
    # Sum across all ranks (only rank 0 contributes non-zero)
    seed_array = mx.distributed.all_sum(seed_array)
    final_seed = int(seed_array.item())
    mx.random.seed(final_seed)
    
    if rank == 0:
        print(f"Random seed synchronized: {final_seed}")
    
    return final_seed


def shard_and_load(model_path: str):
    """Load model with support for both local paths and HF repos."""
    group = mx.distributed.init()
    rank = group.rank()
    
    # Check if model_path is a local directory
    if os.path.exists(model_path):
        # Local path - use directly
        model_path = Path(model_path)
        print(f"[Rank {rank}] Using local model at: {model_path}")
    else:
        # Assume it's a HuggingFace repo
        print(f"[Rank {rank}] Downloading from HuggingFace: {model_path}")
        # Get model path with metadata files
        model_path = download(
            model_path,
            allow_patterns=["*"], # allow all files to prevent missing components
        )

    # Lazy load and shard model to figure out which weights we need
    model, config = load_model(model_path, lazy=True, strict=False)
    model.model.pipeline(group)

    # For local models, we assume all weights are already present
    # For HF models, we need to download specific weight shards
    if not os.path.exists(args.model):
        # Figure out which files we need for the local shard
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as fid:
                weight_index = json.load(fid)["weight_map"]

            local_files = set()
            for k, _ in tree_flatten(model.parameters()):
                if k in weight_index:
                    local_files.add(weight_index[k])

            # Download weights for local shard
            print(f"[Rank {rank}] Downloading weight shards: {local_files}")
            download(args.model, allow_patterns=local_files)

    # Load the model weights
    tokenizer = load_tokenizer(
        model_path, 
        {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id", None),
    )
    model, config = load_model(model_path, lazy=True, strict=False)
    model.model.pipeline(group)
    mx.eval(model.parameters())

    # Synchronize processes before generation
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    
    return model, tokenizer, group


def rprint(*args, **kwargs):
    """Print only from rank 0."""
    group = mx.distributed.init()
    if group.rank() == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM pipelined inference with local model support")
    parser.add_argument(
        "--model",
        required=True,
        help="Local path to model directory or HF repo",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Write a quicksort in C++.",
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=16384,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    args = parser.parse_args()

    # Read prompt from stdin if requested
    if args.prompt == "-":
        import sys
        args.prompt = sys.stdin.read().strip()

    model, tokenizer, group = shard_and_load(args.model)

    # Apply chat template if the model supports it
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    else:
        prompt = args.prompt

    # Synchronize random seed if using temperature > 0
    if args.temp > 0:
        sync_random_seed(group, args.seed)
        rprint(f"Using temperature={args.temp} with synchronized random seed")
    
    # Create sampler from temperature - now safe to use!
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    
    # Generate response
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=sampler
    ):
        rprint(response.text, end="", flush=True)

    rprint()
    rprint("=" * 10)
    rprint(
        f"Prompt: {response.prompt_tokens} tokens, "
        f"{response.prompt_tps:.3f} tokens-per-sec"
    )
    rprint(
        f"Generation: {response.generation_tokens} tokens, "
        f"{response.generation_tps:.3f} tokens-per-sec"
    )
    rprint(f"Peak memory: {response.peak_memory:.3f} GB")