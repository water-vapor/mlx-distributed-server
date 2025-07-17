"""
Distributed MLX Server with Streaming Support, KV Cache, and Tool Calling (V5.2)

This version refactors v5.1 for cleaner code organization, reduced duplication,
and improved maintainability while preserving all functionality.
"""

import argparse
import json
import logging
import os
import resource
import time
import uuid
import threading
import queue
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Dict, Any, List, Union, NamedTuple, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm import stream_generate
from mlx_lm.utils import load_model, load_tokenizer, common_prefix_len
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache

# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

# Constants
class MessageType(IntEnum):
    IDLE = 0
    GENERATE = 1
    SHUTDOWN = 2




class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


@dataclass
class PromptCache:
    """Cache structure for storing KV cache and associated metadata"""
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str], Optional[str]] = ("", None, None)
    tokens: List[int] = field(default_factory=list)


def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    eos_token_id: Union[int, None],
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.
    """
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=0)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids):] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(s1: List[int], s2: List[int]) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


@dataclass
class GenerationRequest:
    prompt: str
    params: Dict[str, Any]
    stop_words: List[str]
    request_id: str
    created: int
    stream: bool
    conversation_id: Optional[str] = None  # For cache management


@dataclass
class StreamToken:
    text: str
    is_final: bool = False
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class ToolCallState:
    """Manages tool calling state during generation"""
    in_tool_call: bool = False
    tool_calls: List[Any] = field(default_factory=list)
    # Kimi-specific state
    tool_func_id: str = ""
    tool_arguments: str = ""
    in_tool_arguments: bool = False
    # Generic state
    tool_text: str = ""


def parse_kimi_tool_call(func_id: str, arguments: str) -> Dict[str, Any]:
    """Parse Kimi-specific tool call function ID and arguments into OpenAI format"""
    try:
        # Parse function ID in format: functions.function_name:index
        if func_id.startswith("functions.") and ":" in func_id:
            func_name = func_id.split("functions.")[1].split(":")[0]
        else:
            func_name = "unknown"
        
        return {
            "function": {
                "name": func_name,
                "arguments": arguments,
            },
            "type": "function",
            "id": None,
        }
    except Exception as e:
        # Fallback for malformed tool calls
        return {
            "function": {
                "name": "unknown",
                "arguments": f"func_id: {func_id}, arguments: {arguments}",
            },
            "type": "function",
            "id": None,
        }


def parse_generic_tool_call(tool_text: str) -> Dict[str, Any]:
    """Parse generic tool call text (JSON) into OpenAI format"""
    try:
        tool_call = json.loads(tool_text.strip())
        return {
            "function": {
                "name": tool_call.get("name", None),
                "arguments": json.dumps(tool_call.get("arguments", "")),
            },
            "type": "function",
            "id": None,
        }
    except json.JSONDecodeError:
        # Fallback for malformed tool calls
        return {
            "function": {
                "name": "unknown",
                "arguments": tool_text,
            },
            "type": "function",
            "id": None,
        }


def extract_generation_params(data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate generation parameters from request data"""
    return {
        'temperature': float(data.get('temperature', defaults['temperature'])),
        'top_p': float(data.get('top_p', defaults['top_p'])),
        'top_k': int(data.get('top_k', defaults['top_k'])),
        'min_p': float(data.get('min_p', defaults['min_p'])),
        'max_tokens': int(data.get('max_tokens', defaults['max_tokens'])),
        'repetition_penalty': float(data.get('repetition_penalty', defaults['repetition_penalty'])),
        'repetition_context_size': int(data.get('repetition_context_size', defaults['repetition_context_size'])),
    }




class DistributedModel:
    """Manages the distributed model and synchronization with KV caching"""
    
    def __init__(self, model_path: str, max_cache_size: Optional[int] = None):
        self.group = mx.distributed.init()
        self.rank = self.group.rank()
        self.size = self.group.size()
        
        # Logging setup - MUST come before any logging calls
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Rank {self.rank}] %(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing rank {self.rank} of {self.size}")
        
        # Store original model path and extract model name
        self.model_path = model_path
        self.model_name = self._extract_model_name(model_path)
        self.model_key = (model_path, None, None)  # (model_path, adapter_path, draft_model_path)
        
        # Load and shard model
        self.model, self.tokenizer = self._load_model(model_path)
        self.logger.info(f"Model '{self.model_name}' loaded successfully")
        
        # Detect model type and tool calling support
        self.model_type = self._detect_model_type()
        self._detect_tool_calling_tokens()
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Tool calling support: {self.has_tool_calling}")
        
        # Initialize KV cache
        self.prompt_cache = PromptCache()
        self.max_cache_size = max_cache_size
        self.logger.info(f"KV cache initialized with max_size={max_cache_size}")
    
    def _extract_model_name(self, model_path: str) -> str:
        """Extract model name from the path - just use folder name"""
        model_path = Path(model_path)
        return model_path.name
    
    def _load_model(self, model_path: str):
        """Load and shard the model across ranks"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        self.logger.info(f"Loading model from: {model_path}")
        
        # Lazy load and shard model
        model, config = load_model(model_path, lazy=True, strict=False)
        model.model.pipeline(self.group)
        
        # Load tokenizer
        tokenizer = load_tokenizer(
            model_path, 
            {"trust_remote_code": True},
            eos_token_ids=config.get("eos_token_id", None),
        )
        
        # Reload model with weights
        model, config = load_model(model_path, lazy=True, strict=False)
        model.model.pipeline(self.group)
        mx.eval(model.parameters())
        
        # Synchronize after loading
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
        
        return model, tokenizer
    
    def _detect_model_type(self) -> str:
        """Detect model type from config.json"""
        config_path = Path(self.model_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('model_type', 'unknown')
            except Exception as e:
                self.logger.warning(f"Failed to read model config: {e}")
        return 'unknown'
    
    def _detect_tool_calling_tokens(self):
        """Detect tool calling tokens based on model type and tokenizer vocabulary"""
        self.has_tool_calling = False
        self.tool_call_start = None
        self.tool_call_end = None
        self.kimi_tool_tokens = {}
        
        try:
            if self.model_type == "kimi_k2":
                # Kimi K2 specific tokens
                kimi_tokens = {
                    "section_begin": "<|tool_calls_section_begin|>",
                    "section_end": "<|tool_calls_section_end|>",
                    "call_begin": "<|tool_call_begin|>",
                    "call_end": "<|tool_call_end|>",
                    "arg_begin": "<|tool_call_argument_begin|>"
                }
                
                # For now, assume Kimi K2 has tool calling
                # In production, you'd check if tokens exist in vocabulary
                self.has_tool_calling = True
                self.kimi_tool_tokens = kimi_tokens
                self.logger.debug(f"Found Kimi K2 tool calling tokens")
            else:
                # Generic tool calling token detection (like mlx-lm server.py)
                TOOL_CALL_TOKENS = [("<tool_call>", "</tool_call>")]
                
                # Check if tokenizer has tool in chat template
                if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template and '"tool"' in str(self.tokenizer.chat_template):
                    for start_token, end_token in TOOL_CALL_TOKENS:
                        # For simplicity, assume generic models use <tool_call> tokens
                        self.has_tool_calling = True
                        self.tool_call_start = start_token
                        self.tool_call_end = end_token
                        self.logger.debug(f"Assuming generic tool calling tokens: {start_token}, {end_token}")
                        break
        except Exception as e:
            self.logger.warning(f"Error detecting tool calling tokens: {e}")
            self.has_tool_calling = False
    
    def reset_prompt_cache(self, prompt_tokens: List[int]):
        """Reset the prompt cache and associated state"""
        self.logger.debug(f"Resetting KV cache")
        self.prompt_cache.model_key = self.model_key
        self.prompt_cache.cache = make_prompt_cache(self.model, self.max_cache_size)
        self.prompt_cache.tokens = list(prompt_tokens)
    
    def get_prompt_cache(self, prompt_tokens: List[int]) -> List[int]:
        """
        Determines the portion of the prompt that needs processing by comparing
        it to the cached prompt and attempting to reuse the common prefix.
        """
        cache_len = len(self.prompt_cache.tokens)
        prompt_len = len(prompt_tokens)
        com_prefix_len = common_prefix_len(self.prompt_cache.tokens, prompt_tokens)
        
        # Leave at least one token in the prompt
        com_prefix_len = min(com_prefix_len, len(prompt_tokens) - 1)
        
        # Condition 1: Model changed or no common prefix at all. Reset cache.
        if (
            self.prompt_cache.model_key != self.model_key
            or com_prefix_len == 0
        ):
            self.reset_prompt_cache(prompt_tokens)
        # Condition 2: Common prefix exists and matches cache length. Process suffix.
        elif com_prefix_len == cache_len:
            self.logger.debug(
                f"Cache is prefix of prompt (cache_len: {cache_len}, prompt_len: {prompt_len}). Processing suffix."
            )
            prompt_tokens = prompt_tokens[com_prefix_len:]
            self.prompt_cache.tokens.extend(prompt_tokens)
        # Condition 3: Common prefix exists but is shorter than cache length. Attempt trim.
        elif com_prefix_len < cache_len:
            self.logger.debug(
                f"Common prefix ({com_prefix_len}) shorter than cache ({cache_len}). Attempting trim."
            )
            if can_trim_prompt_cache(self.prompt_cache.cache):
                num_to_trim = cache_len - com_prefix_len
                self.logger.debug(f"Trimming {num_to_trim} tokens from cache.")
                trim_prompt_cache(self.prompt_cache.cache, num_to_trim)
                self.prompt_cache.tokens = self.prompt_cache.tokens[:com_prefix_len]
                # Process the suffix
                prompt_tokens = prompt_tokens[com_prefix_len:]
                self.prompt_cache.tokens.extend(prompt_tokens)
            else:
                self.logger.debug("Cache cannot be trimmed. Resetting.")
                self.reset_prompt_cache(prompt_tokens)
        else:
            self.logger.error(
                f"Unexpected cache state: com_prefix_len ({com_prefix_len}) > cache_len ({cache_len}). Resetting cache."
            )
            self.reset_prompt_cache(prompt_tokens)
        
        self.logger.debug(f"Returning {len(prompt_tokens)} tokens for processing.")
        return prompt_tokens
    
    def sync_random_seed(self, seed: Optional[int] = None) -> int:
        """Synchronize random seed across all ranks"""
        # Generate seed on rank 0
        if self.rank == 0:
            if seed is None:
                import random
                seed = random.randint(0, 2**31 - 1)
            self.logger.info(f"Rank 0 using seed: {seed}")
        
        # Create array - rank 0 has the seed, others have 0
        if self.rank == 0:
            seed_array = mx.array([seed], dtype=mx.int32)
        else:
            seed_array = mx.array([0], dtype=mx.int32)
        
        # Sum across all ranks (only rank 0 contributes non-zero)
        seed_array = mx.distributed.all_sum(seed_array)
        final_seed = int(seed_array.item())
        
        # Set the seed on all ranks
        mx.random.seed(final_seed)
        
        if self.rank == 0:
            self.logger.info(f"Random seed synchronized: {final_seed}")
        
        return final_seed
    
    def _broadcast_data(self, data: Optional[List[int]], dtype=mx.int32) -> mx.array:
        """Generic method to broadcast data from rank 0 to all ranks"""
        if self.rank == 0:
            if data is None:
                raise ValueError("Rank 0 must have data to broadcast")
            # Send length first
            length = mx.array([len(data)], dtype=mx.int32)
            data_array = mx.array(data, dtype=dtype)
        else:
            # Prepare to receive
            length = mx.array([0], dtype=mx.int32)
            data_array = None
        
        # Broadcast length
        length = mx.distributed.all_sum(length)
        length_val = int(length.item())
        
        if self.rank != 0:
            # Other ranks create buffer
            data_array = mx.zeros((length_val,), dtype=dtype)
        
        # Broadcast data using all_sum (since only rank 0 has non-zero values)
        data_array = mx.distributed.all_sum(data_array)
        
        return data_array
    
    def broadcast_prompt_tokens(self, prompt_tokens: Optional[List[int]]) -> List[int]:
        """Broadcast prompt tokens from rank 0 to all ranks"""
        return self._broadcast_data(prompt_tokens).tolist()
    
    def broadcast_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast all generation parameters from rank 0"""
        if self.rank == 0:
            # Convert parameters to integers for broadcasting
            temp_int = int(params['temperature'] * 1000000)
            top_p_int = int(params['top_p'] * 1000000)
            min_p_int = int(params['min_p'] * 1000000)
            repetition_penalty_int = int(params.get('repetition_penalty', 1.0) * 1000000)
            
            param_array = mx.array([
                temp_int, top_p_int, params['top_k'], min_p_int, 
                params['max_tokens'], repetition_penalty_int,
                params.get('repetition_context_size', 20)
            ], dtype=mx.int32)
        else:
            param_array = mx.array([0, 0, 0, 0, 0, 0, 0], dtype=mx.int32)
        
        # Broadcast parameters
        param_array = mx.distributed.all_sum(param_array)
        params_list = param_array.tolist()
        
        # Unpack and convert back
        return {
            'temperature': params_list[0] / 1000000.0,
            'top_p': params_list[1] / 1000000.0,
            'top_k': params_list[2],
            'min_p': params_list[3] / 1000000.0,
            'max_tokens': params_list[4],
            'repetition_penalty': params_list[5] / 1000000.0,
            'repetition_context_size': params_list[6]
        }
    
    def broadcast_stop_sequences(self, stop_words: List[str]) -> List[List[int]]:
        """Broadcast stop sequences to all ranks"""
        if self.rank == 0:
            # Encode stop words
            stop_id_sequences = [
                self.tokenizer.encode(stop_word, add_special_tokens=False)
                for stop_word in stop_words
            ]
            
            # Flatten all sequences and track lengths
            all_ids = []
            lengths = []
            for seq in stop_id_sequences:
                all_ids.extend(seq)
                lengths.append(len(seq))
            
            # Create arrays
            num_sequences = mx.array([len(stop_id_sequences)], dtype=mx.int32)
            lengths_array = mx.array(lengths if lengths else [0], dtype=mx.int32)
            ids_array = mx.array(all_ids if all_ids else [0], dtype=mx.int32)
        else:
            num_sequences = mx.array([0], dtype=mx.int32)
            lengths_array = None
            ids_array = None
        
        # Broadcast number of sequences
        num_sequences = mx.distributed.all_sum(num_sequences)
        num_seq = int(num_sequences.item())
        
        if num_seq == 0:
            return []
        
        # Prepare arrays on non-rank-0
        if self.rank != 0:
            lengths_array = mx.zeros((num_seq,), dtype=mx.int32)
        
        # Broadcast lengths
        lengths_array = mx.distributed.all_sum(lengths_array)
        lengths = lengths_array.tolist()
        
        # Prepare ids array on non-rank-0
        total_ids = sum(lengths)
        if self.rank != 0:
            ids_array = mx.zeros((total_ids,), dtype=mx.int32)
        
        # Broadcast ids
        ids_array = mx.distributed.all_sum(ids_array)
        all_ids = ids_array.tolist()
        
        # Reconstruct sequences
        stop_id_sequences = []
        idx = 0
        for length in lengths:
            stop_id_sequences.append(all_ids[idx:idx + length])
            idx += length
        
        return stop_id_sequences
    
    def broadcast_generated_tokens(self, generated_tokens: Optional[List[int]]) -> List[int]:
        """Broadcast generated tokens from rank 0 to all ranks for cache synchronization"""
        if self.rank == 0 and (generated_tokens is None or len(generated_tokens) == 0):
            # Broadcast empty signal
            length = mx.array([0], dtype=mx.int32)
            mx.distributed.all_sum(length)
            return []
            
        return self._broadcast_data(generated_tokens).tolist()
    
    def synchronize_message(self, message: MessageType) -> MessageType:
        """Synchronize a message type across all ranks"""
        msg_array = mx.array([float(message)], dtype=mx.float32)
        msg_array = mx.distributed.all_sum(msg_array)
        
        # Use majority vote for robustness
        return MessageType(int(msg_array.item() / self.size + 0.5))
    
    def generate_streaming(self, request: Optional[GenerationRequest], token_queue: Optional[queue.Queue] = None):
        """Generate tokens with streaming support using a queue for communication"""
        try:
            # Broadcast parameters
            if self.rank == 0:
                params = self.broadcast_generation_params(request.params)
            else:
                # Dummy params for broadcasting
                params = self.broadcast_generation_params({
                    'temperature': 0, 'top_p': 0, 'top_k': 0, 'min_p': 0,
                    'max_tokens': 0, 'repetition_penalty': 0, 'repetition_context_size': 0
                })
            
            # Synchronize random seed if using temperature > 0
            if params['temperature'] > 0:
                seed = request.params.get('seed', None) if self.rank == 0 else None
                self.sync_random_seed(seed)
            
            # Tokenize and broadcast prompt
            if self.rank == 0:
                prompt_tokens = self.tokenizer.encode(request.prompt)
                original_prompt_token_count = len(prompt_tokens)
            else:
                prompt_tokens = None
                original_prompt_token_count = 0
            
            prompt_tokens = self.broadcast_prompt_tokens(prompt_tokens)
            
            # Apply KV cache optimization on both ranks
            processed_prompt_tokens = self.get_prompt_cache(prompt_tokens)
            prompt = self.tokenizer.decode(processed_prompt_tokens)
            
            # Broadcast stop sequences
            if self.rank == 0:
                stop_id_sequences = self.broadcast_stop_sequences(request.stop_words)
            else:
                stop_id_sequences = self.broadcast_stop_sequences([])
            
            # Create sampler and logits processors
            sampler = make_sampler(
                temp=params['temperature'],
                top_p=params['top_p'],
                top_k=params['top_k'],
                min_p=params['min_p'],
            )
            
            logits_processors = make_logits_processors(
                repetition_penalty=params['repetition_penalty'],
                repetition_context_size=params['repetition_context_size'],
            )
            
            # Generate tokens
            tokens = []
            segment = ""
            finish_reason = "length"
            full_text = ""
            
            # Initialize tool call state
            tool_state = ToolCallState()
            
            # Log cache usage
            if self.rank == 0:
                cache_hit = len(processed_prompt_tokens) < len(prompt_tokens)
                self.logger.info(f"Cache usage: {len(prompt_tokens) - len(processed_prompt_tokens)}/{len(prompt_tokens)} tokens cached")
            
            for gen_response in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=params['max_tokens'],
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=self.prompt_cache.cache,  # Use the KV cache
            ):
                if self.rank == 0:
                    text = gen_response.text
                    token = gen_response.token
                    tokens.append(token)
                    
                    # Process token for tool calling
                    if self.has_tool_calling and self.model_type == "kimi_k2":
                        # Kimi K2 specific tool calling
                        if text == self.kimi_tool_tokens.get("section_begin", ""):
                            tool_state.in_tool_call = True
                        elif tool_state.in_tool_call:
                            if text == self.kimi_tool_tokens.get("section_end", ""):
                                if tool_state.tool_func_id.strip() and tool_state.tool_arguments.strip():
                                    tool_state.tool_calls.append((tool_state.tool_func_id.strip(), tool_state.tool_arguments.strip()))
                                tool_state.tool_func_id = ""
                                tool_state.tool_arguments = ""
                                tool_state.in_tool_call = False
                                tool_state.in_tool_arguments = False
                            elif text == self.kimi_tool_tokens.get("call_begin", ""):
                                tool_state.tool_func_id = ""
                                tool_state.tool_arguments = ""
                                tool_state.in_tool_arguments = False
                            elif text == self.kimi_tool_tokens.get("arg_begin", ""):
                                tool_state.in_tool_arguments = True
                            elif text == self.kimi_tool_tokens.get("call_end", ""):
                                if tool_state.tool_func_id.strip() and tool_state.tool_arguments.strip():
                                    tool_state.tool_calls.append((tool_state.tool_func_id.strip(), tool_state.tool_arguments.strip()))
                                tool_state.tool_func_id = ""
                                tool_state.tool_arguments = ""
                                tool_state.in_tool_arguments = False
                            else:
                                if tool_state.in_tool_arguments:
                                    tool_state.tool_arguments += text
                                else:
                                    tool_state.tool_func_id += text
                        else:
                            segment += text
                            full_text += text
                    elif self.has_tool_calling and self.tool_call_start and self.tool_call_end:
                        # Generic tool calling
                        if text == self.tool_call_start:
                            tool_state.in_tool_call = True
                        elif tool_state.in_tool_call:
                            if text == self.tool_call_end:
                                if tool_state.tool_text.strip():
                                    tool_state.tool_calls.append(tool_state.tool_text.strip())
                                tool_state.tool_text = ""
                                tool_state.in_tool_call = False
                            else:
                                tool_state.tool_text += text
                        else:
                            segment += text
                            full_text += text
                    else:
                        # No tool calling support
                        segment += text
                        full_text += text
                    
                    # Check stop conditions
                    stop_condition = stopping_criteria(
                        tokens, stop_id_sequences, self.tokenizer.eos_token_id
                    )
                    if stop_condition.stop_met:
                        finish_reason = "stop"
                        if stop_condition.trim_length:
                            # Remove stop sequence from output
                            stop_text = self.tokenizer.decode(
                                tokens[-stop_condition.trim_length:]
                            )
                            segment = segment[:-len(stop_text)]
                            full_text = full_text[:-len(stop_text)]
                        
                        # Send final segment if any
                        if segment and request.stream and token_queue:
                            token_queue.put(StreamToken(text=segment))
                        break
                    
                    # Stream handling (don't stream during tool calls)
                    if request.stream and token_queue and not tool_state.in_tool_call:
                        # Check for overlap with stop sequences
                        if any(sequence_overlap(tokens, seq) for seq in stop_id_sequences):
                            continue  # Wait for more tokens
                        
                        # Stream the segment
                        if segment:
                            token_queue.put(StreamToken(text=segment))
                            segment = ""
            
            # Broadcast generated tokens to all ranks for cache synchronization
            generated_tokens = self.broadcast_generated_tokens(tokens if self.rank == 0 else None)
            
            # Update cache on all ranks with the same generated tokens
            if generated_tokens:
                # All ranks update their cache with the same tokens
                self.prompt_cache.tokens.extend(generated_tokens)
                self.logger.debug(f"Cache updated with {len(generated_tokens)} new tokens. Total cache size: {len(self.prompt_cache.tokens)}")
            
            # Final handling on rank 0
            if self.rank == 0:
                # Parse tool calls based on model type
                if self.model_type == "kimi_k2":
                    parsed_tool_calls = [parse_kimi_tool_call(func_id, arguments) for func_id, arguments in tool_state.tool_calls] if tool_state.tool_calls else []
                else:
                    parsed_tool_calls = [parse_generic_tool_call(tool_text) for tool_text in tool_state.tool_calls] if tool_state.tool_calls else []
                
                if request.stream and token_queue:
                    # Send final token with tool calls
                    token_queue.put(StreamToken(
                        text="", 
                        is_final=True, 
                        finish_reason=finish_reason,
                        tool_calls=parsed_tool_calls
                    ))
                elif token_queue:
                    # For non-streaming, send complete result
                    result = {
                        'full_text': full_text,
                        'tokens': tokens,
                        'prompt_token_count': original_prompt_token_count,
                        'finish_reason': finish_reason,
                        'tool_calls': parsed_tool_calls
                    }
                    token_queue.put(result)
                
                self.logger.info(f"Generation complete: {len(tokens)} tokens, cache size: {len(self.prompt_cache.tokens)}")
                
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"Generation error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                if token_queue:
                    if request and request.stream:
                        token_queue.put(StreamToken(text="", is_final=True, error=str(e)))
                    else:
                        # Non-streaming error
                        token_queue.put({'error': str(e)})


class HTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for rank 0"""
    
    request_queue = None
    dist_model = None
    
    def log_message(self, format, *args):
        """Override to use proper logging"""
        logging.info(f"HTTP: {format % args}")
    
    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
    
    def _send_json_response(self, data: Dict[str, Any], status: int = 200):
        """Send a JSON response with proper headers"""
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/v1/models":
            self._handle_models()
        elif self.path.startswith("/v1/models/"):
            model_id = self.path.split("/")[-1]
            self._handle_model_info(model_id)
        else:
            self.send_error(404, "Not found")
    
    def _handle_health(self):
        """Health check endpoint"""
        response = {
            "status": "healthy",
            "version": "1.0.0",
            "distributed": True,
            "kv_cache": True,
            "rank": self.dist_model.rank if self.dist_model.rank == 0 else "worker",
            "cache_size": len(self.dist_model.prompt_cache.tokens),
            "timestamp": int(time.time())
        }
        self._send_json_response(response)
    
    def _handle_models(self):
        """List available models"""
        models = [{
            "id": self.dist_model.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mlx-distributed-server",
            "permission": [],
            "root": self.dist_model.model_name,
            "parent": None
        }]
        response = {
            "object": "list", 
            "data": models
        }
        self._send_json_response(response)
    
    def _handle_model_info(self, model_id: str):
        """Get specific model information"""
        if model_id != self.dist_model.model_name:
            self.send_error(404, f"Model '{model_id}' not found")
            return
        
        model_info = {
            "id": self.dist_model.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mlx-distributed-server",
            "permission": [],
            "root": self.dist_model.model_name,
            "parent": None,
            "capabilities": {
                "chat_completions": True,
                "text_completions": True,
                "streaming": True,
                "stop_sequences": True,
                "temperature_sampling": True,
                "distributed": True,
                "kv_cache": True,
                "tool_calling": self.dist_model.has_tool_calling
            },
            "model_type": self.dist_model.model_type
        }
        self._send_json_response(model_info)
    
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif self.path == "/v1/completions":
            self._handle_text_completions()
        elif self.path == "/v1/embeddings":
            self._handle_embeddings()
        elif self.path == "/chat/completions":  # Alternative path
            self._handle_chat_completions()
        else:
            self.send_error(404, "Not found")
            return
    
    def _parse_request_body(self):
        """Parse and return request body as JSON"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        return json.loads(post_data.decode('utf-8'))
    
    def _handle_chat_completions(self):
        """Handle /v1/chat/completions endpoint"""
        try:
            data = self._parse_request_body()
            
            messages = data.get("messages", [])
            if not messages:
                raise ValueError("No messages provided")
            
            # Apply chat template with tools support
            if hasattr(self.dist_model.tokenizer, "apply_chat_template"):
                tools = data.get("tools", None)
                prompt = self.dist_model.tokenizer.apply_chat_template(
                    messages,
                    tools,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                # Simple fallback
                prompt = messages[-1].get("content", "")
            
            # Extract parameters with CLI defaults as fallback
            params = extract_generation_params(data, self.dist_model.default_params)
            params['seed'] = data.get("seed", None)
            
            # Handle stop sequences
            stop_words = data.get('stop', [])
            if isinstance(stop_words, str):
                stop_words = [stop_words]
            elif stop_words is None:
                stop_words = []
            
            # Check if streaming
            stream = data.get("stream", False)
            
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                params=params,
                stop_words=stop_words,
                request_id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                stream=stream,
                conversation_id=data.get("conversation_id", None)  # Optional conversation tracking
            )
            
            if stream:
                self._handle_streaming_request(request)
            else:
                self._handle_non_streaming_request(request)
                
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
        except ValueError as e:
            self.send_error(400, str(e))
        except Exception as e:
            logging.error(f"Chat completions error: {e}")
            self.send_error(500, "Internal server error")
    
    def _handle_text_completions(self):
        """Handle /v1/completions endpoint"""
        try:
            data = self._parse_request_body()
            
            prompt = data.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0]  # Simple handling for now
            
            if not prompt:
                raise ValueError("No prompt provided")
            
            # Extract parameters with CLI defaults as fallback
            params = extract_generation_params(data, self.dist_model.default_params)
            params['seed'] = data.get("seed", None)
            
            # Handle stop sequences
            stop_words = data.get('stop', [])
            if isinstance(stop_words, str):
                stop_words = [stop_words]
            elif stop_words is None:
                stop_words = []
            
            # Check if streaming
            stream = data.get("stream", False)
            
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                params=params,
                stop_words=stop_words,
                request_id=f"cmpl-{uuid.uuid4()}",  # Different ID prefix for completions
                created=int(time.time()),
                stream=stream,
                conversation_id=data.get("conversation_id", None)
            )
            
            if stream:
                self._handle_streaming_request(request, completion_type="text_completion")
            else:
                self._handle_non_streaming_request(request, completion_type="text_completion")
                
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
        except ValueError as e:
            self.send_error(400, str(e))
        except Exception as e:
            logging.error(f"Text completions error: {e}")
            self.send_error(500, "Internal server error")
    
    def _handle_embeddings(self):
        """Handle /v1/embeddings endpoint (not supported but returns proper error)"""
        self.send_response(501)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        
        error_response = {
            "error": {
                "message": "Embeddings endpoint not supported by this distributed text generation server",
                "type": "not_supported_error",
                "code": "embeddings_not_supported"
            }
        }
        self.wfile.write(json.dumps(error_response).encode())
    
    def _handle_streaming_request(self, request: GenerationRequest, completion_type: str = "chat_completion"):
        """Handle streaming request"""
        # Send streaming headers
        self.send_response(200)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()
        self.end_headers()
        
        # Create token queue for streaming
        token_queue = queue.Queue()
        
        # Send request to generation queue
        self.request_queue.put((request, token_queue))
        
        # Stream tokens as they come
        while True:
            try:
                token = token_queue.get(timeout=30)  # 30 second timeout
                
                if token.error:
                    # Send error
                    error_chunk = {
                        "id": request.request_id,
                        "object": "chat.completion.chunk",
                        "created": request.created,
                        "error": {"message": token.error, "type": "server_error"}
                    }
                    self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    break
                
                elif token.is_final:
                    # Send final chunk with tool calls if present
                    delta = {}
                    if token.tool_calls:
                        delta["tool_calls"] = token.tool_calls
                    
                    final_chunk = {
                        "id": request.request_id,
                        "object": "chat.completion.chunk",
                        "created": request.created,
                        "model": self.dist_model.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": token.finish_reason
                        }]
                    }
                    self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    break
                
                else:
                    # Send token chunk
                    if completion_type == "text_completion":
                        chunk = {
                            "id": request.request_id,
                            "object": "text_completion",
                            "created": request.created,
                            "model": self.dist_model.model_name,
                            "choices": [{
                                "text": token.text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }]
                        }
                    else:
                        chunk = {
                            "id": request.request_id,
                            "object": "chat.completion.chunk",
                            "created": request.created,
                            "model": self.dist_model.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token.text},
                                "finish_reason": None
                            }]
                        }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                
                self.wfile.flush()
                
            except queue.Empty:
                # Timeout - send error
                error_chunk = {
                    "id": request.request_id,
                    "object": "chat.completion.chunk",
                    "created": request.created,
                    "error": {"message": "Generation timeout", "type": "timeout"}
                }
                self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                break
    
    def _handle_non_streaming_request(self, request: GenerationRequest, completion_type: str = "chat_completion"):
        """Handle non-streaming request"""
        # Use a result queue for non-streaming
        result_queue = queue.Queue()
        self.request_queue.put((request, result_queue))
        
        # Wait for complete result
        try:
            result = result_queue.get(timeout=300)  # 5 minute timeout for non-streaming
            
            if result.get('error'):
                self.send_error(500, result['error'])
                return
            
            # Send complete response
            if completion_type == "text_completion":
                response_data = {
                    "id": request.request_id,
                    "object": "text_completion",
                    "created": request.created,
                    "model": self.dist_model.model_name,
                    "choices": [{
                        "text": result['full_text'],
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": result['finish_reason']
                    }],
                    "usage": {
                        "prompt_tokens": result.get('prompt_token_count', 0),
                        "completion_tokens": len(result.get('tokens', [])),
                        "total_tokens": result.get('prompt_token_count', 0) + len(result.get('tokens', []))
                    }
                }
            else:
                # Prepare message with tool calls if present
                message = {
                    "role": "assistant", 
                    "content": result['full_text']
                }
                if result.get('tool_calls'):
                    message["tool_calls"] = result['tool_calls']
                
                response_data = {
                    "id": request.request_id,
                    "object": "chat.completion",
                    "created": request.created,
                    "model": self.dist_model.model_name,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": result['finish_reason']
                    }],
                    "usage": {
                        "prompt_tokens": result.get('prompt_token_count', 0),
                        "completion_tokens": len(result.get('tokens', [])),
                        "total_tokens": result.get('prompt_token_count', 0) + len(result.get('tokens', []))
                    }
                }
            
            response_json = json.dumps(response_data).encode()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()
            
        except queue.Empty:
            self.send_error(504, "Generation timeout")
            return


def run_http_server(host: str, port: int, request_queue: queue.Queue):
    """Run HTTP server in a separate thread"""
    server = HTTPServer((host, port), HTTPHandler)
    logging.info(f"HTTP server started on {host}:{port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Distributed MLX Server with True Streaming, KV Cache, and Tool Calling V5")
    parser.add_argument("--model", required=True, help="Path to local model directory")
    parser.add_argument("--server-host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--server-port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--max-cache-size", type=int, default=None, help="Maximum KV cache size (uses RotatingKVCache if set)")
    
    # Sampling parameters (matching server.py defaults)
    parser.add_argument("--temp", type=float, default=0.0, help="Default sampling temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Default nucleus sampling top-p (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=0, help="Default top-k sampling (default: 0, disables top-k)")
    parser.add_argument("--min-p", type=float, default=0.0, help="Default min-p sampling (default: 0.0, disables min-p)")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Default maximum number of tokens to generate (default: 16384)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Default repetition penalty (default: 1.0)")
    parser.add_argument("--repetition-context-size", type=int, default=20, help="Default repetition context size (default: 20)")
    
    args = parser.parse_args()
    
    # Initialize distributed model
    dist_model = DistributedModel(args.model, max_cache_size=args.max_cache_size)
    
    # Store CLI defaults for parameter fallback
    dist_model.default_params = {
        'temperature': args.temp,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'min_p': args.min_p,
        'max_tokens': args.max_tokens,
        'repetition_penalty': args.repetition_penalty,
        'repetition_context_size': args.repetition_context_size,
    }
    
    # Communication queue
    request_queue = queue.Queue()
    
    # Start HTTP server on rank 0
    if dist_model.rank == 0:
        HTTPHandler.request_queue = request_queue
        HTTPHandler.dist_model = dist_model
        
        http_thread = threading.Thread(
            target=run_http_server,
            args=(args.server_host, args.server_port, request_queue),
            daemon=True
        )
        http_thread.start()
        dist_model.logger.info(f"HTTP server thread started with KV cache support")
    
    # Main generation loop for ALL ranks
    try:
        while True:
            if dist_model.rank == 0:
                # Check for requests
                try:
                    request_data = request_queue.get_nowait()
                    request, token_queue = request_data
                    message = MessageType.GENERATE
                except queue.Empty:
                    request = None
                    token_queue = None
                    message = MessageType.IDLE
            else:
                request = None
                token_queue = None
                message = MessageType.IDLE
            
            # Synchronize message type across all ranks
            message = dist_model.synchronize_message(message)
            
            if message == MessageType.GENERATE:
                # All ranks participate in generation
                dist_model.generate_streaming(request, token_queue)
            
            elif message == MessageType.SHUTDOWN:
                dist_model.logger.info("Received shutdown signal")
                break
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        dist_model.logger.info("Interrupted by user")
    except Exception as e:
        dist_model.logger.error(f"Main loop error: {e}")
        raise
    
    dist_model.logger.info("Server shutting down")


if __name__ == "__main__":
    main()