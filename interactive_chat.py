#!/usr/bin/env python3
"""
Interactive chat client with streaming support for the distributed server.
V4: Refactored for cleaner code, less duplication, and better organization.
"""

import requests
import json
import time
import sys
import threading
import tempfile
import subprocess
import os
import argparse
from datetime import datetime
from contextlib import contextmanager
import shutil
from typing import List, Dict


# Constants
LONG_INPUT_THRESHOLD = 200
PREVIEW_LENGTH = 200
STATS_WIDTH = 60

# Parameter validation
PARAM_VALIDATORS = {
    'temperature': lambda x: 0.0 <= x <= 2.0,
    'top_p': lambda x: 0.0 <= x <= 1.0,
    'max_tokens': lambda x: x > 0,
    'top_k': lambda x: x >= 0,
    'min_p': lambda x: 0.0 <= x <= 1.0,
    'repetition_penalty': lambda x: x > 0,
    'repetition_context_size': lambda x: x > 0,
}


class StreamingChatClient:
    def __init__(self, base_url="http://localhost:8080", default_params=None):
        self.base_url = base_url
        self.conversation = []
        self.default_params = default_params or {
            'max_tokens': 1024,
            'temperature': 0.6,
            'top_p': 1.0,
            'top_k': 0,
            'min_p': 0.0,
            'repetition_penalty': 1.0,
            'repetition_context_size': 20,
        }
        self.streaming_mode = True
        
    def get_user_input(self, prompt="👤 You: "):
        """Get user input with support for long and multi-line text"""
        print(prompt, end="", flush=True)
        
        try:
            user_input = input().strip()
            
            # Suggest multiline for long inputs
            if len(user_input) > LONG_INPUT_THRESHOLD or user_input.endswith("..."):
                print("📝 Input seems long. Use '/multiline' command for better editing.")
            
            return user_input
                
        except (EOFError, KeyboardInterrupt):
            return None
    
    @contextmanager
    def temp_editor_file(self):
        """Context manager for temporary editor files"""
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
                temp_file = f.name
                f.write("# Enter your message here\n")
                f.write("# Lines starting with # will be ignored\n")
                f.write("# Save and close this file when done\n\n")
            yield temp_file
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
    
    def get_multiline_input(self):
        """Get multi-line input using a temporary editor"""
        print("🖊️  Opening editor for multi-line input...")
        
        # Find available editor
        editor = os.environ.get('EDITOR', 'nano')
        editors_to_try = [editor, 'nano', 'vim'] if editor != 'nano' else ['nano', 'vim']
        
        available_editor = None
        for ed in editors_to_try:
            if shutil.which(ed):
                available_editor = ed
                break
        
        if not available_editor:
            return self.get_simple_multiline_input()
        
        # Use editor
        with self.temp_editor_file() as temp_file:
            try:
                subprocess.run([available_editor, temp_file], check=True)
                
                # Read and filter content
                with open(temp_file, 'r') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                
                result = '\n'.join(lines).strip()
                if not result:
                    print("⚠️  Empty input, cancelling...")
                    return ""
                
                print(f"📝 Got {len(result)} characters of input.")
                return result
                
            except subprocess.CalledProcessError:
                print("⚠️  Editor cancelled")
                return ""
    
    def get_simple_multiline_input(self):
        """Fallback multi-line input without editor"""
        print("📝 Enter your message (type 'END' on a new line to finish):")
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        return '\n'.join(lines).strip()
    
    def handle_multiline_command(self):
        """Handle the /multiline command by getting multiline input and sending it"""
        multiline_input = self.get_multiline_input()
        if multiline_input:
            # Send the multiline input as a regular message
            if self.streaming_mode:
                self.chat_streaming(multiline_input)
            else:
                self.chat_non_streaming(multiline_input)
    
    def build_payload(self, messages, params, streaming=True):
        """Build the API payload"""
        return {
            "messages": messages,
            "stream": streaming,
            **params  # Unpack all parameters
        }
    
    def chat_streaming(self, user_input):
        """Send chat request and stream the response"""
        self.conversation.append({"role": "user", "content": user_input})
        
        # Show preview for long inputs
        if len(user_input) > PREVIEW_LENGTH:
            preview = user_input[:PREVIEW_LENGTH] + "..."
            print(f"💬 Sending message preview: {preview}")
        
        payload = self.build_payload(self.conversation, self.default_params, streaming=True)
        
        print("🤖 Assistant: ", end="", flush=True)
        
        start_time = time.time()
        first_token_time = None
        response_text = ""
        chunk_count = 0  # Count actual chunks for better token approximation
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"\n❌ Error: {response.status_code} - {response.text}")
                self.conversation.pop()  # Remove failed message
                return
            
            # Stream response
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and chunk['choices']:
                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                print(content, end='', flush=True)
                                response_text += content
                                chunk_count += 1  # Count each chunk as a token
                    except json.JSONDecodeError:
                        continue
            
            print()  # New line after response
            
            # Add assistant response to conversation
            if response_text:
                self.conversation.append({"role": "assistant", "content": response_text})
            
            # Print statistics
            self.print_stats(start_time, first_token_time, chunk_count, is_streaming=True)
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Network error: {e}")
            self.conversation.pop()  # Remove failed message
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            self.conversation.pop()
    
    def chat_non_streaming(self, user_input):
        """Send chat request and get complete response"""
        self.conversation.append({"role": "user", "content": user_input})
        
        # Show preview for long inputs
        if len(user_input) > PREVIEW_LENGTH:
            preview = user_input[:PREVIEW_LENGTH] + "..."
            print(f"💬 Sending message preview: {preview}")
        
        payload = self.build_payload(self.conversation, self.default_params, streaming=False)
        
        print("⏳ Waiting for response...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            end_time = time.time()
            
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code} - {response.text}")
                self.conversation.pop()
                return
            
            result = response.json()
            
            if 'choices' in result and result['choices']:
                response_text = result['choices'][0]['message']['content']
                print("🤖 Assistant:", response_text)
                self.conversation.append({"role": "assistant", "content": response_text})
                
                # Print statistics
                self.print_stats(start_time, None, 0, is_streaming=False)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            self.conversation.pop()
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.conversation.pop()
    
    def print_stats(self, start_time, first_token_time, token_count, is_streaming=True):
        """Print response statistics"""
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'─' * STATS_WIDTH}")
        if is_streaming and first_token_time:
            print(f"⏱️  First token: {first_token_time - start_time:.2f}s")
            # For streaming, calculate tokens/sec from first token to end
            generation_time = end_time - first_token_time
            if generation_time > 0 and token_count > 0:
                tokens_per_sec = token_count / generation_time
            else:
                tokens_per_sec = 0
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"📊 {token_count} tokens")
            if tokens_per_sec > 0:
                print(f"⚡ {tokens_per_sec:.1f} tokens/s")
        else:
            # For non-streaming, we don't have token count
            print(f"⏱️  Total time: {total_time:.2f}s")
            
        print(f"{'─' * STATS_WIDTH}")
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation = []
        print("🗑️  Conversation cleared.")
    
    def show_conversation(self):
        """Display the conversation history"""
        if not self.conversation:
            print("💬 No conversation history.")
            return
        
        print("\n📜 Conversation History:")
        print("─" * STATS_WIDTH)
        for i, msg in enumerate(self.conversation):
            role_icon = "👤" if msg['role'] == 'user' else "🤖"
            content = msg['content']
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"{i+1}. {role_icon} {msg['role'].capitalize()}: {content}")
        print("─" * STATS_WIDTH)
    
    def show_params(self):
        """Show current parameters"""
        print("\n⚙️  Current Parameters:")
        print("─" * STATS_WIDTH)
        for key, value in self.default_params.items():
            print(f"  {key}: {value}")
        print(f"  streaming: {self.streaming_mode}")
        print("─" * STATS_WIDTH)
    
    def show_help(self):
        """Show help message"""
        print("\n📚 Available Commands:")
        print("─" * STATS_WIDTH)
        print("  /help         - Show this help message")
        print("  /clear        - Clear conversation history")
        print("  /history      - Show conversation history")
        print("  /params       - Show current parameters")
        print("  /set <p>=<v>  - Set parameter (e.g., /set temperature=0.8)")
        print("  /save [file]  - Save conversation to JSON")
        print("  /export [file]- Export to JSONL with metadata")
        print("  /load <file>  - Load conversation from file")
        print("  /multiline    - Enter multi-line input mode")
        print("  /quit         - Exit the chat")
        print("─" * STATS_WIDTH)
    
    def update_param(self, param_str):
        """Update a parameter value"""
        try:
            key, value = param_str.split('=', 1)
            key = key.strip()
            
            if key == 'streaming':
                self.streaming_mode = value.strip().lower() in ['true', '1', 'yes', 'on']
                print(f"✅ Streaming mode: {self.streaming_mode}")
                return
            
            if key not in self.default_params:
                print(f"❌ Unknown parameter: {key}")
                print(f"Available: {', '.join(self.default_params.keys())}")
                return
            
            # Convert value to appropriate type
            try:
                if key in ['temperature', 'top_p', 'min_p', 'repetition_penalty']:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                print(f"❌ Invalid value for {key}: {value}")
                return
            
            # Validate parameter
            if key in PARAM_VALIDATORS and not PARAM_VALIDATORS[key](value):
                print(f"❌ Invalid {key} value. Check the valid range.")
                return
            
            self.default_params[key] = value
            print(f"✅ Updated {key} to {value}")
            
        except ValueError:
            print("❌ Invalid format. Use: set param=value")
    
    def save_conversation(self, filename: str = None):
        """Save conversation to a JSON file"""
        if not self.conversation:
            print("💬 No conversation to save.")
            return
            
        if filename is None:
            filename = f"chat_{int(time.time())}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.conversation, f, indent=2)
            print(f"💾 Conversation saved to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
            return None
    
    def export_conversation_jsonl(self, filename: str = None):
        """Export conversation to JSONL format with metadata"""
        if not self.conversation:
            print("💬 No conversation to export.")
            return
            
        if filename is None:
            filename = f"chat_{int(time.time())}.jsonl"
        
        try:
            with open(filename, 'w') as f:
                # Write metadata
                metadata = {
                    "type": "metadata",
                    "timestamp": datetime.now().isoformat(),
                    "message_count": len(self.conversation),
                    "format": "openai_chat"
                }
                f.write(json.dumps(metadata) + '\n')
                
                # Write each message as a separate JSON line
                for i, msg in enumerate(self.conversation):
                    msg_with_meta = {
                        "type": "message",
                        "index": i,
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": datetime.now().isoformat()
                    }
                    f.write(json.dumps(msg_with_meta) + '\n')
                
                # Write conversation as training format
                if len(self.conversation) >= 2:
                    f.write('\n# Training format:\n')
                    training_example = {
                        "messages": self.conversation
                    }
                    f.write(json.dumps(training_example) + '\n')
            
            print(f"📤 Conversation exported to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error exporting conversation: {e}")
            return None
    
    def load_conversation(self, filename: str):
        """Load conversation from a JSON file"""
        try:
            with open(filename, 'r') as f:
                messages = json.load(f)
            
            if not isinstance(messages, list):
                print(f"❌ Invalid format: expected a list of messages")
                return False
                
            # Validate message format
            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    print(f"❌ Invalid message format in file")
                    return False
            
            self.conversation = messages
            print(f"📥 Loaded {len(messages)} messages from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON file: {filename}")
            return False
        except Exception as e:
            print(f"❌ Error loading conversation: {e}")
            return False
    
    def run_interactive(self):
        """Run the interactive chat loop"""
        print("🌟 Interactive Chat Client (V4)")
        print("💡 Commands: /quit, /clear, /history, /params, /set <param>=<value>")
        print("   /save, /export, /load <file>, /help, /multiline")
        print("─" * STATS_WIDTH)
        
        # Test server connection
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Connected to server")
            else:
                print("⚠️  Server returned unexpected status")
        except requests.exceptions.RequestException:
            print("⚠️  Could not connect to server at", self.base_url)
            print("   Make sure the server is running")
        
        print()
        
        # Command handlers
        commands = {
            '/clear': self.clear_conversation,
            '/history': self.show_conversation,
            '/params': self.show_params,
            '/save': lambda: self.save_conversation(),
            '/export': lambda: self.export_conversation_jsonl(),
            '/help': lambda: self.show_help(),
            '/multiline': lambda: self.handle_multiline_command(),
        }
        
        while True:
            try:
                user_input = self.get_user_input()
                
                if user_input is None:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Check for quit commands
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("👋 Goodbye!")
                    break
                
                # Check if it's a slash command
                if user_input.startswith('/'):
                    # Parse command and arguments
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else None
                    
                    # Check for commands
                    if command in commands:
                        commands[command]()
                    elif command == '/set' and args:
                        self.update_param(args)
                    elif command == '/save' and args:
                        self.save_conversation(args)
                    elif command == '/export' and args:
                        self.export_conversation_jsonl(args)
                    elif command == '/load' and args:
                        self.load_conversation(args)
                    elif command == '/load':
                        print("❌ Usage: /load <filename>")
                    else:
                        print(f"❌ Unknown command: {command}")
                        print("💡 Type /help for available commands")
                else:
                    # Send message
                    if self.streaming_mode:
                        self.chat_streaming(user_input)
                    else:
                        self.chat_non_streaming(user_input)
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive chat client with streaming support")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--repetition-context-size", type=int, default=20, help="Repetition context size")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming mode")
    
    args = parser.parse_args()
    
    # Build default parameters from args
    default_params = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'min_p': args.min_p,
        'repetition_penalty': args.repetition_penalty,
        'repetition_context_size': args.repetition_context_size,
    }
    
    client = StreamingChatClient(args.server, default_params)
    client.streaming_mode = not args.no_streaming
    
    try:
        client.run_interactive()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()