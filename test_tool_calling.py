#!/usr/bin/env python3
"""
Test script for tool calling functionality in the distributed server V5.
Tests with a fake weather tool function.
"""

import requests
import json
import time
from typing import Dict, Any


def fake_get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Fake weather function that returns mock weather data.
    In a real implementation, this would call a weather API.
    """
    fake_weather_data = {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_speed": "10 km/h",
        "forecast": "Sunny with a chance of rain in the afternoon"
    }
    
    print(f"🌤️  Mock weather API called for {location}")
    return fake_weather_data


def execute_function_call(function_name: str, arguments: str) -> str:
    """Execute a function call and return the result."""
    try:
        args = json.loads(arguments)
        
        if function_name == "get_current_weather":
            result = fake_get_weather(**args)
            return json.dumps(result)
        else:
            return json.dumps({"error": f"Unknown function: {function_name}"})
            
    except Exception as e:
        return json.dumps({"error": str(e)})


def test_tool_calling(server_url: str = "http://localhost:8080"):
    """Test tool calling functionality."""
    
    print("🚀 Testing Tool Calling with Distributed MLX Server V5")
    print("=" * 60)
    
    # Define the weather tool specification
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit for temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Test conversation
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Tokyo?"
        }
    ]
    
    print("📋 Tool Specification:")
    print(json.dumps(tools[0], indent=2))
    print()
    
    print("💬 User Message:")
    print(f"  {messages[0]['content']}")
    print()
    
    # Send request to the server
    payload = {
        "model": "distributed-model",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",  # Let the model decide when to use tools
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": False  # Start with non-streaming for simplicity
    }
    
    print("🌐 Sending request to server...")
    print(f"   URL: {server_url}/v1/chat/completions")
    print(f"   Tools provided: {len(tools)}")
    print()
    
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        result = response.json()
        
        print("✅ Server Response:")
        print(f"   Status: {response.status_code}")
        print(f"   Model: {result.get('model', 'unknown')}")
        print()
        
        
        choice = result["choices"][0]
        message = choice["message"]
        
        print("🤖 Assistant Response:")
        print(f"   Content: {message.get('content', '')}")
        print()
        
        # Check for tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            print(f"   Tool Calls: {len(tool_calls)}")
            print()
            
            # Process each tool call
            for i, tool_call in enumerate(tool_calls):
                function_name = tool_call["function"]["name"]
                function_args = tool_call["function"]["arguments"]
                
                print(f"🔧 Tool Call #{i+1}:")
                print(f"   Function: {function_name}")
                print(f"   Arguments: {function_args}")
                
                # Execute the function
                function_result = execute_function_call(function_name, function_args)
                print(f"   Result: {function_result}")
                print()
                
                # In a real implementation, you would send the function result back
                # to the model in a follow-up conversation turn
                messages.append({
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": tool_calls
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", "call_1"),
                    "name": function_name,
                    "content": function_result
                })
            
            # Send follow-up request with function results
            print("🔄 Sending follow-up with function results...")
            
            followup_payload = {
                "model": "distributed-model",
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.3,
                "max_tokens": 1024,
                "stream": False
            }
            
            followup_response = requests.post(
                f"{server_url}/v1/chat/completions",
                json=followup_payload,
                timeout=30
            )
            
            if followup_response.status_code == 200:
                followup_result = followup_response.json()
                final_message = followup_result["choices"][0]["message"]
                
                print("🎯 Final Assistant Response:")
                print(f"   Content: {final_message.get('content', '')}")
                print()
            else:
                print(f"❌ Follow-up failed: {followup_response.status_code}")
                print(f"   Response: {followup_response.text}")
        else:
            print("   No tool calls made")
            print()
        
        # Print usage statistics
        usage = result.get("usage", {})
        if usage:
            print("📊 Usage Statistics:")
            print(f"   Prompt tokens: {usage.get('prompt_tokens', 0)}")
            print(f"   Completion tokens: {usage.get('completion_tokens', 0)}")
            print(f"   Total tokens: {usage.get('total_tokens', 0)}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")
        print(f"   Raw response: {response.text}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


def test_streaming_tool_calling(server_url: str = "http://localhost:8080"):
    """Test streaming tool calling functionality."""
    
    print("\n" + "=" * 60)
    print("🌊 Testing Streaming Tool Calling")
    print("=" * 60)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "Can you check the weather in Paris for me?"
        }
    ]
    
    payload = {
        "model": "distributed-model",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": True
    }
    
    print("🌐 Sending streaming request...")
    print()
    
    try:
        with requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=30
        ) as response:
            
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                return
            
            print("📡 Streaming response:")
            tool_calls_received = []
            content_received = ""
            all_chunks = []  # Store all chunks for debugging
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            print("\n✅ Stream completed")
                            break
                        
                        try:
                            chunk = json.loads(data)
                            all_chunks.append(chunk)  # Store for debugging
                            
                            choices = chunk.get('choices', [])
                            if choices:
                                delta = choices[0].get('delta', {})
                                
                                # Handle content
                                if 'content' in delta and delta['content']:
                                    content = delta['content']
                                    content_received += content
                                    print(content, end='', flush=True)
                                
                                # Handle tool calls
                                if 'tool_calls' in delta:
                                    tool_calls_received.extend(delta['tool_calls'])
                                    print(f"\n🔧 Tool calls received: {len(delta['tool_calls'])}")
                                
                                # Handle finish reason
                                finish_reason = choices[0].get('finish_reason')
                                if finish_reason:
                                    print(f"\n🏁 Finished: {finish_reason}")
                        
                        except json.JSONDecodeError:
                            continue
            
            
            print()
            if tool_calls_received:
                print(f"🔧 Total tool calls: {len(tool_calls_received)}")
                for tool_call in tool_calls_received:
                    print(f"   Function: {tool_call.get('function', {}).get('name', 'unknown')}")
            
    except Exception as e:
        print(f"❌ Streaming Error: {e}")


def main():
    """Main test function."""
    server_url = "http://localhost:8080"
    
    # Test non-streaming first
    test_tool_calling(server_url)
    
    # Test streaming
    test_streaming_tool_calling(server_url)
    
    print("\n" + "=" * 60)
    print("🎉 Tool calling tests completed!")
    print("💡 Make sure your model supports tool calling for this to work properly.")
    print("   Common tool-calling models: gpt-4, claude-3, function-calling variants")


if __name__ == "__main__":
    main()