# Qwen3 XML Tool Parser - Adapted for MLX without vllm dependencies
import json
import re
import uuid
import logging
from typing import Optional, Any, List, Dict, Union

logger = logging.getLogger(__name__)


class Qwen3XMLToolParser:
    def __init__(self):
        # Sentinel tokens for parsing
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_prefix = "<function="
        self.function_end_token = "</function>"
        self.parameter_prefix = "<parameter="
        self.parameter_end_token = "</parameter>"
        
        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)>(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)>(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the tool configuration."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        param_type = "string"
        if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
            param_type = str(param_config[param_name]["type"]).strip().lower()

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif param_type.startswith(("int", "uint", "long", "short", "unsigned")):
            try:
                return int(param_value)
            except:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
                return param_value
        elif param_type.startswith(("num", "float")):
            try:
                float_value = float(param_value)
                return float_value if float_value - int(float_value) != 0 else int(float_value)
            except:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value_lower = param_value.lower()
            if param_value_lower not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean in tool '{func_name}', degenerating to false."
                )
            return param_value_lower == "true"
        else:
            if param_type == "object" or param_type.startswith("dict"):
                try:
                    return json.loads(param_value)
                except:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' is not a valid JSON object in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                return eval(param_value)
            except:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `eval()` in tool '{func_name}', degenerating to string."
                )
                return param_value

    def _get_arguments_config(self, func_name: str, tools: Optional[List[dict]]) -> dict:
        """Extract parameter configuration for a specific function from tools list."""
        if tools is None:
            return {}
        
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                if function.get("name") == func_name:
                    params = function.get("parameters", {})
                    if isinstance(params, dict) and "properties" in params:
                        return params["properties"]
                    elif isinstance(params, dict):
                        return params
                    else:
                        return {}
        
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def parse_tool_call(
        self, tool_text: str, tools: Optional[List[dict]] = None
    ) -> Optional[dict]:
        """Parse a single tool call from XML format to dict format."""
        if not tool_text or not tool_text.strip():
            logger.warning("Empty tool_text received")
            return None
        
        tool_text = tool_text.strip()
        logger.debug(f"Parsing tool_text: {tool_text}")
        
        # First check if it's already JSON format (backward compatibility)
        try:
            tool_call = json.loads(tool_text)
            return {
                "function": {
                    "name": tool_call.get("name", None),
                    "arguments": json.dumps(tool_call.get("arguments", {})),
                },
                "type": "function",
                "id": self._generate_tool_call_id(),
            }
        except json.JSONDecodeError:
            pass
        
        # Parse XML format
        function_match = self.tool_call_function_regex.search(tool_text)
        if not function_match:
            logger.error(f"Could not parse tool call: {tool_text}")
            return None
        
        # Extract function name and content
        function_name = function_match.group(1) or function_match.group(3)
        function_content = function_match.group(2) or ""
        
        # Get parameter configuration
        param_config = self._get_arguments_config(function_name, tools)
        
        # Extract parameters
        params = {}
        for match in self.tool_call_parameter_regex.finditer(function_content):
            param_name = match.group(1) or match.group(3)
            param_value = (match.group(2) or "").strip()
            
            # Remove leading/trailing newlines
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]
            
            # Convert parameter value based on type
            params[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        
        return {
            "function": {
                "name": function_name,
                "arguments": json.dumps(params, ensure_ascii=False),
            },
            "type": "function",
            "id": self._generate_tool_call_id(),
        }

    def extract_tool_calls(
        self, model_output: str, tools: Optional[List[dict]] = None
    ) -> tuple[bool, List[dict], Optional[str]]:
        """
        Extract all tool calls from model output.
        
        Returns:
            tuple: (tools_called, tool_calls, content)
                - tools_called: whether any tools were called
                - tool_calls: list of parsed tool call dicts
                - content: text content before tool calls
        """
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:
            return False, [], model_output
        
        try:
            # Find all function calls
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return False, [], model_output
            
            # Parse each function call
            tool_calls = []
            for function_call_str in function_calls:
                parsed = self.parse_tool_call(function_call_str, tools)
                if parsed:
                    tool_calls.append(parsed)
            
            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            if content_index == -1:
                content_index = model_output.find(self.tool_call_prefix)
            
            content = model_output[:content_index] if content_index >= 0 else ""
            
            return len(tool_calls) > 0, tool_calls, content
            
        except Exception as e:
            logger.exception("Error in extracting tool call from response.")
            return False, [], model_output

    def _get_function_calls(self, model_output: str) -> List[str]:
        """Extract all function call strings from model output."""
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]
        
        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]
        
        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))
        
        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls