"""Tool Parser - Parse tool calls from text and convert tools to prompts"""

import json
import re
from typing import List, Dict, Tuple


class ToolParser:
    """Parse tool calls from model responses"""

    @staticmethod
    def tools_to_system_prompt(tools: List[Dict]) -> str:
        """Convert tools to system prompt format"""
        tool_descriptions = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters", {})

                tool_desc = f"<tool>\n<name>{name}</name>\n<description>{description}</description>"

                if parameters:
                    params_desc = json.dumps(parameters, ensure_ascii=False, indent=2)
                    tool_desc += f"\n<parameters>{params_desc}</parameters>"

                tool_desc += "\n</tool>"
                tool_descriptions.append(tool_desc)

        if not tool_descriptions:
            return ""

        prompt = """Available Tools:
You can use the following tools to help answer the user's question. When you need to use a tool, output in this format:

<tool_calling>
<name>tool_name</name>
<arguments>{"param1": "value1", "param2": "value2"}</arguments>
</tool_calling>

Here are the available tools:

"""
        prompt += "\n\n".join(tool_descriptions)
        prompt += """

Remember: Only use tools when necessary. If you can answer directly, do so without using tools."""

        return prompt

    @staticmethod
    def parse_tool_calls_from_text(text: str) -> List[Dict]:
        """Parse tool calls from text response"""
        tool_calls = []

        # Match tool_calling blocks
        pattern = r"<tool_calling>\s*<name>([^<]+)</name>\s*<arguments>([^<]+)</arguments>\s*</tool_calling>"
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)

        for idx, match in enumerate(matches):
            name = match.group(1).strip()
            arguments_str = match.group(2).strip()

            # Try to parse arguments as JSON
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {"raw": arguments_str}

            tool_call = {
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
            tool_calls.append(tool_call)

        return tool_calls

    @staticmethod
    def parse_tool_calls_from_text_with_content(text: str) -> Tuple[str, List[Dict]]:
        """Parse tool calls and return cleaned content + tool calls"""
        tool_calls = ToolParser.parse_tool_calls_from_text(text)

        # Remove tool_calling blocks from text
        pattern = r"<tool_calling>\s*<name>[^<]+</name>\s*<arguments>[^<]+</arguments>\s*</tool_calling>"
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text, tool_calls

    @staticmethod
    def extract_tool_results(text: str) -> List[Dict]:
        """Extract tool results from text"""
        results = []

        # Match tool_response blocks
        pattern = r'<tool_response\s+tool_call_id="([^"]+)">\s*(.+?)\s*</tool_response>'
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            tool_call_id = match.group(1)
            content = match.group(2).strip()

            results.append({"tool_call_id": tool_call_id, "content": content})

        return results


# 便捷函数
def parse_tool_calls(text: str) -> List[Dict]:
    """Parse tool calls from text"""
    return ToolParser.parse_tool_calls_from_text(text)


def parse_tool_calls_with_content(text: str) -> Tuple[str, List[Dict]]:
    """Parse tool calls and return cleaned content"""
    return ToolParser.parse_tool_calls_from_text_with_content(text)


def tools_to_prompt(tools: List[Dict]) -> str:
    """Convert tools to system prompt"""
    return ToolParser.tools_to_system_prompt(tools)
