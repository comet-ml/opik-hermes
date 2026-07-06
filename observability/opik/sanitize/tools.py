"""Serialization of tool calls and assistant messages.

Extracts a readable, safe representation of an assistant message's tool calls
(from provider response objects) for the LLM span output.
"""

from __future__ import annotations

from typing import Any

from .values import safe_value


def serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    serialized = []
    for tool_call in tool_calls:
        fn = getattr(tool_call, "function", None)
        name = getattr(fn, "name", None) if fn else None
        arguments = getattr(fn, "arguments", None) if fn else None
        safe_arguments = safe_value(arguments, parse_json_strings=False)
        serialized.append(
            {
                "id": getattr(tool_call, "id", None),
                "type": getattr(tool_call, "type", None) or "function",
                "name": name,
                "arguments": safe_arguments,
                "function": {"name": name, "arguments": safe_arguments},
            }
        )
    return serialized


def serialize_assistant_message(message: Any) -> dict[str, Any]:
    return {
        "content": safe_value(getattr(message, "content", None)),
        "reasoning": safe_value(getattr(message, "reasoning", None)),
        "tool_calls": serialize_tool_calls(getattr(message, "tool_calls", None)),
    }
