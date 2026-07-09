"""Serialization of tool calls and assistant messages.

Extracts a readable, safe representation of an assistant message's tool calls
(from provider response objects) for the LLM span output.
"""

from __future__ import annotations

import json
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


def _tool_calls_as_text(tool_calls: list[dict[str, Any]]) -> str:
    lines = []
    for tc in tool_calls:
        name = tc.get("name") or "tool"
        arguments = tc.get("arguments")
        if isinstance(arguments, (dict, list)):
            rendered = json.dumps(arguments, ensure_ascii=False)
        elif arguments is None:
            rendered = ""
        else:
            rendered = str(arguments)
        lines.append(f"{name}({rendered})")
    return "Tool calls:\n" + "\n".join(f"- {line}" for line in lines)


def assistant_output(
    content: Any,
    reasoning: Any = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Shape the LLM-span output as an OpenAI chat-completion so Opik's pretty
    renderer (which reads ``choices[-1].message.content``) recognizes it.

    Pretty mode only surfaces non-empty ``content``, so on a tool-call turn with
    no assistant text we render the tool calls as readable text; the structured
    ``tool_calls`` stay on the message for the JSON/YAML view.
    """
    tool_calls = tool_calls or []
    pretty_content = content
    if not pretty_content and tool_calls:
        pretty_content = _tool_calls_as_text(tool_calls)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": pretty_content,
                    "reasoning": reasoning,
                    "tool_calls": tool_calls,
                }
            }
        ]
    }


def serialize_assistant_message(message: Any) -> dict[str, Any]:
    return assistant_output(
        content=safe_value(getattr(message, "content", None)),
        reasoning=safe_value(getattr(message, "reasoning", None)),
        tool_calls=serialize_tool_calls(getattr(message, "tool_calls", None)),
    )
