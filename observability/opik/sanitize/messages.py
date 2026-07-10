"""Message coercion and serialization for span/trace inputs.

Normalizes the several message shapes Hermes emits — Chat-Completions
``{role, content}`` dicts and Responses-API typed items — into a readable
``{role, content}`` form, derives a trace name from the latest user message,
and wraps values as span-input dicts.
"""

from __future__ import annotations

from typing import Any, Optional

from .values import safe_value

_TRACE_NAME_MAX_CHARS = 60
_MAX_SERIALIZED_MESSAGES = 12


def extract_last_user_message(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return {"role": "user", "content": safe_value(message.get("content"))}
    return None


def content_to_text(content: Any) -> str:
    """Flatten a message ``content`` (str, or list of content parts) to text.

    OpenAI-style content can be a plain string or a list of parts like
    ``[{"type": "text", "text": "..."}]``. Returns "" when no text is present.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return ""


def trace_name_from_messages(messages: Any) -> Optional[str]:
    """Derive a scannable trace name from the latest user message.

    Follows the Opik convention of naming a trace after the unit of work it
    represents rather than a constant. For an agent turn the most identifying
    signal is what the user asked, so we use a whitespace-collapsed, truncated
    preview of the last user message. Returns ``None`` when there is no usable
    user text, so the caller can fall back to a stable default.
    """
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            text = " ".join(content_to_text(message.get("content")).split())
            if not text:
                return None
            if len(text) <= _TRACE_NAME_MAX_CHARS:
                return text
            return text[:_TRACE_NAME_MAX_CHARS].rstrip() + "…"
    return None


def coerce_request_messages(
    *,
    request_messages: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
) -> list[dict[str, Any]]:
    for candidate in (request_messages, messages, conversation_history):
        if isinstance(candidate, list):
            return candidate
    if user_message is None:
        return []
    return [{"role": "user", "content": user_message}]


def responses_parts_to_text(content: Any) -> Any:
    """Flatten Responses-API content parts to text.

    In codex_responses mode a message's ``content`` is a list of typed parts
    like ``{"type": "input_text"|"output_text", "text": "..."}`` or
    ``{"type": "input_image"|"output_image", ...}``. Join the text parts and
    note any images, so the span input is readable instead of a raw blob.
    """
    if not isinstance(content, list):
        return safe_value(content)
    texts, images = [], 0
    for part in content:
        if isinstance(part, dict):
            ptype = part.get("type", "")
            if "image" in ptype:
                images += 1
            elif isinstance(part.get("text"), str):
                texts.append(part["text"])
        elif isinstance(part, str):
            texts.append(part)
    text = safe_value(" ".join(texts)) if texts else None
    if images and text is not None:
        return {"text": text, "images": images}
    if images:
        return {"images": images}
    return text


def serialize_one_message(message: Any) -> Optional[dict[str, Any]]:
    """Normalize a single message item (Chat-Completions OR Responses API).

    Hermes passes Chat-Completions ``{role, content}`` dicts in
    chat_completions mode, but Responses-API items in codex_responses mode:
    ``{type: "message"|"function_call"|"function_call_output"|"reasoning", ...}``
    — most of which have no top-level role/content. Map each to a readable
    ``{role, content}`` shape so the span input isn't a wall of nulls.
    """
    if not isinstance(message, dict):
        return None

    item_type = message.get("type")

    # Responses-API typed items (codex_responses mode).
    if item_type == "function_call":
        return {
            "role": "assistant",
            "content": {
                "tool_call": message.get("name"),
                "arguments": safe_value(
                    message.get("arguments"), parse_json_strings=True
                ),
                "call_id": message.get("call_id"),
            },
        }
    if item_type == "function_call_output":
        return {
            "role": "tool",
            "call_id": message.get("call_id"),
            "content": safe_value(message.get("output"), parse_json_strings=True),
        }
    if item_type == "reasoning":
        return {"role": "assistant", "content": "[reasoning]"}
    if item_type == "message":
        return {
            "role": message.get("role"),
            "content": responses_parts_to_text(message.get("content")),
        }

    # Chat-Completions ``{role, content}`` (chat_completions mode).
    role = message.get("role")
    if role is None and "content" not in message:
        # Unknown item shape — keep its type so it isn't a silent null.
        return {"role": None, "content": {"unrecognized_item": safe_value(message)}}
    content = safe_value(message.get("content"), parse_json_strings=(role == "tool"))
    tool_calls = message.get("tool_calls")

    # An assistant turn that produced neither text nor tool calls has no signal
    # — it renders as an empty message bubble. Drop it rather than emit noise.
    if role == "assistant" and not content and not tool_calls:
        return None

    item: dict[str, Any] = {"role": role, "content": content}
    if role == "tool":
        if message.get("tool_call_id"):
            item["tool_call_id"] = message.get("tool_call_id")
        if message.get("name"):
            item["name"] = safe_value(message.get("name"))
    if tool_calls:
        item["tool_calls"] = safe_value(tool_calls, parse_json_strings=True)
    return item


def serialize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    serialized = []
    for message in messages[-_MAX_SERIALIZED_MESSAGES:]:
        item = serialize_one_message(message)
        if item is not None:
            serialized.append(item)
    return serialized


def as_input_dict(value: Any) -> dict[str, Any]:
    """Opik span input is a dict; wrap non-dict values under a stable key."""
    if isinstance(value, dict):
        return value
    return {"input": value}
