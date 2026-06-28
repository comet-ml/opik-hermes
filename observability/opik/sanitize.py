"""Payload sanitization: truncation, redaction, and structure normalization.

All functions here are pure (no plugin state). They make tool/LLM payloads
safe and compact before they reach the Opik SDK — truncating long strings,
redacting base64 data URIs, parsing embedded JSON, and normalizing the
``read_file`` tool's verbose payload into a preview.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .config import _env

_READ_FILE_LINE_RE = re.compile(r"^\s*(\d+)\|(.*)$")
_READ_FILE_HEAD_LINES = 25
_READ_FILE_TAIL_LINES = 15


def _is_base64_data_uri(value: str) -> bool:
    prefix = value[:200].lower()
    return prefix.startswith("data:") and ";base64," in prefix


def _redact_data_uri(value: str) -> dict[str, Any]:
    header = value.split(",", 1)[0] if "," in value else "data:"
    media_type = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
    return {
        "type": "data_uri",
        "media_type": media_type or None,
        "omitted": True,
        "length": len(value),
    }


def _truncate_text(value: str, max_chars: int) -> Any:
    if _is_base64_data_uri(value):
        return _redact_data_uri(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _maybe_parse_json_string(value: str) -> Any:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in "{[":
        return value
    try:
        parsed, idx = json.JSONDecoder().raw_decode(stripped)
    except Exception:
        return value
    if not isinstance(parsed, (dict, list)):
        return value

    trailing = stripped[idx:].strip()
    if not trailing:
        return parsed

    hint_key = "_hint" if trailing.startswith("[Hint:") else "_trailing_text"
    if isinstance(parsed, dict):
        merged = dict(parsed)
        key = hint_key if hint_key not in merged else "_trailing_text"
        merged[key] = trailing
        return merged

    return {"data": parsed, hint_key: trailing}


def _looks_like_read_file_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    content = value.get("content")
    return (
        isinstance(content, str)
        and "total_lines" in value
        and "file_size" in value
        and "is_binary" in value
        and "is_image" in value
        and not value.get("error")
    )


def _parse_read_file_lines(content: str) -> list[dict[str, Any]]:
    if not isinstance(content, str) or not content:
        return []

    lines = []
    for raw_line in content.splitlines():
        match = _READ_FILE_LINE_RE.match(raw_line)
        if not match:
            return []
        lines.append({"line": int(match.group(1)), "text": match.group(2)})
    return lines


def _build_read_file_preview(lines: list[dict[str, Any]]) -> dict[str, Any]:
    if len(lines) <= (_READ_FILE_HEAD_LINES + _READ_FILE_TAIL_LINES):
        return {"lines": lines}

    return {
        "head": lines[:_READ_FILE_HEAD_LINES],
        "tail": lines[-_READ_FILE_TAIL_LINES:],
        "omitted_line_count": len(lines)
        - _READ_FILE_HEAD_LINES
        - _READ_FILE_TAIL_LINES,
    }


def _normalize_read_file_payload(
    value: dict[str, Any], *, args: Any = None
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if isinstance(args, dict):
        path = args.get("path")
        offset = args.get("offset")
        limit = args.get("limit")
        if isinstance(path, str) and path:
            normalized["path"] = path
        if isinstance(offset, int):
            normalized["offset"] = offset
        if isinstance(limit, int):
            normalized["limit"] = limit

    lines = _parse_read_file_lines(value.get("content", ""))
    if lines:
        normalized["returned_lines"] = {
            "start": lines[0]["line"],
            "end": lines[-1]["line"],
            "count": len(lines),
        }
        normalized["content_preview"] = _build_read_file_preview(lines)
    elif value.get("content"):
        normalized["content_preview"] = {"text": value.get("content", "")}

    for key in (
        "total_lines",
        "file_size",
        "truncated",
        "is_binary",
        "is_image",
        "hint",
        "_warning",
        "mime_type",
        "dimensions",
        "similar_files",
        "error",
    ):
        if key in value:
            normalized[key] = value[key]

    base64_content = value.get("base64_content")
    if isinstance(base64_content, str) and base64_content:
        normalized["base64_content"] = {"omitted": True, "length": len(base64_content)}

    return normalized


def _normalize_payload(value: Any, *, tool_name: str = "", args: Any = None) -> Any:
    if _looks_like_read_file_payload(value):
        return _normalize_read_file_payload(
            value,
            args=args if tool_name == "read_file" else None,
        )
    return value


def _safe_value(
    value: Any,
    *,
    max_chars: Optional[int] = None,
    depth: int = 0,
    parse_json_strings: bool = False,
) -> Any:
    max_chars = (
        max_chars
        if max_chars is not None
        else int(_env("HERMES_OPIK_MAX_CHARS", "12000") or "12000")
    )
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, str):
        if parse_json_strings:
            parsed = _maybe_parse_json_string(value)
            if parsed is not value:
                return _safe_value(
                    parsed, max_chars=max_chars, depth=depth, parse_json_strings=True
                )
        return _truncate_text(value, max_chars)
    if isinstance(value, dict):
        normalized = _normalize_payload(value)
        if normalized is not value:
            return _safe_value(
                normalized,
                max_chars=max_chars,
                depth=depth,
                parse_json_strings=parse_json_strings,
            )
        return {
            str(k): _safe_value(
                v,
                max_chars=max_chars,
                depth=depth + 1,
                parse_json_strings=parse_json_strings,
            )
            for k, v in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _safe_value(
                v,
                max_chars=max_chars,
                depth=depth + 1,
                parse_json_strings=parse_json_strings,
            )
            for v in list(value)[:50]
        ]
    if hasattr(value, "__dict__"):
        return _safe_value(
            vars(value),
            max_chars=max_chars,
            depth=depth + 1,
            parse_json_strings=parse_json_strings,
        )
    return _truncate_text(repr(value), max_chars)


def _extract_last_user_message(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return {"role": "user", "content": _safe_value(message.get("content"))}
    return None


_TRACE_NAME_MAX_CHARS = 60


def _content_to_text(content: Any) -> str:
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


def _trace_name_from_messages(messages: Any) -> Optional[str]:
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
            text = " ".join(_content_to_text(message.get("content")).split())
            if not text:
                return None
            if len(text) <= _TRACE_NAME_MAX_CHARS:
                return text
            return text[:_TRACE_NAME_MAX_CHARS].rstrip() + "…"
    return None


def _coerce_request_messages(
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


def _responses_parts_to_text(content: Any) -> Any:
    """Flatten Responses-API content parts to text.

    In codex_responses mode a message's ``content`` is a list of typed parts
    like ``{"type": "input_text"|"output_text", "text": "..."}`` or
    ``{"type": "input_image"|"output_image", ...}``. Join the text parts and
    note any images, so the span input is readable instead of a raw blob.
    """
    if not isinstance(content, list):
        return _safe_value(content)
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
    text = _safe_value(" ".join(texts)) if texts else None
    if images and text is not None:
        return {"text": text, "images": images}
    if images:
        return {"images": images}
    return text


def _serialize_one_message(message: Any) -> Optional[dict[str, Any]]:
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
                "arguments": _safe_value(
                    message.get("arguments"), parse_json_strings=True
                ),
                "call_id": message.get("call_id"),
            },
        }
    if item_type == "function_call_output":
        return {
            "role": "tool",
            "call_id": message.get("call_id"),
            "content": _safe_value(message.get("output"), parse_json_strings=True),
        }
    if item_type == "reasoning":
        return {"role": "assistant", "content": "[reasoning]"}
    if item_type == "message":
        return {
            "role": message.get("role"),
            "content": _responses_parts_to_text(message.get("content")),
        }

    # Chat-Completions ``{role, content}`` (chat_completions mode).
    role = message.get("role")
    if role is None and "content" not in message:
        # Unknown item shape — keep its type so it isn't a silent null.
        return {"role": None, "content": {"unrecognized_item": _safe_value(message)}}
    item: dict[str, Any] = {
        "role": role,
        "content": _safe_value(
            message.get("content"), parse_json_strings=(role == "tool")
        ),
    }
    if role == "tool":
        if message.get("tool_call_id"):
            item["tool_call_id"] = message.get("tool_call_id")
        if message.get("name"):
            item["name"] = _safe_value(message.get("name"))
    if message.get("tool_calls"):
        item["tool_calls"] = _safe_value(
            message.get("tool_calls"), parse_json_strings=True
        )
    return item


def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    serialized = []
    for message in messages[-12:]:
        item = _serialize_one_message(message)
        if item is not None:
            serialized.append(item)
    return serialized


def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    serialized = []
    for tool_call in tool_calls:
        fn = getattr(tool_call, "function", None)
        name = getattr(fn, "name", None) if fn else None
        arguments = getattr(fn, "arguments", None) if fn else None
        safe_arguments = _safe_value(arguments, parse_json_strings=False)
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


def _serialize_assistant_message(message: Any) -> dict[str, Any]:
    return {
        "content": _safe_value(getattr(message, "content", None)),
        "reasoning": _safe_value(getattr(message, "reasoning", None)),
        "tool_calls": _serialize_tool_calls(getattr(message, "tool_calls", None)),
    }


def _as_input_dict(value: Any) -> dict[str, Any]:
    """Opik span input is a dict; wrap non-dict values under a stable key."""
    if isinstance(value, dict):
        return value
    return {"input": value}
