"""Generic value sanitization: truncation, redaction, and safe recursion.

The core of the sanitizer. ``safe_value`` recursively bounds any payload
(depth, width, string length) before it reaches the Opik SDK, redacting base64
data URIs and optionally parsing embedded JSON strings. All functions here are
pure (no plugin state).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ..config import env
from .read_file import looks_like_read_file_payload, normalize_read_file_payload

_MAX_DEPTH = 4
_MAX_ITEMS = 50


def is_base64_data_uri(value: str) -> bool:
    prefix = value[:200].lower()
    return prefix.startswith("data:") and ";base64," in prefix


def redact_data_uri(value: str) -> dict[str, Any]:
    header = value.split(",", 1)[0] if "," in value else "data:"
    media_type = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
    return {
        "type": "data_uri",
        "media_type": media_type or None,
        "omitted": True,
        "length": len(value),
    }


def truncate_text(value: str, max_chars: int) -> Any:
    if is_base64_data_uri(value):
        return redact_data_uri(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def maybe_parse_json_string(value: str) -> Any:
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


def normalize_payload(value: Any, *, tool_name: str = "", args: Any = None) -> Any:
    if looks_like_read_file_payload(value):
        return normalize_read_file_payload(
            value,
            args=args if tool_name == "read_file" else None,
        )
    return value


def safe_value(
    value: Any,
    *,
    max_chars: Optional[int] = None,
    depth: int = 0,
    parse_json_strings: bool = False,
) -> Any:
    max_chars = (
        max_chars
        if max_chars is not None
        else int(env("HERMES_OPIK_MAX_CHARS", "12000") or "12000")
    )
    if depth > _MAX_DEPTH:
        return "<max-depth>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, str):
        if parse_json_strings:
            parsed = maybe_parse_json_string(value)
            if parsed is not value:
                return safe_value(
                    parsed, max_chars=max_chars, depth=depth, parse_json_strings=True
                )
        return truncate_text(value, max_chars)
    if isinstance(value, dict):
        normalized = normalize_payload(value)
        if normalized is not value:
            return safe_value(
                normalized,
                max_chars=max_chars,
                depth=depth,
                parse_json_strings=parse_json_strings,
            )
        return {
            str(k): safe_value(
                v,
                max_chars=max_chars,
                depth=depth + 1,
                parse_json_strings=parse_json_strings,
            )
            for k, v in list(value.items())[:_MAX_ITEMS]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            safe_value(
                v,
                max_chars=max_chars,
                depth=depth + 1,
                parse_json_strings=parse_json_strings,
            )
            for v in list(value)[:_MAX_ITEMS]
        ]
    if hasattr(value, "__dict__"):
        return safe_value(
            vars(value),
            max_chars=max_chars,
            depth=depth + 1,
            parse_json_strings=parse_json_strings,
        )
    return truncate_text(repr(value), max_chars)
