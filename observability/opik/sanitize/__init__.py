"""Pure payload sanitization for the Opik Hermes plugin.

Makes tool/LLM payloads safe and compact before they reach the Opik SDK.
Grouped by concern:

- ``values``    — truncation, base64/data-uri redaction, JSON parsing, and the
                  recursive ``safe_value`` bounding depth/width/length.
- ``messages``  — coercing/serializing the several message shapes Hermes emits,
                  and deriving a trace name from the latest user message.
- ``tools``     — serializing tool calls and assistant messages.
- ``read_file`` — compressing the ``read_file`` tool's verbose payload.

None of these functions touch plugin state.
"""

from __future__ import annotations

from .messages import (
    as_input_dict,
    content_to_text,
    coerce_request_messages,
    extract_last_user_message,
    responses_parts_to_text,
    serialize_messages,
    serialize_one_message,
    trace_name_from_messages,
)
from .read_file import (
    build_read_file_preview,
    looks_like_read_file_payload,
    normalize_read_file_payload,
    parse_read_file_lines,
)
from .tools import serialize_assistant_message, serialize_tool_calls
from .values import (
    is_base64_data_uri,
    maybe_parse_json_string,
    normalize_payload,
    redact_data_uri,
    safe_value,
    truncate_text,
)

__all__ = [
    # values
    "safe_value",
    "truncate_text",
    "maybe_parse_json_string",
    "is_base64_data_uri",
    "redact_data_uri",
    "normalize_payload",
    # messages
    "as_input_dict",
    "content_to_text",
    "coerce_request_messages",
    "extract_last_user_message",
    "responses_parts_to_text",
    "serialize_messages",
    "serialize_one_message",
    "trace_name_from_messages",
    # tools
    "serialize_assistant_message",
    "serialize_tool_calls",
    # read_file
    "build_read_file_preview",
    "looks_like_read_file_payload",
    "normalize_read_file_payload",
    "parse_read_file_lines",
]
