"""opik — Hermes plugin for Opik observability.

Traces Hermes conversations, LLM calls, and tool usage to Opik
(https://github.com/comet-ml/opik), Comet's open-source LLM observability
platform.

Activation is handled by the Hermes plugin system — standalone plugins only
load when listed in ``plugins.enabled`` (via ``hermes plugins enable
observability/opik`` or ``hermes tools → Opik Observability``). At runtime the
plugin also requires the ``opik`` SDK and, for Comet-hosted Opik, credentials;
if the SDK is missing the hooks are inert.

Configuration (set via ``hermes tools`` or ~/.hermes/.env):
  OPIK_API_KEY        - Opik/Comet API key. NOT required for a local
                        open-source Opik (http://localhost:5174/api) — only
                        for Comet-hosted or self-hosted-with-auth deployments.
  OPIK_URL_OVERRIDE   - Opik API base URL. Defaults to the Opik SDK default
                        (Comet cloud). For local Opik set e.g.
                        http://localhost:5174/api.
  OPIK_WORKSPACE      - Opik workspace name (Comet-hosted only).
  OPIK_PROJECT_NAME   - Project to log traces under (default: "hermes").

Optional env vars:
  HERMES_OPIK_TAGS         - comma-separated tags applied to every trace
  HERMES_OPIK_SAMPLE_RATE  - reserved for future use
  HERMES_OPIK_MAX_CHARS    - max chars per field (default: 12000)
  HERMES_OPIK_DEBUG        - set to "true" for verbose plugin logging
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import opik as _opik_sdk
except Exception:  # pragma: no cover - fail-open when optional dep is missing
    _opik_sdk = None


@dataclass
class TraceState:
    trace: Any
    generations: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, list] = field(default_factory=dict)
    turn_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.Lock()
_TRACE_STATE: Dict[str, TraceState] = {}
# Hard cap on live trace state. Each turn keys _TRACE_STATE by a unique
# turn_id, and an entry is normally reclaimed by _finish_trace when a turn
# ends cleanly (final response has content and no tool calls). A turn that
# never reaches that state — interrupted, a tool-only final step, or empty
# final content — would otherwise linger forever, so over the cap we evict
# the least-recently-updated entries (ending their trace first). The cap is
# far above any realistic concurrent-live-turn working set; it exists only to
# bound the leak from non-finalizing turns, not to limit concurrency.
_MAX_TRACE_STATE = 256
_OPIK_CLIENT = None
_READ_FILE_LINE_RE = re.compile(r"^\s*(\d+)\|(.*)$")
_READ_FILE_HEAD_LINES = 25
_READ_FILE_TAIL_LINES = 15

# Sentinel: "_get_opik() has tried and failed". Lets us short-circuit every
# subsequent hook call without re-attempting SDK init. Runtime callers cannot
# reset the cache; if an operator fixes a misconfigured credential they must
# restart the process.
_INIT_FAILED = object()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(*names: str) -> bool:
    for name in names:
        value = _env(name).lower()
        if value:
            return value in {"1", "true", "yes", "on"}
    return False


def _debug_enabled() -> bool:
    return _env_bool("HERMES_OPIK_DEBUG")


def _debug(message: str) -> None:
    if _debug_enabled():
        logger.info("Opik tracing: %s", message)


def _project_name() -> str:
    return _env("OPIK_PROJECT_NAME") or "hermes"


def _tags() -> list[str]:
    raw = _env("HERMES_OPIK_TAGS")
    extra = [t.strip() for t in raw.split(",") if t.strip()] if raw else []
    return ["hermes", *extra]


def _get_opik() -> Optional[Any]:
    """Return a cached Opik client, or ``None`` if unavailable.

    Activation of this plugin is controlled by the Hermes plugin system —
    this function only handles the runtime-availability gate (SDK installed).
    The result is cached: on the first call we try to construct a client, and
    every subsequent call returns that client (or fast-returns ``None`` if
    init failed).

    A local open-source Opik needs no API key, so — unlike the Langfuse plugin
    — we do NOT gate on credentials here. The Opik SDK reads OPIK_API_KEY /
    OPIK_URL_OVERRIDE / OPIK_WORKSPACE from the environment itself.
    """
    global _OPIK_CLIENT
    if _OPIK_CLIENT is _INIT_FAILED:
        return None
    if _OPIK_CLIENT is not None:
        return _OPIK_CLIENT

    if _opik_sdk is None:
        _OPIK_CLIENT = _INIT_FAILED
        return None

    try:
        _OPIK_CLIENT = _opik_sdk.Opik(project_name=_project_name())
    except Exception as exc:  # pragma: no cover - fail-open
        logger.warning("Could not initialize Opik client: %s", exc)
        _OPIK_CLIENT = _INIT_FAILED
        return None

    return _OPIK_CLIENT


def _scope_prefix(task_id: str, session_id: str) -> str:
    """The task/session/thread prefix shared by every trace-key shape."""
    if task_id:
        return f"task:{task_id}"
    if session_id:
        return f"session:{session_id}"
    return f"thread:{threading.get_ident()}"


def _trace_key(
    task_id: str,
    session_id: str,
    *,
    turn_id: str = "",
    api_request_id: str = "",
) -> str:
    """Build a stable in-process trace scope key for one agent turn.

    Older Hermes paths only expose ``task_id``/``session_id``. Newer paths
    pass ``turn_id`` and ``api_request_id`` in LLM/tool hooks; when present,
    they must scope trace state so concurrent requests sharing one task/session
    never collide. ``turn_id`` is preferred over ``api_request_id`` so the
    turn-level ``post_llm_call`` hook (which carries ``turn_id`` but no
    ``api_request_id``) resolves to the same key as the request-level hooks.
    """
    if turn_id:
        return f"{_scope_prefix(task_id, session_id)}:turn:{turn_id}"
    if api_request_id:
        return f"{_scope_prefix(task_id, session_id)}:api:{api_request_id}"
    if task_id:
        return task_id
    return _scope_prefix(task_id, session_id)


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
        "omitted_line_count": len(lines) - _READ_FILE_HEAD_LINES - _READ_FILE_TAIL_LINES,
    }


def _normalize_read_file_payload(value: dict[str, Any], *, args: Any = None) -> dict[str, Any]:
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


def _safe_value(value: Any, *, max_chars: Optional[int] = None, depth: int = 0,
                parse_json_strings: bool = False) -> Any:
    max_chars = max_chars if max_chars is not None else int(_env("HERMES_OPIK_MAX_CHARS", "12000") or "12000")
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
                return _safe_value(parsed, max_chars=max_chars, depth=depth, parse_json_strings=True)
        return _truncate_text(value, max_chars)
    if isinstance(value, dict):
        normalized = _normalize_payload(value)
        if normalized is not value:
            return _safe_value(normalized, max_chars=max_chars, depth=depth, parse_json_strings=parse_json_strings)
        return {
            str(k): _safe_value(v, max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings)
            for k, v in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _safe_value(v, max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings)
            for v in list(value)[:50]
        ]
    if hasattr(value, "__dict__"):
        return _safe_value(vars(value), max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings)
    return _truncate_text(repr(value), max_chars)


def _extract_last_user_message(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return {"role": "user", "content": _safe_value(message.get("content"))}
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


def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    serialized = []
    for message in messages[-12:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        item = {
            "role": role,
            "content": _safe_value(message.get("content"), parse_json_strings=(role == "tool")),
        }
        if role == "tool":
            if message.get("tool_call_id"):
                item["tool_call_id"] = message.get("tool_call_id")
            if message.get("name"):
                item["name"] = _safe_value(message.get("name"))
        if message.get("tool_calls"):
            item["tool_calls"] = _safe_value(message.get("tool_calls"), parse_json_strings=True)
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
        serialized.append({
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", None) or "function",
            "name": name,
            "arguments": safe_arguments,
            "function": {"name": name, "arguments": safe_arguments},
        })
    return serialized


def _serialize_assistant_message(message: Any) -> dict[str, Any]:
    return {
        "content": _safe_value(getattr(message, "content", None)),
        "reasoning": _safe_value(getattr(message, "reasoning", None)),
        "tool_calls": _serialize_tool_calls(getattr(message, "tool_calls", None)),
    }


def _usage_and_cost(response: Any, *, provider: str, api_mode: str, model: str,
                    base_url: str) -> tuple[dict[str, int], Optional[float]]:
    """Return (opik_usage_dict, total_cost_usd) from a provider response.

    Reuses Hermes' own ``agent.usage_pricing`` for canonical token counts and
    USD cost, then maps to the OpenAI-style keys Opik expects so input/output/
    total tokens render in the UI. ``total_cost`` is passed to Opik directly
    (it takes priority over Opik's own usage-derived estimate).
    """
    usage_details: Dict[str, int] = {}
    total_cost: Optional[float] = None
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return usage_details, total_cost

    try:
        from agent.usage_pricing import estimate_usage_cost, normalize_usage

        canonical = normalize_usage(raw_usage, provider=provider, api_mode=api_mode)
        usage_details = _opik_usage_from_canonical(
            input_tokens=canonical.input_tokens,
            output_tokens=canonical.output_tokens,
            cache_read=canonical.cache_read_tokens,
            cache_write=canonical.cache_write_tokens,
            reasoning=canonical.reasoning_tokens,
        )
        cost = estimate_usage_cost(model, canonical, provider=provider, base_url=base_url, api_key="")
        if cost.amount_usd is not None:
            total_cost = float(cost.amount_usd)
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"usage normalization failed: {exc}")

    return usage_details, total_cost


def _opik_usage_from_canonical(*, input_tokens: int, output_tokens: int,
                               cache_read: int = 0, cache_write: int = 0,
                               reasoning: int = 0) -> dict[str, int]:
    """Build an Opik-friendly usage dict using OpenAI-style key names.

    Opik surfaces input/output/total tokens when the usage dict carries the
    OpenAI keys ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.
    Cache and reasoning tokens are passed through under their canonical names
    for completeness.
    """
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write:
        usage["cache_creation_input_tokens"] = cache_write
    if reasoning:
        usage["reasoning_tokens"] = reasoning
    return usage


def _start_root_trace(task_key: str, *, task_id: str, session_id: str, platform: str,
                      provider: str, model: str, api_mode: str, messages: Any, client: Any,
                      turn_id: str = "", api_request_id: str = "") -> TraceState:
    trace_input = _extract_last_user_message(messages)
    metadata = {
        "source": "hermes",
        "task_id": task_id,
        "turn_id": turn_id,
        "api_request_id": api_request_id,
        "platform": platform,
        "provider": provider,
        "model": model,
        "api_mode": api_mode,
    }
    trace = client.trace(
        name="Hermes turn",
        project_name=_project_name(),
        thread_id=session_id or None,
        input=trace_input,
        metadata=metadata,
        tags=_tags(),
    )
    _debug(f"started trace {trace.id} for {task_key}")
    return TraceState(trace=trace)


def _end_observation(observation: Any, *, output: Any = None, metadata: Optional[dict] = None,
                     usage: Optional[dict] = None, total_cost: Optional[float] = None,
                     model: Optional[str] = None, provider: Optional[str] = None,
                     error_info: Optional[dict] = None) -> None:
    if observation is None:
        return
    try:
        update_kwargs: Dict[str, Any] = {}
        if output is not None:
            update_kwargs["output"] = output if isinstance(output, dict) else {"output": output}
        if metadata:
            update_kwargs["metadata"] = metadata
        if usage:
            update_kwargs["usage"] = usage
        if total_cost is not None:
            update_kwargs["total_cost"] = total_cost
        if model:
            update_kwargs["model"] = model
        if provider:
            update_kwargs["provider"] = provider
        if error_info:
            update_kwargs["error_info"] = error_info
        if update_kwargs:
            observation.update(**update_kwargs)
        observation.end()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"end observation failed: {exc}")


def _merge_trace_output(output: Any, state: TraceState) -> Any:
    if not state.turn_tool_calls:
        return output
    merged = dict(output) if isinstance(output, dict) else {"content": output}
    merged["tool_calls"] = list(state.turn_tool_calls)
    return merged


def _evict_stale_locked() -> None:
    """Drop least-recently-updated trace state to make room for a new entry.

    Caller MUST hold ``_STATE_LOCK`` and call this immediately before inserting
    one new entry. Bounds the leak from turns that never reach ``_finish_trace``
    (interrupted / tool-only final step / empty final content), whose unique
    per-turn key would otherwise linger forever. The evicted entry's trace is
    ended so it is not left dangling on the Opik side.
    """
    over = len(_TRACE_STATE) - (_MAX_TRACE_STATE - 1)
    if over <= 0:
        return
    stale = sorted(_TRACE_STATE.items(), key=lambda kv: kv[1].last_updated_at)[:over]
    for key, state in stale:
        _TRACE_STATE.pop(key, None)
        try:
            state.trace.end()
        except Exception as exc:  # pragma: no cover - fail-open
            _debug(f"evict stale trace failed: {exc}")


def _finish_trace(task_key: str, *, output: Any = None) -> None:
    client = _get_opik()
    if client is None:
        return

    with _STATE_LOCK:
        state = _TRACE_STATE.pop(task_key, None)
    if state is None:
        return

    try:
        for observation in state.generations.values():
            _end_observation(observation)
        for observation in state.tools.values():
            _end_observation(observation)
        for queue in state.pending_tools_by_name.values():
            for observation in queue:
                _end_observation(observation)
        final_output = _merge_trace_output(output, state)
        if final_output is not None:
            state.trace.update(
                output=final_output if isinstance(final_output, dict) else {"content": final_output}
            )
        state.trace.end()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"finish trace failed: {exc}")
    finally:
        try:
            client.flush()
        except Exception:
            pass


def _assistant_has_tool_calls(message: Any) -> bool:
    return bool(getattr(message, "tool_calls", None))


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)


def on_pre_llm_call(*, task_id: str = "", session_id: str = "", platform: str = "", model: str = "",
                    provider: str = "", base_url: str = "", api_mode: str = "",
                    api_call_count: int = 0, messages: Any = None, turn_type: str = "user",
                    conversation_history: Any = None, user_message: Any = None,
                    turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    # Current Hermes also fires pre_llm_call for context injection (no messages
    # list). Only the legacy request-shaped call carries a messages list; trace
    # that one so we don't create orphan root traces.
    if not isinstance(messages, list):
        return

    client = _get_opik()
    if client is None:
        return

    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _start_root_trace(
                task_key, task_id=task_id, session_id=session_id, platform=platform,
                provider=provider, model=model, api_mode=api_mode, messages=messages,
                client=client, turn_id=turn_id, api_request_id=api_request_id,
            )
            _evict_stale_locked()
            _TRACE_STATE[task_key] = state
        state.last_updated_at = time.time()


def on_pre_llm_request(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    request_messages: Any = None,
    messages: Any = None,
    turn_type: str = "user",
    message_count: int = 0,
    tool_count: int = 0,
    approx_input_tokens: int = 0,
    request_char_count: int = 0,
    max_tokens: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
    turn_id: str = "",
    api_request_id: str = "",
    **_: Any,
) -> None:
    client = _get_opik()
    if client is None:
        return

    input_messages = _coerce_request_messages(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
    )

    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _start_root_trace(
                task_key, task_id=task_id, session_id=session_id, platform=platform,
                provider=provider, model=model, api_mode=api_mode, messages=input_messages,
                client=client, turn_id=turn_id, api_request_id=api_request_id,
            )
            _evict_stale_locked()
            _TRACE_STATE[task_key] = state
        state.last_updated_at = time.time()
        previous = state.generations.pop(req_key, None)
        if previous is not None:
            _end_observation(previous)
        state.generations[req_key] = state.trace.span(
            name=f"LLM call {api_call_count}",
            type="llm",
            input={"messages": _serialize_messages(input_messages)},
            metadata={"platform": platform, "api_mode": api_mode, "base_url": base_url},
            model=model,
            provider=provider or None,
        )


def on_post_llm_call(*, task_id: str = "", session_id: str = "", provider: str = "", base_url: str = "",
                     api_mode: str = "", model: str = "", api_call_count: int = 0,
                     assistant_message: Any = None, response: Any = None,
                     api_duration: float = 0.0, finish_reason: str = "",
                     usage: Any = None, assistant_content_chars: int = 0,
                     assistant_tool_call_count: int = 0, assistant_response: Any = None,
                     turn_id: str = "", api_request_id: str = "",
                     **_: Any) -> None:
    client = _get_opik()
    if client is None:
        return

    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(req_key, None) if state else None
    if state is None or generation is None:
        return

    if assistant_message is not None:
        output = _serialize_assistant_message(assistant_message)
    elif assistant_response is not None:
        output = {"content": _safe_value(assistant_response), "reasoning": None, "tool_calls": []}
    else:
        output = {
            "content": f"[{assistant_content_chars} chars]" if assistant_content_chars else None,
            "reasoning": None,
            "tool_calls": [{"id": f"tc_{i}"} for i in range(assistant_tool_call_count)] if assistant_tool_call_count else [],
        }

    if output.get("tool_calls"):
        state.turn_tool_calls.extend(output["tool_calls"])

    # Prefer a real response object that carries .usage; else fall back to the
    # usage summary dict from post_api_request (post_api_request passes
    # `response` as a sanitized dict with no .usage attribute).
    usage_details: dict[str, int] = {}
    total_cost: Optional[float] = None
    if getattr(response, "usage", None) is not None:
        usage_details, total_cost = _usage_and_cost(
            response, provider=provider, api_mode=api_mode, model=model, base_url=base_url,
        )
    elif isinstance(usage, dict) and usage:
        _input = usage.get("input_tokens", 0)
        _output = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        usage_details = _opik_usage_from_canonical(
            input_tokens=_input,
            output_tokens=_output,
            cache_read=usage.get("cache_read_tokens", 0),
            cache_write=usage.get("cache_write_tokens", 0),
            reasoning=usage.get("reasoning_tokens", 0),
        )
        try:
            from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
            _cu = CanonicalUsage(
                input_tokens=_input,
                output_tokens=_output,
                cache_read_tokens=usage.get("cache_read_tokens", 0),
                cache_write_tokens=usage.get("cache_write_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
            )
            _cost = estimate_usage_cost(model, _cu, provider=provider, base_url=base_url, api_key="")
            if _cost.amount_usd is not None:
                total_cost = float(_cost.amount_usd)
        except Exception:
            pass

    tool_count = len(output.get("tool_calls", [])) or assistant_tool_call_count
    gen_metadata: Dict[str, Any] = {"tool_call_count": tool_count}
    if api_duration and api_duration > 0:
        gen_metadata["api_duration_s"] = round(api_duration, 3)
    if finish_reason:
        gen_metadata["finish_reason"] = finish_reason
    _end_observation(
        generation,
        output=output,
        usage=usage_details or None,
        total_cost=total_cost,
        model=model or None,
        provider=provider or None,
        metadata=gen_metadata,
    )

    has_tools = _assistant_has_tool_calls(assistant_message) if assistant_message else (assistant_tool_call_count > 0)
    has_content = bool(output.get("content"))
    if not has_tools and has_content:
        _finish_trace(task_key, output=output)


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, task_id: str = "",
                     session_id: str = "", tool_call_id: str = "",
                     turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    client = _get_opik()
    if client is None:
        return

    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        observation = state.trace.span(
            name=f"Tool: {tool_name}",
            type="tool",
            input=_as_input_dict(_safe_value(args)),
            metadata={"tool_name": tool_name, "tool_call_id": tool_call_id},
        )
        if tool_call_id:
            state.tools[tool_call_id] = observation
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(observation)


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None,
                      task_id: str = "", session_id: str = "", tool_call_id: str = "",
                      turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    observation = None

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        if tool_call_id:
            observation = state.tools.pop(tool_call_id, None)
        if observation is None:
            queue = state.pending_tools_by_name.get(tool_name)
            if queue:
                observation = queue.pop(0)
                if not queue:
                    state.pending_tools_by_name.pop(tool_name, None)

    if observation is None:
        return

    if isinstance(result, str):
        result_value = _maybe_parse_json_string(result)
    else:
        result_value = result
    result_value = _normalize_payload(result_value, tool_name=tool_name, args=args)
    safe_result_value = _safe_value(result_value, parse_json_strings=True)

    # Backfill so the generation's tool_call record carries the result.
    if tool_call_id:
        with _STATE_LOCK:
            state = _TRACE_STATE.get(task_key)
            if state is not None:
                for tool_call in reversed(state.turn_tool_calls):
                    if tool_call.get("id") == tool_call_id:
                        tool_call["output"] = safe_result_value
                        function_payload = tool_call.get("function")
                        if isinstance(function_payload, dict):
                            function_payload["output"] = safe_result_value
                        break

    _end_observation(
        observation,
        output=safe_result_value,
        metadata={"tool_name": tool_name, "args": _safe_value(args, parse_json_strings=True)},
    )


def _as_input_dict(value: Any) -> dict[str, Any]:
    """Opik span input is a dict; wrap non-dict values under a stable key."""
    if isinstance(value, dict):
        return value
    return {"input": value}


def register(ctx) -> None:
    # Register for both hook-name variants so the plugin works across Hermes
    # versions. pre_api_request / post_api_request fire per API call
    # (preferred); pre_llm_call / post_llm_call fire once per turn.
    ctx.register_hook("pre_api_request", on_pre_llm_request)
    ctx.register_hook("post_api_request", on_post_llm_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
