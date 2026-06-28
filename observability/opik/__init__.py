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

Module layout:
  config.py    - env/config helpers (_env, _debug, _project_name, _tags)
  keys.py      - trace-scope key construction (_trace_key, _request_key)
  sanitize.py  - pure payload sanitization / serialization (_safe_value, ...)
  usage.py     - token-usage + cost mapping via agent.usage_pricing
  __init__.py  - runtime state, Opik client, trace lifecycle, hooks, register
"""

from __future__ import annotations

import datetime
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import (  # noqa: F401
    _debug,
    _debug_enabled,
    _env,
    _env_bool,
    _project_name,
    _tags,
    logger,
)
from .keys import _request_key, _scope_prefix, _trace_key  # noqa: F401
from .sanitize import (  # noqa: F401
    _as_input_dict,
    _build_read_file_preview,
    _coerce_request_messages,
    _content_to_text,
    _extract_last_user_message,
    _trace_name_from_messages,
    _is_base64_data_uri,
    _looks_like_read_file_payload,
    _maybe_parse_json_string,
    _normalize_payload,
    _normalize_read_file_payload,
    _parse_read_file_lines,
    _redact_data_uri,
    _safe_value,
    _serialize_assistant_message,
    _serialize_messages,
    _serialize_tool_calls,
    _truncate_text,
)
from .usage import _cost_from_usage_dict, _opik_usage_from_canonical, _usage_and_cost  # noqa: F401

try:
    import opik as _opik_sdk
except Exception:  # pragma: no cover - fail-open when optional dep is missing
    _opik_sdk = None


@dataclass
class PendingTool:
    """A tool call seen at pre_tool_call, awaiting its result.

    We do NOT create the Opik span at pre-time. Fast tools (e.g. execute_code
    returning in milliseconds) fire post_tool_call within the SDK's batch
    window, so a span created at pre and ended at post loses its create message
    to the batching race — the span lands with no name/type/start_time ("NA").
    Instead we hold the start_time + input here and create the span once, fully
    formed (start + end together), at post-time.
    """

    start_time: datetime.datetime
    input: Any


@dataclass
class TraceState:
    trace: Any
    generations: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, PendingTool] = field(default_factory=dict)
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

# Sentinel: "_get_opik() has tried and failed". Lets us short-circuit every
# subsequent hook call without re-attempting SDK init. Runtime callers cannot
# reset the cache; if an operator fixes a misconfigured credential they must
# restart the process.
_INIT_FAILED = object()


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


# --- Trace lifecycle --------------------------------------------------------


def _start_root_trace(
    task_key: str,
    *,
    task_id: str,
    session_id: str,
    platform: str,
    provider: str,
    model: str,
    api_mode: str,
    messages: Any,
    client: Any,
    turn_id: str = "",
    api_request_id: str = "",
) -> TraceState:
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
        name=_trace_name_from_messages(messages) or "Hermes turn",
        project_name=_project_name(),
        thread_id=session_id or None,
        input=trace_input,
        metadata=metadata,
        tags=_tags(),
    )
    _debug(f"started trace {trace.id} for {task_key}")
    return TraceState(trace=trace)


def _end_observation(
    observation: Any,
    *,
    output: Any = None,
    metadata: Optional[dict] = None,
    usage: Optional[dict] = None,
    total_cost: Optional[float] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    error_info: Optional[dict] = None,
) -> None:
    if observation is None:
        return
    try:
        update_kwargs: Dict[str, Any] = {}
        if output is not None:
            update_kwargs["output"] = (
                output if isinstance(output, dict) else {"output": output}
            )
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
                output=final_output
                if isinstance(final_output, dict)
                else {"content": final_output}
            )
        state.trace.end()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"finish trace failed: {exc}")
    finally:
        try:
            client.flush()
        except Exception:
            pass


# --- Hooks ------------------------------------------------------------------


def on_pre_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    messages: Any = None,
    turn_type: str = "user",
    conversation_history: Any = None,
    user_message: Any = None,
    turn_id: str = "",
    api_request_id: str = "",
    **_: Any,
) -> None:
    # Current Hermes also fires pre_llm_call for context injection (no messages
    # list). Only the legacy request-shaped call carries a messages list; trace
    # that one so we don't create orphan root traces.
    if not isinstance(messages, list):
        return

    client = _get_opik()
    if client is None:
        return

    task_key = _trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _start_root_trace(
                task_key,
                task_id=task_id,
                session_id=session_id,
                platform=platform,
                provider=provider,
                model=model,
                api_mode=api_mode,
                messages=messages,
                client=client,
                turn_id=turn_id,
                api_request_id=api_request_id,
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

    task_key = _trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _start_root_trace(
                task_key,
                task_id=task_id,
                session_id=session_id,
                platform=platform,
                provider=provider,
                model=model,
                api_mode=api_mode,
                messages=input_messages,
                client=client,
                turn_id=turn_id,
                api_request_id=api_request_id,
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


def on_post_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    model: str = "",
    api_call_count: int = 0,
    assistant_message: Any = None,
    response: Any = None,
    api_duration: float = 0.0,
    finish_reason: str = "",
    usage: Any = None,
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    assistant_response: Any = None,
    turn_id: str = "",
    api_request_id: str = "",
    **_: Any,
) -> None:
    client = _get_opik()
    if client is None:
        return

    task_key = _trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(req_key, None) if state else None

    # Two callers share this handler:
    #   - post_api_request: per-API-call. Carries a generation span we opened in
    #     on_pre_llm_request (matched by req_key); we close that span here.
    #   - post_llm_call: per-TURN, fired once after the tool loop completes
    #     (agent/turn_finalizer.py). It carries `assistant_response` but no
    #     api_call_count, so it never matches a generation. This is the only
    #     reliable end-of-turn signal — finalize the root trace here, otherwise
    #     the trace is never .end()-ed and never surfaces as completed in Opik.
    if generation is None:
        if state is not None and assistant_response is not None:
            _finish_trace(task_key, output={"content": _safe_value(assistant_response)})
        return
    if state is None:
        return

    if assistant_message is not None:
        output = _serialize_assistant_message(assistant_message)
    elif assistant_response is not None:
        output = {
            "content": _safe_value(assistant_response),
            "reasoning": None,
            "tool_calls": [],
        }
    else:
        output = {
            "content": f"[{assistant_content_chars} chars]"
            if assistant_content_chars
            else None,
            "reasoning": None,
            "tool_calls": (
                [{"id": f"tc_{i}"} for i in range(assistant_tool_call_count)]
                if assistant_tool_call_count
                else []
            ),
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
            response,
            provider=provider,
            api_mode=api_mode,
            model=model,
            base_url=base_url,
        )
    elif isinstance(usage, dict) and usage:
        usage_details = _opik_usage_from_canonical(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0)
            or usage.get("completion_tokens", 0),
            cache_read=usage.get("cache_read_tokens", 0),
            cache_write=usage.get("cache_write_tokens", 0),
            reasoning=usage.get("reasoning_tokens", 0),
        )
        total_cost = _cost_from_usage_dict(
            usage, provider=provider, base_url=base_url, model=model
        )

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

    # NOTE: the root trace is finalized by the per-turn post_llm_call signal
    # (see the generation-is-None branch above), NOT here. Finalizing on a
    # per-API-call content heuristic would close the trace mid-turn — before
    # later tool calls / API calls in the same turn are recorded.


def on_pre_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    **_: Any,
) -> None:
    client = _get_opik()
    if client is None:
        return

    task_key = _trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )

    pending = PendingTool(
        start_time=datetime.datetime.now(datetime.timezone.utc),
        input=_as_input_dict(_safe_value(args)),
    )
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        # Record the start; the span is created (fully formed) at post-time to
        # avoid the create/end batching race for fast tools. See PendingTool.
        if tool_call_id:
            state.tools[tool_call_id] = pending
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(pending)


def on_post_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    **_: Any,
) -> None:
    task_key = _trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )
    pending = None

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        if tool_call_id:
            pending = state.tools.pop(tool_call_id, None)
        if pending is None:
            queue = state.pending_tools_by_name.get(tool_name)
            if queue:
                pending = queue.pop(0)
                if not queue:
                    state.pending_tools_by_name.pop(tool_name, None)

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

    # Create the span fully formed in a single message — start_time from the
    # recorded pre, end_time now — so the create/end batching race can't strip
    # its name/type/start_time. If pre was missed, fall back to now() for start.
    span_input = pending.input if pending else _as_input_dict(_safe_value(args))
    start_time = (
        pending.start_time if pending else datetime.datetime.now(datetime.timezone.utc)
    )
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        state.trace.span(
            name=f"Tool: {tool_name}",
            type="tool",
            input=span_input,
            output=safe_result_value
            if isinstance(safe_result_value, dict)
            else {"output": safe_result_value},
            start_time=start_time,
            end_time=datetime.datetime.now(datetime.timezone.utc),
            metadata={
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "args": _safe_value(args, parse_json_strings=True),
            },
        )


def on_api_request_error(
    *,
    task_id: str = "",
    session_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    model: str = "",
    provider: str = "",
    api_duration: float = 0.0,
    status_code: Any = None,
    retry_count: Any = None,
    max_retries: Any = None,
    retryable: Any = None,
    reason: str = "",
    error: Any = None,
    **_: Any,
) -> None:
    """Record a failed API request on its open generation span.

    Without this, a failed LLM call leaves the generation span we opened in
    on_pre_llm_request dangling with no outcome. Here we attach error_info +
    the failure metadata and close the span. The trace itself is left open —
    Hermes retries/falls back, and the turn still finalizes via post_llm_call.
    """
    client = _get_opik()
    if client is None:
        return

    err = error if isinstance(error, dict) else {}
    message = str(err.get("message") or reason or "API request error")
    exc_type = str(err.get("type") or "APIRequestError")

    task_key = _trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(req_key, None) if state else None
        if state is not None:
            state.last_updated_at = time.time()

    if generation is None:
        return

    error_meta: Dict[str, Any] = {
        "status_code": status_code,
        "retry_count": retry_count,
        "max_retries": max_retries,
        "retryable": retryable,
        "reason": reason,
    }
    if api_duration and api_duration > 0:
        error_meta["api_duration_s"] = round(api_duration, 3)
    _end_observation(
        generation,
        output={"error": message},
        model=model or None,
        provider=provider or None,
        error_info={
            "exception_type": exc_type,
            "message": message,
            "traceback": message,
        },
        metadata={k: v for k, v in error_meta.items() if v is not None},
    )


def on_subagent_stop(
    *,
    parent_session_id: str = "",
    child_role: Any = None,
    child_summary: Any = None,
    child_status: str = "",
    duration_ms: Any = None,
    task_id: str = "",
    session_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    **_: Any,
) -> None:
    """Record a finished subagent as a span under the parent's trace.

    Hermes spawns isolated subagents for parallel work; without this they are
    invisible in Opik. The subagent's own LLM/tool calls run in a separate
    process/session, so all we get here is a summary at completion — logged as
    one span on whichever active trace belongs to the parent session.
    """
    client = _get_opik()
    if client is None:
        return

    # Find the parent's active trace: prefer an explicit turn/session key,
    # else fall back to the parent_session_id's most recent trace.
    parent = parent_session_id or session_id
    with _STATE_LOCK:
        state = None
        if task_id or turn_id or api_request_id or session_id:
            state = _TRACE_STATE.get(
                _trace_key(
                    task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
                )
            )
        if state is None and parent:
            candidates = [
                s
                for k, s in _TRACE_STATE.items()
                if k.startswith(f"session:{parent}") or k == parent
            ]
            state = (
                max(candidates, key=lambda s: s.last_updated_at) if candidates else None
            )
        if state is None:
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        start_time = now
        if isinstance(duration_ms, (int, float)) and duration_ms > 0:
            start_time = now - datetime.timedelta(milliseconds=float(duration_ms))
        state.trace.span(
            name=f"Subagent: {child_role or 'subagent'}",
            type="general",
            input={"role": _safe_value(child_role)} if child_role else {},
            output={"summary": _safe_value(child_summary)} if child_summary else None,
            start_time=start_time,
            end_time=now,
            metadata={
                "subagent": True,
                "child_role": child_role,
                "child_status": child_status,
                "duration_ms": duration_ms,
                "parent_session_id": parent,
            },
        )
        state.last_updated_at = time.time()


def on_session_end(
    *, session_id: str = "", task_id: str = "", turn_id: str = "", **_: Any
) -> None:
    """Flush at an explicit session boundary so nothing is left buffered.

    Turns finalize on post_llm_call, but a session ending (CLI exit, /reset,
    gateway drain) is the last chance to push any still-open trace. We finalize
    any trace still tracked for this session and flush.
    """
    client = _get_opik()
    if client is None:
        return
    _debug(f"session end: {session_id or task_id}")
    if not session_id:
        return
    with _STATE_LOCK:
        keys = [k for k in _TRACE_STATE if k.startswith(f"session:{session_id}")]
    for key in keys:
        _finish_trace(key)
    try:
        client.flush()
    except Exception:
        pass


def on_session_event(*, session_id: str = "", **_: Any) -> None:
    """Lightweight breadcrumb for session start / finalize / reset.

    Observer-only: these mark conversation boundaries. We just log them under
    debug so the lifecycle is visible without creating spurious traces.
    """
    if _debug_enabled():
        _debug(f"session event for {session_id}")


def register(ctx) -> None:
    # Per-API-call (preferred) and per-turn LLM hooks. pre_api_request /
    # post_api_request fire per API call; pre_llm_call / post_llm_call once
    # per turn (post_llm_call is the reliable end-of-turn finalize signal).
    ctx.register_hook("pre_api_request", on_pre_llm_request)
    ctx.register_hook("post_api_request", on_post_llm_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    # Failure + agentic-structure coverage.
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    # Session lifecycle: flush on end; breadcrumbs for start/finalize/reset.
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_start", on_session_event)
    ctx.register_hook("on_session_finalize", on_session_end)
    ctx.register_hook("on_session_reset", on_session_event)
    # NOTE: transform_* hooks are intentionally NOT registered — they mutate
    # agent output (return a value that replaces the response / tool result),
    # which a passive observability plugin must never do. Approval, kanban, and
    # gateway-dispatch hooks are likewise out of scope for tracing.
