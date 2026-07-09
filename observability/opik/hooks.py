"""Hermes hook handlers and plugin registration.

The thin entry layer: each handler maps a Hermes lifecycle event onto the
trace/span primitives. Handlers create fully-formed spans in a single message
(start + end together) to dodge the SDK's create/end batching race for fast
calls, and hold pending records in the trace state between the pre- and
post-hooks. All handlers fail open — if the client is unavailable they no-op.
"""

from __future__ import annotations

import datetime
import time
from typing import Any, Dict, Optional

from .client import get_client
from .config import debug, debug_enabled
from .keys import request_key, trace_key
from .lifecycle import finish_trace, flush_trace_create, start_root_trace
from .providers import to_opik_provider
from .sanitize import (
    as_input_dict,
    coerce_request_messages,
    maybe_parse_json_string,
    normalize_payload,
    safe_value,
    serialize_assistant_message,
    serialize_messages,
)
from .state import (
    PendingGeneration,
    PendingTool,
    TraceState,
    ensure_trace_state,
    lock,
    states_for_session,
    store,
)
from .usage import cost_from_usage_dict, opik_usage_from_canonical, usage_and_cost


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

    client = get_client()
    if client is None:
        return

    task_key = trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )

    _, created = ensure_trace_state(
        task_key,
        lambda: start_root_trace(
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
        ),
    )
    if created:
        flush_trace_create(client)


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
    client = get_client()
    if client is None:
        return

    input_messages = coerce_request_messages(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
    )

    task_key = trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )
    req_key = request_key(api_call_count)

    def record_generation(state: "TraceState") -> None:
        # Record the call; the span is created (fully formed) at post-time to
        # avoid the create/end batching race for fast API calls. A duplicate
        # req_key just overwrites the pending record (no span to end yet).
        state.generations[req_key] = PendingGeneration(
            start_time=datetime.datetime.now(datetime.timezone.utc),
            api_call_count=api_call_count,
            input={"messages": serialize_messages(input_messages)},
            metadata={"platform": platform, "api_mode": api_mode, "base_url": base_url},
            model=model,
            provider=provider or "",
        )

    _, created = ensure_trace_state(
        task_key,
        lambda: start_root_trace(
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
        ),
        on_state=record_generation,
    )
    if created:
        flush_trace_create(client)


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
    client = get_client()
    if client is None:
        return

    task_key = trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )
    req_key = request_key(api_call_count)

    with lock:
        state = store.get(task_key)
        pending = state.generations.pop(req_key, None) if state else None

    # Two callers share this handler:
    #   - post_api_request: per-API-call. Matches the PendingGeneration we
    #     recorded in on_pre_llm_request (by req_key); we create+end its span
    #     here, in one message, to avoid the batching race.
    #   - post_llm_call: per-TURN, fired once after the tool loop completes
    #     (agent/turn_finalizer.py). It carries `assistant_response` but no
    #     api_call_count, so it never matches a PendingGeneration. This is the
    #     only reliable end-of-turn signal — finalize the root trace here,
    #     otherwise the trace never gets its finalize re-send (output +
    #     end_time) and never surfaces as completed.
    if pending is None:
        if state is not None and assistant_response is not None:
            finish_trace(task_key, output={"content": safe_value(assistant_response)})
        return
    if state is None:
        return

    if assistant_message is not None:
        output = serialize_assistant_message(assistant_message)
    elif assistant_response is not None:
        output = {
            "content": safe_value(assistant_response),
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
        usage_details, total_cost = usage_and_cost(
            response,
            provider=provider,
            api_mode=api_mode,
            model=model,
            base_url=base_url,
        )
    elif isinstance(usage, dict) and usage:
        usage_details = opik_usage_from_canonical(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0)
            or usage.get("completion_tokens", 0),
            cache_read=usage.get("cache_read_tokens", 0),
            cache_write=usage.get("cache_write_tokens", 0),
            reasoning=usage.get("reasoning_tokens", 0),
        )
        total_cost = cost_from_usage_dict(
            usage, provider=provider, base_url=base_url, model=model
        )

    tool_count = len(output.get("tool_calls", [])) or assistant_tool_call_count
    gen_metadata: Dict[str, Any] = dict(pending.metadata)
    gen_metadata["tool_call_count"] = tool_count
    if api_duration and api_duration > 0:
        gen_metadata["api_duration_s"] = round(api_duration, 3)
    if finish_reason:
        gen_metadata["finish_reason"] = finish_reason

    # Create the generation span fully formed in a single message (start_time
    # from the recorded pre, end_time now) so a fast API call can't lose its
    # create to the batching race (the NA-span bug, same as tool spans).
    with lock:
        state = store.get(task_key)
        if state is None:
            return
        state.trace.span(
            name=f"LLM call {pending.api_call_count}",
            type="llm",
            input=pending.input,
            output=output,
            usage=usage_details or None,
            total_cost=total_cost,
            model=model or pending.model or None,
            provider=to_opik_provider(provider or pending.provider or None),
            metadata=gen_metadata,
            start_time=pending.start_time,
            end_time=datetime.datetime.now(datetime.timezone.utc),
        )

    # NOTE: the root trace is finalized by the per-turn post_llm_call signal
    # (see the pending-is-None branch above), NOT here. Finalizing on a
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
    client = get_client()
    if client is None:
        return

    task_key = trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )

    pending = PendingTool(
        start_time=datetime.datetime.now(datetime.timezone.utc),
        input=as_input_dict(safe_value(args)),
    )
    with lock:
        state = store.get(task_key)
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
    task_key = trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )
    pending = None

    with lock:
        state = store.get(task_key)
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
        result_value = maybe_parse_json_string(result)
    else:
        result_value = result
    result_value = normalize_payload(result_value, tool_name=tool_name, args=args)
    safe_result_value = safe_value(result_value, parse_json_strings=True)

    # Backfill so the generation's tool_call record carries the result.
    if tool_call_id:
        with lock:
            state = store.get(task_key)
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
    span_input = pending.input if pending else as_input_dict(safe_value(args))
    start_time = (
        pending.start_time if pending else datetime.datetime.now(datetime.timezone.utc)
    )
    with lock:
        state = store.get(task_key)
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
                "args": safe_value(args, parse_json_strings=True),
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
    """Record a failed API request as the generation span for that call.

    Without this, a failed LLM call would leave its PendingGeneration unconsumed
    and never produce a span. We create the generation span here (one message,
    start from the recorded pre + end now) carrying error_info and the failure
    metadata. The trace itself is left open — Hermes retries/falls back, and the
    turn still finalizes via post_llm_call.
    """
    client = get_client()
    if client is None:
        return

    err = error if isinstance(error, dict) else {}
    message = str(err.get("message") or reason or "API request error")
    exc_type = str(err.get("type") or "APIRequestError")

    task_key = trace_key(
        task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
    )
    req_key = request_key(api_call_count)

    with lock:
        state = store.get(task_key)
        pending = state.generations.pop(req_key, None) if state else None
        if state is None or pending is None:
            return
        state.last_updated_at = time.time()

        error_meta: Dict[str, Any] = dict(pending.metadata)
        for k, v in {
            "status_code": status_code,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "retryable": retryable,
            "reason": reason,
        }.items():
            if v is not None:
                error_meta[k] = v
        if api_duration and api_duration > 0:
            error_meta["api_duration_s"] = round(api_duration, 3)

        state.trace.span(
            name=f"LLM call {pending.api_call_count}",
            type="llm",
            input=pending.input,
            output={"error": message},
            model=model or pending.model or None,
            provider=to_opik_provider(provider or pending.provider or None),
            error_info={
                "exception_type": exc_type,
                "message": message,
                "traceback": message,
            },
            metadata=error_meta,
            start_time=pending.start_time,
            end_time=datetime.datetime.now(datetime.timezone.utc),
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
    client = get_client()
    if client is None:
        return

    # Find the parent's active trace: prefer an explicit turn/session key,
    # else fall back to the parent_session_id's most recent trace.
    parent = parent_session_id or session_id
    with lock:
        state = None
        if task_id or turn_id or api_request_id or session_id:
            state = store.get(
                trace_key(
                    task_id, session_id, turn_id=turn_id, api_request_id=api_request_id
                )
            )
        if state is None and parent:
            # Bind only when exactly one turn is live for the session — with
            # multiple concurrent turns there's no reliable way to pick the
            # parent here (recency is not it), so skip rather than attach to the
            # wrong trace.
            candidates = states_for_session(parent)
            state = candidates[0][1] if len(candidates) == 1 else None
        if state is None:
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        start_time = now
        if isinstance(duration_ms, (int, float)) and duration_ms > 0:
            start_time = now - datetime.timedelta(milliseconds=float(duration_ms))
        state.trace.span(
            name=f"Subagent: {child_role or 'subagent'}",
            type="general",
            input={"role": safe_value(child_role)} if child_role else {},
            output={"summary": safe_value(child_summary)} if child_summary else None,
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
    client = get_client()
    if client is None:
        return
    debug(f"session end: {session_id or task_id}")
    if not session_id:
        return
    with lock:
        keys = [k for k, _ in states_for_session(session_id)]
    for key in keys:
        finish_trace(key)
    try:
        client.flush()
    except Exception:
        pass


def on_session_event(*, session_id: str = "", **_: Any) -> None:
    """Lightweight breadcrumb for session start / finalize / reset.

    Observer-only: these mark conversation boundaries. We just log them under
    debug so the lifecycle is visible without creating spurious traces.
    """
    if debug_enabled():
        debug(f"session event for {session_id}")


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
