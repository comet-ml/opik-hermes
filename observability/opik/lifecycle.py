"""Root-trace lifecycle: create, flush-on-create, finalize.

The pre-hooks create a root trace per turn and (once) flush it as its own
batch; the turn-level post signal finalizes it. Kept separate from the hook
handlers so the create/finalize sequencing — and the batching-race notes that
motivate it — live in one place.
"""

from __future__ import annotations

import datetime
from typing import Any

from .client import get_client
from .config import debug, project_name, tags
from .sanitize import extract_last_user_message, trace_name_from_messages
from .state import TraceState, lock, store


def start_root_trace(
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
    trace_input = extract_last_user_message(messages)
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
    # Build the create payload once and keep it: the finalize/eviction re-send
    # (upsert via client.trace(id=...)) must replay this FULL payload, not just
    # output+end_time. An upsert is a whole CreateTraceMessage — any omitted
    # field is sent as null and the backend's last-write-wins merge clobbers the
    # create (name/thread_id -> NA trace). start_time is pinned here so the
    # re-send carries the same value instead of a fresh now() (the SDK defaults
    # start_time to now on every client.trace() call).
    create_kwargs: dict[str, Any] = {
        "name": trace_name_from_messages(messages) or "Hermes turn",
        "project_name": project_name(),
        "thread_id": session_id or None,
        "input": trace_input,
        "metadata": metadata,
        "tags": tags(),
        "start_time": datetime.datetime.now(datetime.timezone.utc),
    }
    trace = client.trace(**create_kwargs)
    # NOTE: the caller flushes this create (via flush_trace_create) AFTER
    # releasing the state lock. name/thread_id/input are set only at creation, so
    # the create must not coalesce with the turn's later finalize re-send in one
    # batch window (a fast turn) or the trace lands NA (name=None/thread=None/
    # input=null) — the trace-level twin of the span NA-bug. flush() blocks on
    # the network, so it is deliberately kept out of the lock.
    debug(f"started trace {trace.id} for {task_key}")
    return TraceState(trace=trace, session_id=session_id, create_kwargs=create_kwargs)


def flush_trace_create(client: Any) -> None:
    """Flush a just-created root trace as its own batch.

    Called by the pre-hooks AFTER releasing the state lock, and only when a new
    trace was created this call. Closing the create's batch keeps it from
    coalescing with the turn's later update()+end() (the NA-trace race). flush()
    is blocking/network-bound, so it must run unlocked to avoid serializing
    concurrent turns behind it.
    """
    try:
        client.flush()
    except Exception as exc:  # pragma: no cover - fail-open
        debug(f"flush after trace create failed: {exc}")


def merge_trace_output(output: Any, state: TraceState) -> Any:
    if not state.turn_tool_calls:
        return output
    merged = dict(output) if isinstance(output, dict) else {"content": output}
    merged["tool_calls"] = list(state.turn_tool_calls)
    return merged


def finish_trace(task_key: str, *, output: Any = None) -> None:
    client = get_client()
    if client is None:
        return

    with lock:
        state = store.pop(task_key, None)
    if state is None:
        return

    try:
        # Leftover generations/tools are PendingGeneration/PendingTool records
        # for calls that never received a post (interrupted turn). They have no
        # span yet — an in-flight call with no response isn't a meaningful span,
        # so we simply drop them rather than emit a partial span.
        #
        # Upsert-only finalize: re-send the SAME trace id with the finished
        # payload instead of trace.update()/trace.end(). The SDK's batching layer
        # coalesces this with the create into one final row. An update() shortly
        # after create trips the "may cause data loss" warning; the upsert is the
        # mandated pattern and avoids it.
        #
        # Replay the FULL create payload (name/thread_id/input/start_time/...)
        # plus output+end_time — NOT just output+end_time. An upsert is a whole
        # CreateTraceMessage; omitted fields go as null and the backend's
        # last-write-wins merge would clobber the create, landing an NA trace
        # (name=None/thread_id=None). See TraceState.create_kwargs.
        final_output = merge_trace_output(output, state)
        client.trace(
            id=state.trace.id,
            output=final_output
            if final_output is None or isinstance(final_output, dict)
            else {"content": final_output},
            end_time=datetime.datetime.now(datetime.timezone.utc),
            **state.create_kwargs,
        )
    except Exception as exc:  # pragma: no cover - fail-open
        debug(f"finish trace failed: {exc}")
    finally:
        try:
            client.flush()
        except Exception:
            pass
