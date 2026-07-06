"""Root-trace lifecycle: create, flush-on-create, finalize.

The pre-hooks create a root trace per turn and (once) flush it as its own
batch; the turn-level post signal finalizes it. Kept separate from the hook
handlers so the create/finalize sequencing — and the batching-race notes that
motivate it — live in one place.
"""

from __future__ import annotations

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
    trace = client.trace(
        name=trace_name_from_messages(messages) or "Hermes turn",
        project_name=project_name(),
        thread_id=session_id or None,
        input=trace_input,
        metadata=metadata,
        tags=tags(),
    )
    # NOTE: the caller flushes this create (via flush_trace_create) AFTER
    # releasing the state lock. name/thread_id/input are set only at creation, so
    # the create must not coalesce with the turn's later update()+end() in one
    # batch window (a fast turn) or the trace lands NA (name=None/thread=None/
    # input=null) — the trace-level twin of the span NA-bug. flush() blocks on
    # the network, so it is deliberately kept out of the lock.
    debug(f"started trace {trace.id} for {task_key}")
    return TraceState(trace=trace, session_id=session_id)


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
        final_output = merge_trace_output(output, state)
        if final_output is not None:
            state.trace.update(
                output=final_output
                if isinstance(final_output, dict)
                else {"content": final_output}
            )
        state.trace.end()
    except Exception as exc:  # pragma: no cover - fail-open
        debug(f"finish trace failed: {exc}")
    finally:
        try:
            client.flush()
        except Exception:
            pass
