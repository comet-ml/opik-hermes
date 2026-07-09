"""Per-turn trace state and the in-process store that holds it.

Each agent turn keys one ``TraceState`` (its root trace plus the pending
generations/tools awaiting their post-hook) by a unique per-turn key. The store
is guarded by a lock and bounded by a hard cap so turns that never finalize
can't leak forever.
"""

from __future__ import annotations

import datetime
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .client import get_client
from .config import debug

# Hard cap on live trace state. Each turn keys the store by a unique turn_id,
# and an entry is normally reclaimed by finish_trace when a turn ends cleanly
# (final response has content and no tool calls). A turn that never reaches that
# state — interrupted, a tool-only final step, or empty final content — would
# otherwise linger forever, so over the cap we evict the least-recently-updated
# entries (ending their trace first). The cap is far above any realistic
# concurrent-live-turn working set; it exists only to bound the leak from
# non-finalizing turns, not to limit concurrency.
MAX_TRACE_STATE = 256

lock = threading.Lock()
store: Dict[str, "TraceState"] = {}


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
class PendingGeneration:
    """An LLM API call seen at pre_api_request, awaiting its response.

    Same batching-race reasoning as PendingTool: a fast API call fires
    post_api_request within the SDK batch window, so a generation span created
    at pre and ended at post loses its create message and lands as an "NA" span
    (no name/type/start_time). We hold the creation fields here and build the
    span once, fully formed, at post-time.
    """

    start_time: datetime.datetime
    api_call_count: int
    input: Any
    metadata: Dict[str, Any]
    model: str
    provider: str


@dataclass
class TraceState:
    trace: Any
    session_id: str = ""
    generations: Dict[str, PendingGeneration] = field(default_factory=dict)
    tools: Dict[str, PendingTool] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, list] = field(default_factory=dict)
    turn_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    last_updated_at: float = field(default_factory=time.time)


def ensure_trace_state(
    task_key: str,
    factory: Callable[[], "TraceState"],
    *,
    on_state: Optional[Callable[["TraceState"], None]] = None,
) -> Tuple["TraceState", bool]:
    """Get-or-create the trace state for ``task_key`` under the store lock.

    Shared by the LLM pre-hooks. On a miss, ``factory()`` builds a new
    ``TraceState`` (a deferred ``start_root_trace``), the store is made room for
    via eviction, and the entry is inserted. ``last_updated_at`` is always
    touched. ``on_state`` runs under the same lock so a caller's extra
    bookkeeping (e.g. recording a pending generation) stays in the single lock
    acquisition. Returns ``(state, created)``; the caller flushes the create
    outside the lock when ``created`` is True (the NA-trace race — see
    ``lifecycle.flush_trace_create``).
    """
    with lock:
        state = store.get(task_key)
        created = state is None
        if created:
            state = factory()
            evict_stale_locked()
            store[task_key] = state
        state.last_updated_at = time.time()
        if on_state is not None:
            on_state(state)
    return state, created


def states_for_session(session_id: str) -> list[Tuple[str, "TraceState"]]:
    """Return ``(key, state)`` for every live trace belonging to ``session_id``.

    Matches on the ``session_id`` recorded on each ``TraceState``, not the store
    key: ``trace_key()`` keys most turns under ``task:{task_id}:...`` when a
    task_id is present, so a ``session:``-prefix key scan would miss them.
    Callers hold ``lock`` and apply their own policy — ``on_session_end``
    finalizes every match; ``on_subagent_stop`` binds only when there's exactly
    one — so the session-matching rule itself lives in one place.
    """
    return [(k, s) for k, s in store.items() if s.session_id == session_id]


def evict_stale_locked() -> None:
    """Drop least-recently-updated trace state to make room for a new entry.

    Caller MUST hold ``lock`` and call this immediately before inserting one new
    entry. Bounds the leak from turns that never reach ``finish_trace``
    (interrupted / tool-only final step / empty final content), whose unique
    per-turn key would otherwise linger forever. The evicted entry's trace is
    finalized via an upsert re-send (same id + end_time) so it is not left
    dangling on the Opik side — never trace.end(), which is the forbidden
    post-create-mutation that trips the batching "may cause data loss" warning.
    """
    over = len(store) - (MAX_TRACE_STATE - 1)
    if over <= 0:
        return
    client = get_client()
    stale = sorted(store.items(), key=lambda kv: kv[1].last_updated_at)[:over]
    for key, state in stale:
        store.pop(key, None)
        if client is None:
            continue
        try:
            client.trace(
                id=state.trace.id,
                end_time=datetime.datetime.now(datetime.timezone.utc),
            )
        except Exception as exc:  # pragma: no cover - fail-open
            debug(f"evict stale trace failed: {exc}")
