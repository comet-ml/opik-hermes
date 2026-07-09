"""Trace-key scoping precedence and LRU eviction at the state cap."""

from __future__ import annotations


def test_turn_id_takes_precedence(plugin):
    key = plugin.keys.trace_key("task1", "sess1", turn_id="T9", api_request_id="A9")
    assert key == "task:task1:turn:T9"


def test_api_request_id_when_no_turn(plugin):
    key = plugin.keys.trace_key("task1", "sess1", api_request_id="A9")
    assert key == "task:task1:api:A9"


def test_session_scope_prefix_when_no_task(plugin):
    key = plugin.keys.trace_key("", "sess1", turn_id="T1")
    assert key == "session:sess1:turn:T1"


def test_legacy_bare_task_id(plugin):
    # No turn/api ids -> legacy bare task_id (not the task: prefix).
    assert plugin.keys.trace_key("task1", "sess1") == "task1"


def test_concurrent_turns_dont_collide(plugin):
    a = plugin.keys.trace_key("t", "s", turn_id="T1")
    b = plugin.keys.trace_key("t", "s", turn_id="T2")
    assert a != b


def test_lru_eviction_bounds_state_and_finalizes_evicted_trace(plugin):
    cap = plugin.state.MAX_TRACE_STATE
    # Open cap+5 distinct turns; each opens a root trace via pre_llm_request.
    for i in range(cap + 5):
        plugin.on_pre_llm_request(
            task_id="t",
            session_id="s",
            turn_id=f"T{i}",
            api_call_count=1,
            messages=[{"role": "user", "content": f"m{i}"}],
            model="gpt-5",
        )
    # State never exceeds the cap...
    assert len(plugin.state.store) <= cap
    # ...and evicted traces were finalized via an upsert re-send (end_time,
    # same id), not the forbidden trace.end() — not left dangling on Opik.
    finalized = [t for t in plugin._fake.traces if t.finalized]
    assert len(finalized) >= 5
    assert all(t.finalize_kwargs.get("end_time") is not None for t in finalized)
    # The eviction re-send replays the full create payload (name/thread_id), so
    # the evicted trace isn't clobbered to an NA trace — same fix as finish.
    assert all(t.finalize_kwargs.get("name") for t in finalized)
    assert all(t.finalize_kwargs.get("thread_id") == "s" for t in finalized)
    assert not any(e[0] in ("trace.update", "trace.end") for e in plugin._fake.events)
