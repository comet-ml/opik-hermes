"""Trace-key scoping precedence and LRU eviction at the state cap."""

from __future__ import annotations


def test_turn_id_takes_precedence(plugin):
    key = plugin._trace_key("task1", "sess1", turn_id="T9", api_request_id="A9")
    assert key == "task:task1:turn:T9"


def test_api_request_id_when_no_turn(plugin):
    key = plugin._trace_key("task1", "sess1", api_request_id="A9")
    assert key == "task:task1:api:A9"


def test_session_scope_prefix_when_no_task(plugin):
    key = plugin._trace_key("", "sess1", turn_id="T1")
    assert key == "session:sess1:turn:T1"


def test_legacy_bare_task_id(plugin):
    # No turn/api ids -> legacy bare task_id (not the task: prefix).
    assert plugin._trace_key("task1", "sess1") == "task1"


def test_concurrent_turns_dont_collide(plugin):
    a = plugin._trace_key("t", "s", turn_id="T1")
    b = plugin._trace_key("t", "s", turn_id="T2")
    assert a != b


def test_lru_eviction_bounds_state_and_ends_evicted_trace(plugin):
    cap = plugin._MAX_TRACE_STATE
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
    assert len(plugin._TRACE_STATE) <= cap
    # ...and evicted traces were ended (not left dangling on the Opik side).
    ended = [t for t in plugin._fake.traces if t.ended]
    assert len(ended) >= 5
