"""api_request_error, subagent_stop, and session-lifecycle hook behavior."""

from __future__ import annotations


def _open_generation(plugin, **kw):
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5",
        provider="openai-api",
        **kw,
    )


# --- api_request_error ------------------------------------------------------


def test_api_error_sets_error_info_and_ends_generation(plugin):
    kw = dict(task_id="t", session_id="s", turn_id="T1")
    _open_generation(plugin, **kw)
    plugin.on_api_request_error(
        api_call_count=1,
        model="gpt-5",
        provider="openai-api",
        status_code=500,
        retryable=True,
        reason="server_error",
        error={"type": "ServerError", "message": "upstream 500"},
        **kw,
    )
    # The errored call becomes a fully-formed LLM span (one message) carrying
    # error_info — not a created-then-updated span.
    llm = [s for s in plugin._fake.traces[0].spans if s.type == "llm"][0]
    ck = llm.create_kwargs
    assert ck["error_info"]["message"] == "upstream 500"
    assert ck["error_info"]["exception_type"] == "ServerError"
    assert ck.get("start_time") is not None and ck.get("end_time") is not None
    assert ck["output"] == {"error": "upstream 500"}


def test_api_error_without_open_generation_is_noop(plugin):
    # error arriving with no matching generation must not raise or create spans
    plugin.on_api_request_error(
        task_id="t",
        session_id="s",
        turn_id="T1",
        api_call_count=9,
        error={"type": "X", "message": "y"},
    )
    assert plugin._fake.traces == []


# --- subagent_stop ----------------------------------------------------------


def test_subagent_stop_adds_span_under_parent_trace(plugin):
    kw = dict(task_id="t", session_id="s", turn_id="T1")
    _open_generation(plugin, **kw)
    plugin.on_subagent_stop(
        parent_session_id="s",
        child_role="researcher",
        child_summary="found 3 sources",
        child_status="completed",
        duration_ms=1500,
        **kw,
    )
    spans = plugin._fake.traces[0].spans
    sub = [s for s in spans if s.create_kwargs.get("metadata", {}).get("subagent")]
    assert sub, "expected a subagent span"
    ck = sub[0].create_kwargs
    assert ck["name"] == "Subagent: researcher"
    assert ck["output"]["summary"] == "found 3 sources"
    assert ck.get("start_time") is not None and ck.get("end_time") is not None


def test_subagent_stop_resolves_parent_by_session_when_no_turn_key(plugin):
    # Subagent stop often carries only parent_session_id; resolve to that
    # session's active trace.
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5",
        task_id="",
        session_id="sess-9",
        turn_id="T9",
    )
    plugin.on_subagent_stop(
        parent_session_id="sess-9", child_role="coder", child_status="completed"
    )
    spans = plugin._fake.traces[0].spans
    assert any(s.create_kwargs.get("metadata", {}).get("subagent") for s in spans)


def test_subagent_stop_no_parent_trace_is_noop(plugin):
    plugin.on_subagent_stop(parent_session_id="unknown", child_role="x")
    assert plugin._fake.traces == []


def test_subagent_stop_skips_when_session_is_ambiguous(plugin):
    # Two concurrent turns in the same session, and the subagent carries only
    # parent_session_id (no turn key). There's no reliable way to pick the
    # parent, so we must skip rather than guess by recency and attach the span
    # under the wrong turn.
    for turn in ("T1", "T2"):
        plugin.on_pre_llm_request(
            api_call_count=1,
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-5",
            task_id=f"task-{turn}",
            session_id="sess-multi",
            turn_id=turn,
        )
    plugin.on_subagent_stop(
        parent_session_id="sess-multi", child_role="coder", child_status="completed"
    )
    # No subagent span attached to either turn's trace.
    for trace in plugin._fake.traces:
        assert not any(
            s.create_kwargs.get("metadata", {}).get("subagent") for s in trace.spans
        )


# --- session lifecycle ------------------------------------------------------


def test_session_end_finalizes_open_traces_and_flushes(plugin):
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5",
        session_id="sX",
        turn_id="T1",
    )
    assert not plugin._fake.traces[0].ended
    plugin.on_session_end(session_id="sX")
    assert plugin._fake.traces[0].ended
    assert plugin._fake.flushed >= 1


def test_session_end_finalizes_task_keyed_traces(plugin):
    # Regression: with a task_id present, trace_key() stores the turn under
    # task:{task_id}:... — NOT session:{id}. on_session_end must still finalize
    # it by matching TraceState.session_id, not a session: key-prefix scan.
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5",
        task_id="task-42",
        session_id="sess-y",
        turn_id="T1",
    )
    key = plugin.keys.trace_key("task-42", "sess-y", turn_id="T1")
    assert key.startswith("task:"), "precondition: turn is task-keyed"
    assert not plugin._fake.traces[0].ended
    plugin.on_session_end(session_id="sess-y")
    assert plugin._fake.traces[0].ended
    assert plugin._fake.flushed >= 1


def test_subagent_stop_resolves_task_keyed_parent_by_session(plugin):
    # Regression twin of the above for the subagent_stop fallback scan: a
    # task-keyed parent trace must be found by session_id when only
    # parent_session_id is supplied.
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5",
        task_id="task-77",
        session_id="sess-z",
        turn_id="T1",
    )
    plugin.on_subagent_stop(
        parent_session_id="sess-z", child_role="coder", child_status="completed"
    )
    spans = plugin._fake.traces[0].spans
    assert any(s.create_kwargs.get("metadata", {}).get("subagent") for s in spans)


def test_session_end_without_session_id_is_noop(plugin):
    plugin.on_session_end(session_id="")  # must not raise


def test_session_event_is_noop_observer(plugin):
    plugin.on_session_event(session_id="s")  # breadcrumb only; no traces
    assert plugin._fake.traces == []
