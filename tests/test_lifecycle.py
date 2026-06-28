"""End-to-end hook lifecycle, including the trace-finalization regression."""

from __future__ import annotations


def _run_turn_with_tool(plugin, *, finalize: bool = True):
    """Drive one turn: pre_request -> post_request(tool) -> tool -> [post_llm_call]."""
    kw = dict(task_id="t", session_id="s", turn_id="T1")
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5",
        provider="openai-api",
        **kw,
    )
    plugin.on_post_llm_call(
        api_call_count=1,
        assistant_tool_call_count=1,
        usage={"input_tokens": 10, "output_tokens": 5},
        model="gpt-5",
        **kw,
    )
    plugin.on_pre_tool_call(
        tool_name="terminal", args={"cmd": "ls"}, tool_call_id="tc1", **kw
    )
    plugin.on_post_tool_call(
        tool_name="terminal",
        args={"cmd": "ls"},
        result="a\nb",
        tool_call_id="tc1",
        **kw,
    )
    if finalize:
        # Per-turn signal: assistant_response present, NO api_call_count generation.
        plugin.on_post_llm_call(
            assistant_response="here are the files", model="gpt-5", **kw
        )


def test_full_turn_creates_trace_llm_and_tool_spans(plugin):
    _run_turn_with_tool(plugin)
    fake = plugin._fake
    assert len(fake.traces) == 1
    trace = fake.traces[0]
    span_types = [s.type for s in trace.spans]
    assert "llm" in span_types
    assert "tool" in span_types


def test_llm_span_carries_model_and_usage(plugin):
    _run_turn_with_tool(plugin)
    llm_spans = [s for s in plugin._fake.traces[0].spans if s.type == "llm"]
    assert llm_spans, "expected at least one llm span"
    last_update = llm_spans[0].updates[-1]
    assert last_update.get("model") == "gpt-5"
    assert last_update.get("usage") is not None


def test_tool_span_captures_output(plugin):
    _run_turn_with_tool(plugin)
    tool_spans = [s for s in plugin._fake.traces[0].spans if s.type == "tool"]
    assert tool_spans
    assert tool_spans[0].ended
    assert any("output" in u for u in tool_spans[0].updates)


def test_every_span_has_name_and_type(plugin):
    # name/type are creation-only in Opik (span.update can't set them), so a
    # span must never be created without them — else it renders as "NA" in the
    # UI. Guards our creation paths (LLM + tool) against that regression.
    _run_turn_with_tool(plugin)
    for span in plugin._fake.traces[0].spans:
        assert span.name, f"span created without a name: {span!r}"
        assert span.type in {"llm", "tool"}, f"unexpected span type: {span.type!r}"


def test_tool_span_named_after_tool(plugin):
    _run_turn_with_tool(plugin)
    tool_spans = [s for s in plugin._fake.traces[0].spans if s.type == "tool"]
    assert tool_spans[0].name == "Tool: terminal"


# --- Regression: the bug fixed in #2 ----------------------------------------
# Before the fix, the root trace was only ended under a content heuristic the
# per-turn post_llm_call never satisfied, so traces were created but never
# .end()-ed (end_time stayed null in Opik). These guard that the per-turn
# signal finalizes and flushes the trace.


def test_post_llm_call_finalizes_and_flushes_trace(plugin):
    _run_turn_with_tool(plugin, finalize=True)
    trace = plugin._fake.traces[0]
    assert trace.ended, "root trace must be ended on the per-turn post_llm_call"
    assert plugin._fake.flushed >= 1, "finish_trace must flush"
    assert any("output" in u for u in trace.updates), (
        "trace output must be set on finalize"
    )


def test_trace_not_finalized_without_turn_end_signal(plugin):
    # Without the per-turn post_llm_call, the trace stays open (matches reality:
    # an interrupted/incomplete turn is not a completed trace).
    _run_turn_with_tool(plugin, finalize=False)
    trace = plugin._fake.traces[0]
    assert not trace.ended


def test_per_api_call_does_not_prematurely_finalize(plugin):
    # A content-bearing per-API-call response mid-turn must NOT close the trace
    # (the old heuristic did this and dropped later tool/API calls).
    kw = dict(task_id="t", session_id="s", turn_id="T1")
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5",
        **kw,
    )
    plugin.on_post_llm_call(
        api_call_count=1,
        assistant_response="partial",
        assistant_tool_call_count=0,
        model="gpt-5",
        **kw,
    )
    assert not plugin._fake.traces[0].ended


def test_session_id_becomes_thread_id(plugin):
    _run_turn_with_tool(plugin)
    assert plugin._fake.traces[0].create_kwargs.get("thread_id") == "s"
