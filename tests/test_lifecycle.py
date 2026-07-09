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
    # LLM spans are now created fully-formed in one call (start+end together),
    # not created-then-updated — so model/usage are in the create kwargs.
    ck = llm_spans[0].create_kwargs
    assert ck.get("model") == "gpt-5"
    assert ck.get("usage") is not None
    assert ck.get("start_time") is not None and ck.get("end_time") is not None
    assert llm_spans[0].updates == []  # no post-creation mutation


def test_llm_span_is_single_message_not_create_then_end(plugin):
    # Regression: a fast API call (pre+post within one SDK batch window) used to
    # lose the generation span's create message, leaving an "NA" llm span
    # (name=None/type=None/epoch start). The span must now be born complete in
    # one trace.span() call.
    _run_turn_with_tool(plugin)
    llm = [s for s in plugin._fake.traces[0].spans if s.type == "llm"]
    assert llm
    for span in llm:
        assert span.name and span.name.startswith("LLM call")
        assert span.create_kwargs.get("start_time") is not None
        assert span.create_kwargs.get("end_time") is not None
        assert span.updates == []
        assert not span.ended


def test_tool_span_captures_output(plugin):
    _run_turn_with_tool(plugin)
    tool_spans = [s for s in plugin._fake.traces[0].spans if s.type == "tool"]
    assert tool_spans
    # The tool span is created fully-formed in one call (output + start/end
    # together), not created-then-ended — so the create/end batching race
    # can't strip its fields. See the NA-span fix.
    ck = tool_spans[0].create_kwargs
    assert "output" in ck and ck["output"]
    assert ck.get("start_time") is not None
    assert ck.get("end_time") is not None


def test_tool_span_is_single_message_not_create_then_end(plugin):
    # Regression: fast tools (pre+post within one SDK batch window) used to lose
    # the create message, leaving an "NA" span (name=None/type=None/epoch start).
    # The span must now be born complete in one trace.span() call — no separate
    # update()/end() on it afterwards.
    _run_turn_with_tool(plugin)
    tool_spans = [s for s in plugin._fake.traces[0].spans if s.type == "tool"]
    span = tool_spans[0]
    assert span.name == "Tool: terminal"
    assert span.type == "tool"
    assert span.create_kwargs.get("start_time") is not None
    assert span.create_kwargs.get("end_time") is not None
    # No post-creation mutation: the span carried everything at birth.
    assert span.updates == []
    assert not span.ended


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
# finalized (end_time stayed null in Opik). These guard that the per-turn
# signal finalizes and flushes the trace — now via an upsert re-send (same id,
# output+end_time), not the forbidden trace.update()/trace.end().


def test_post_llm_call_finalizes_and_flushes_trace(plugin):
    _run_turn_with_tool(plugin, finalize=True)
    trace = plugin._fake.traces[0]
    assert trace.finalized, "root trace must be finalized on the per-turn post_llm_call"
    assert plugin._fake.flushed >= 1, "finish_trace must flush"
    assert "output" in trace.finalize_kwargs, "trace output must be set on finalize"
    assert trace.finalize_kwargs.get("end_time") is not None, (
        "finalize re-send must carry end_time"
    )


def test_finalize_is_upsert_not_update_or_end(plugin):
    # Upsert-only: finalize must be a same-id client.trace(...) re-send, never
    # trace.update()/trace.end() (FakeTrace raises on those). The re-send targets
    # the create's id so the SDK coalesces them into one row. This is the fix for
    # OPIK-7279 — the "may cause data loss" warning is the symptom of an
    # update() shortly after create.
    _run_turn_with_tool(plugin, finalize=True)
    fake = plugin._fake
    assert len(fake.traces) == 1, "upsert must not mint a second trace"
    assert not any(e[0] in ("trace.update", "trace.end") for e in fake.events)
    # Two client.trace calls sharing one id: the create and the finalize re-send.
    trace_calls = [e for e in fake.events if e[0] == "client.trace"]
    assert len(trace_calls) == 2, "expected create + finalize re-send"


def test_trace_not_finalized_without_turn_end_signal(plugin):
    # Without the per-turn post_llm_call, the trace stays open (matches reality:
    # an interrupted/incomplete turn is not a completed trace).
    _run_turn_with_tool(plugin, finalize=False)
    trace = plugin._fake.traces[0]
    assert not trace.finalized


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
    assert not plugin._fake.traces[0].finalized


def test_session_id_becomes_thread_id(plugin):
    _run_turn_with_tool(plugin)
    assert plugin._fake.traces[0].create_kwargs.get("thread_id") == "s"


# --- Regression: NA trace name (the trace-level twin of the NA-span bug) -----
# The root trace's name/thread_id/input are set only at creation. For a fast
# turn the create message coalesced with the finalize re-send in one SDK batch
# window, and the trace landed with name=None/thread_id=None/input=null — an
# "NA" trace in the UI. Confirmed against real Opik: a trace created then
# immediately re-sent+finalized loses its name, while one whose create is
# flushed first keeps it. We flush right after trace creation to close the
# create's batch; these guard that invariant so the bug can't silently return.


def test_trace_created_with_name_and_thread(plugin):
    _run_turn_with_tool(plugin)
    ck = plugin._fake.traces[0].create_kwargs
    assert ck.get("name"), "root trace must be created WITH a name"
    assert ck.get("thread_id") == "s"
    assert ck.get("input") is not None, "root trace must carry input at creation"


def test_trace_create_is_flushed_before_finalize(plugin):
    # The fix: the create message must be sent as its own batch, so it cannot
    # coalesce with the finalize re-send and lose name/thread/input.
    _run_turn_with_tool(plugin, finalize=True)
    events = plugin._fake.events
    create_idx = next(i for i, e in enumerate(events) if e[0] == "client.trace")
    # There must be a flush AFTER the create and BEFORE the finalize re-send.
    finalize_idx = next(
        (i for i, e in enumerate(events) if e[0] == "trace.upsert"),
        len(events),
    )
    flush_between = any(e[0] == "flush" for e in events[create_idx + 1 : finalize_idx])
    assert flush_between, (
        "trace create must be flushed before the finalize re-send so its "
        "name/thread/input survive the batching race"
    )
