"""Trace naming: derive a scannable name from the user message."""

from __future__ import annotations


def _open_turn(plugin, messages, **kw):
    plugin.on_pre_llm_request(
        task_id="t",
        session_id="s",
        turn_id="T1",
        api_call_count=1,
        messages=messages,
        model="gpt-5",
        **kw,
    )
    return plugin._fake.traces[0]


def test_trace_named_from_user_message(plugin):
    trace = _open_turn(
        plugin, [{"role": "user", "content": "Update the World Cup stats"}]
    )
    assert trace.create_kwargs["name"] == "Update the World Cup stats"


def test_long_user_message_is_truncated(plugin):
    long = "Can you update the stats, as now we are moving to the Knockout phase and we know who plays whom"
    trace = _open_turn(plugin, [{"role": "user", "content": long}])
    name = trace.create_kwargs["name"]
    assert len(name) <= 61  # 60 chars + ellipsis
    assert name.endswith("…")
    assert name.startswith("Can you update the stats")


def test_whitespace_is_collapsed(plugin):
    trace = _open_turn(plugin, [{"role": "user", "content": "  hello\n\n  world  "}])
    assert trace.create_kwargs["name"] == "hello world"


def test_latest_user_message_wins(plugin):
    msgs = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second question"},
    ]
    trace = _open_turn(plugin, msgs)
    assert trace.create_kwargs["name"] == "second question"


def test_list_content_parts_flattened(plugin):
    content = [
        {"type": "text", "text": "describe"},
        {"type": "text", "text": "this image"},
    ]
    trace = _open_turn(plugin, [{"role": "user", "content": content}])
    assert trace.create_kwargs["name"] == "describe this image"


def test_falls_back_to_hermes_turn_when_no_user_text(plugin):
    # No user message at all -> stable default name.
    trace = _open_turn(plugin, [{"role": "assistant", "content": "hi"}])
    assert trace.create_kwargs["name"] == "Hermes turn"


def test_falls_back_when_user_content_empty(plugin):
    trace = _open_turn(plugin, [{"role": "user", "content": ""}])
    assert trace.create_kwargs["name"] == "Hermes turn"


def test_trace_name_helper_returns_none_for_non_list(plugin):
    assert plugin.sanitize.trace_name_from_messages("not a list") is None
    assert plugin.sanitize.trace_name_from_messages(None) is None
