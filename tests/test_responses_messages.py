"""Serialize Responses-API message items (codex_responses mode), not just chat."""

from __future__ import annotations


def _ser(plugin, messages):
    return plugin.sanitize.serialize_messages(messages)


def test_chat_completions_message_unchanged(plugin):
    # Plain {role, content} still serializes as before.
    out = _ser(plugin, [{"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": "hi"}]


def test_responses_message_with_content_parts_flattened(plugin):
    msg = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "Hello"},
            {"type": "output_text", "text": "world"},
        ],
    }
    out = _ser(plugin, [msg])
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == "Hello world"


def test_responses_function_call_item(plugin):
    msg = {
        "type": "function_call",
        "name": "terminal",
        "arguments": '{"command": "ls"}',
        "call_id": "call_1",
    }
    out = _ser(plugin, [msg])
    assert out[0]["role"] == "assistant"
    # Readable text summary so the message bubble isn't empty ...
    assert out[0]["content"] == 'terminal({"command": "ls"})'
    # ... plus the structured tool call for pretty renderers.
    tc = out[0]["tool_calls"][0]
    assert tc["name"] == "terminal"
    assert tc["arguments"] == {"command": "ls"}
    assert tc["id"] == "call_1"
    # function.arguments must be a JSON *string* (OpenAI wire format) or the
    # Opik pretty renderer shows an empty tool-call block.
    assert tc["function"]["arguments"] == '{"command": "ls"}'


def test_responses_function_call_output_item(plugin):
    msg = {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"ok": true}',
    }
    out = _ser(plugin, [msg])
    assert out[0]["role"] == "tool"
    assert out[0]["call_id"] == "call_1"
    assert out[0]["content"] == {"ok": True}


def test_responses_reasoning_item_dropped(plugin):
    # Reasoning content is encrypted/opaque — no readable text, so it would
    # render as an empty bubble. Drop it entirely on the input side.
    out = _ser(plugin, [{"type": "reasoning", "encrypted_content": "xxxxx"}])
    assert out == []


def test_no_null_role_content_for_responses_payload(plugin):
    # The exact shape from the bug report: a mixed Responses-API request body.
    payload = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "go"}],
        },
        {"type": "reasoning", "encrypted_content": "..."},
        {
            "type": "function_call",
            "name": "terminal",
            "arguments": "{}",
            "call_id": "c1",
        },
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        },
    ]
    out = _ser(plugin, payload)
    # Reasoning item is dropped (opaque content); the rest serialize readably.
    assert len(out) == 4
    assert all(not (m["role"] is None and m["content"] is None) for m in out)
    assert [m["role"] for m in out] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_responses_image_part_noted(plugin):
    msg = {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "describe"},
            {"type": "input_image", "image_url": "https://x/y.png"},
        ],
    }
    out = _ser(plugin, [msg])
    assert out[0]["content"]["text"] == "describe"
    assert out[0]["content"]["images"] == 1


def test_unrecognized_item_kept_not_nulled(plugin):
    out = _ser(plugin, [{"type": "mystery", "blah": 1}])
    assert out[0]["role"] is None
    assert "unrecognized_item" in out[0]["content"]


def test_empty_assistant_message_dropped(plugin):
    # A tool-call-only assistant turn arrives with content "" and its tool calls
    # emitted as separate items — the bare {assistant, ""} record is pure noise.
    out = _ser(plugin, [{"role": "assistant", "content": ""}])
    assert out == []


def test_assistant_with_tool_calls_kept_despite_empty_content(plugin):
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
    }
    out = _ser(plugin, [msg])
    assert len(out) == 1
    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"]


def test_empty_user_message_kept(plugin):
    # Only assistant no-signal turns are dropped; other roles pass through.
    out = _ser(plugin, [{"role": "user", "content": ""}])
    assert out == [{"role": "user", "content": ""}]
