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
    assert out[0]["content"]["tool_call"] == "terminal"
    assert out[0]["content"]["arguments"] == {"command": "ls"}
    assert out[0]["content"]["call_id"] == "call_1"


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


def test_responses_reasoning_item_is_marked_not_null(plugin):
    out = _ser(plugin, [{"type": "reasoning", "encrypted_content": "xxxxx"}])
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == "[reasoning]"
    # The bug was these landing as {role: null, content: null}.
    assert out[0]["content"] is not None


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
    assert len(out) == 5
    # No item is a bare null/null anymore.
    assert all(not (m["role"] is None and m["content"] is None) for m in out)
    assert [m["role"] for m in out] == [
        "user",
        "assistant",
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
