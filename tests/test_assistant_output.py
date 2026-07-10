"""LLM-span output shape (OPIK-7280).

Opik's pretty renderer only recognizes LLM output when it looks like an OpenAI
chat completion — it reads ``choices[-1].message.content``. These tests pin the
shape the plugin emits so the trace-view Output offers Pretty mode for both
content-only and tool-call turns.
"""

from __future__ import annotations


class FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = type("Fn", (), {"name": name, "arguments": arguments})()


class FakeMessage:
    def __init__(self, content=None, reasoning=None, tool_calls=None):
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls


def _message(output):
    return output["choices"][-1]["message"]


def test_content_only_response_is_prettifiable(plugin):
    out = plugin.sanitize.serialize_assistant_message(
        FakeMessage(content="Hello, world.")
    )
    msg = _message(out)
    # Pretty mode reads choices[-1].message.content as a non-empty string.
    assert msg["content"] == "Hello, world."
    assert msg["role"] == "assistant"
    assert msg["tool_calls"] == []


def test_tool_call_turn_surfaces_readable_content(plugin):
    out = plugin.sanitize.serialize_assistant_message(
        FakeMessage(
            content=None,
            tool_calls=[FakeToolCall("call_1", "get_weather", {"city": "SF"})],
        )
    )
    msg = _message(out)
    # No assistant text, so content is a readable rendering of the tool calls
    # (empty content would fall back to raw JSON in the UI).
    assert isinstance(msg["content"], str) and msg["content"]
    assert "get_weather" in msg["content"]
    assert msg["tool_calls"][0]["name"] == "get_weather"


def test_content_takes_precedence_over_tool_calls(plugin):
    out = plugin.sanitize.serialize_assistant_message(
        FakeMessage(
            content="Sure, checking.",
            tool_calls=[FakeToolCall("call_1", "get_weather", {"city": "SF"})],
        )
    )
    msg = _message(out)
    assert msg["content"] == "Sure, checking."
    assert msg["tool_calls"][0]["name"] == "get_weather"


def test_reasoning_preserved_as_structured_field(plugin):
    out = plugin.sanitize.serialize_assistant_message(
        FakeMessage(content="Answer.", reasoning="chain of thought")
    )
    assert _message(out)["reasoning"] == "chain of thought"


def test_assistant_output_helper_empty_content_no_tools(plugin):
    out = plugin.sanitize.assistant_output(content=None)
    msg = _message(out)
    assert msg["content"] is None
    assert msg["tool_calls"] == []
