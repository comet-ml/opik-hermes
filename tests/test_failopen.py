"""Fail-open: with no SDK / no client, every hook is an inert no-op."""

from __future__ import annotations

import pytest

HOOK_CALLS = [
    (
        "on_pre_llm_call",
        dict(task_id="t", session_id="s", messages=[{"role": "user", "content": "x"}]),
    ),
    (
        "on_pre_llm_request",
        dict(
            task_id="t",
            session_id="s",
            api_call_count=1,
            messages=[{"role": "user", "content": "x"}],
        ),
    ),
    (
        "on_post_llm_call",
        dict(task_id="t", session_id="s", api_call_count=1, assistant_response="y"),
    ),
    (
        "on_pre_tool_call",
        dict(
            tool_name="terminal",
            args={"cmd": "ls"},
            task_id="t",
            session_id="s",
            tool_call_id="tc1",
        ),
    ),
    (
        "on_post_tool_call",
        dict(
            tool_name="terminal",
            args={"cmd": "ls"},
            result="z",
            task_id="t",
            session_id="s",
            tool_call_id="tc1",
        ),
    ),
    (
        "on_api_request_error",
        dict(
            task_id="t",
            session_id="s",
            api_call_count=1,
            error={"type": "Timeout", "message": "boom"},
        ),
    ),
    (
        "on_subagent_stop",
        dict(parent_session_id="s", child_role="researcher", child_status="completed"),
    ),
    ("on_session_end", dict(session_id="s")),
    ("on_session_event", dict(session_id="s")),
]


@pytest.mark.parametrize("hook_name,kwargs", HOOK_CALLS)
def test_hooks_noop_when_sdk_missing(plugin_no_sdk, hook_name, kwargs):
    getattr(plugin_no_sdk, hook_name)(**kwargs)  # must not raise
    # _get_opik should short-circuit to None and cache the failure.
    assert plugin_no_sdk._get_opik() is None


def test_register_wires_expected_hooks(plugin_no_sdk):
    registered = []

    class Ctx:
        def register_hook(self, name, cb):
            registered.append(name)

    plugin_no_sdk.register(Ctx())
    assert set(registered) == {
        # LLM (per-API-call + per-turn)
        "pre_api_request",
        "post_api_request",
        "pre_llm_call",
        "post_llm_call",
        # tools
        "pre_tool_call",
        "post_tool_call",
        # failure + agentic structure
        "api_request_error",
        "subagent_stop",
        # session lifecycle
        "on_session_start",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
    }
    # We must NOT register mutation hooks (transform_*) — passive observer only.
    assert not any(h.startswith("transform_") for h in registered)


def test_post_tool_call_without_prior_state_is_noop(plugin):
    # A tool result arriving with no tracked trace must not raise.
    plugin.on_post_tool_call(
        tool_name="terminal",
        args={},
        result="r",
        task_id="t",
        session_id="s",
        turn_id="T1",
        tool_call_id="tc1",
    )


def test_unknown_extra_kwargs_are_ignored(plugin):
    # Hermes may add kwargs across versions; hooks accept **_ and must not break.
    plugin.on_pre_llm_request(
        task_id="t",
        session_id="s",
        api_call_count=1,
        messages=[{"role": "user", "content": "x"}],
        some_future_kwarg=123,
        another="v",
    )
