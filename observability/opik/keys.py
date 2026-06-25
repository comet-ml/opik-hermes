"""In-process trace-scope key construction."""

from __future__ import annotations

import threading
from typing import Any


def _scope_prefix(task_id: str, session_id: str) -> str:
    """The task/session/thread prefix shared by every trace-key shape."""
    if task_id:
        return f"task:{task_id}"
    if session_id:
        return f"session:{session_id}"
    return f"thread:{threading.get_ident()}"


def _trace_key(
    task_id: str,
    session_id: str,
    *,
    turn_id: str = "",
    api_request_id: str = "",
) -> str:
    """Build a stable in-process trace scope key for one agent turn.

    Older Hermes paths only expose ``task_id``/``session_id``. Newer paths
    pass ``turn_id`` and ``api_request_id`` in LLM/tool hooks; when present,
    they must scope trace state so concurrent requests sharing one task/session
    never collide. ``turn_id`` is preferred over ``api_request_id`` so the
    turn-level ``post_llm_call`` hook (which carries ``turn_id`` but no
    ``api_request_id``) resolves to the same key as the request-level hooks.
    """
    if turn_id:
        return f"{_scope_prefix(task_id, session_id)}:turn:{turn_id}"
    if api_request_id:
        return f"{_scope_prefix(task_id, session_id)}:api:{api_request_id}"
    if task_id:
        return task_id
    return _scope_prefix(task_id, session_id)


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)
