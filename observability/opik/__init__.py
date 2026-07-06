"""opik — Hermes plugin for Opik observability.

Traces Hermes conversations, LLM calls, and tool usage to Opik
(https://github.com/comet-ml/opik), Comet's open-source LLM observability
platform.

Activation is handled by the Hermes plugin system — standalone plugins only
load when listed in ``plugins.enabled`` (via ``hermes plugins enable
observability/opik`` or ``hermes tools → Opik Observability``). At runtime the
plugin also requires the ``opik`` SDK and, for Comet-hosted Opik, credentials;
if the SDK is missing the hooks are inert.

Configuration (set via ``hermes tools`` or ~/.hermes/.env):
  OPIK_API_KEY        - Opik/Comet API key. NOT required for a local
                        open-source Opik (http://localhost:5173/api) — only
                        for Comet-hosted or self-hosted-with-auth deployments.
  OPIK_URL_OVERRIDE   - Opik API base URL. Defaults to the Opik SDK default
                        (Comet cloud). For local Opik set e.g.
                        http://localhost:5173/api.
  OPIK_WORKSPACE      - Opik workspace name (Comet-hosted only).
  OPIK_PROJECT_NAME   - Project to log traces under (default: "hermes").

Optional env vars:
  HERMES_OPIK_TAGS         - comma-separated tags applied to every trace
  HERMES_OPIK_MAX_CHARS    - max chars per field (default: 12000)
  HERMES_OPIK_DEBUG        - set to "true" for verbose plugin logging

Module layout:
  config.py     - env/config helpers (env, debug, project_name, tags, logger)
  client.py     - Opik client factory + process-wide cache (get_client)
  keys.py       - trace-scope key construction (trace_key, request_key)
  state.py      - trace-state store: TraceState + pending records + eviction
  lifecycle.py  - root-trace create / flush-on-create / finalize
  usage.py      - token-usage + cost mapping via agent.usage_pricing
  sanitize/     - pure payload sanitization (values, messages, tools, read_file)
  hooks.py      - Hermes hook handlers + register(ctx)
  __init__.py   - public surface + register entry point
"""

from __future__ import annotations

from .hooks import (
    on_api_request_error,
    on_post_llm_call,
    on_post_tool_call,
    on_pre_llm_call,
    on_pre_llm_request,
    on_pre_tool_call,
    on_session_end,
    on_session_event,
    on_subagent_stop,
    register,
)

__all__ = [
    "register",
    "on_pre_llm_call",
    "on_pre_llm_request",
    "on_post_llm_call",
    "on_pre_tool_call",
    "on_post_tool_call",
    "on_api_request_error",
    "on_subagent_stop",
    "on_session_end",
    "on_session_event",
]
