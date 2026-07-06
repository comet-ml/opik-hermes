"""Provider-name mapping: Hermes identifiers -> Opik canonical names.

Opik only computes LLM cost when the span's ``provider`` is one it recognizes
(openai, anthropic, groq, bedrock, google_ai, google_vertexai,
anthropic_vertexai). Hermes emits its own names (openai-api, codex_responses,
...), so without mapping, cost silently isn't calculated. These guard the map.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "hermes, opik",
    [
        ("openai-api", "openai"),
        ("openai_responses", "openai"),
        ("codex_responses", "openai"),
        ("azure-openai", "openai"),
        ("anthropic-api", "anthropic"),
        ("anthropic_messages", "anthropic"),
        ("gemini", "google_ai"),
        ("google", "google_ai"),
        ("vertex", "google_vertexai"),
        ("bedrock-runtime", "bedrock"),
        ("groq", "groq"),
    ],
)
def test_known_hermes_providers_map_to_opik_canonical(plugin, hermes, opik):
    assert plugin.providers.to_opik_provider(hermes) == opik


@pytest.mark.parametrize(
    "canonical", ["openai", "anthropic", "groq", "bedrock", "google_ai"]
)
def test_already_canonical_passes_through(plugin, canonical):
    assert plugin.providers.to_opik_provider(canonical) == canonical


def test_unknown_provider_passes_through_unchanged(plugin):
    # Custom / user-defined Hermes providers must not be dropped or mangled —
    # Opik just won't cost them, same as before the mapping existed.
    assert (
        plugin.providers.to_opik_provider("my-custom-provider") == "my-custom-provider"
    )


def test_case_and_whitespace_insensitive(plugin):
    assert plugin.providers.to_opik_provider("  OpenAI-API  ") == "openai"


def test_none_and_empty(plugin):
    assert plugin.providers.to_opik_provider(None) is None
    assert plugin.providers.to_opik_provider("") is None
    assert plugin.providers.to_opik_provider("   ") is None


@pytest.mark.parametrize(
    "hermes_provider, opik_provider",
    [
        ("openai-api", "openai"),
        ("openai_responses", "openai"),
        ("codex_responses", "openai"),
        ("anthropic-api", "anthropic"),
        ("anthropic_messages", "anthropic"),
        ("gemini", "google_ai"),
        ("google", "google_ai"),
        ("vertex", "google_vertexai"),
        ("bedrock-runtime", "bedrock"),
        ("groq", "groq"),
    ],
)
def test_llm_span_provider_is_mapped_through_the_hook_path(
    plugin, hermes_provider, opik_provider
):
    # Not just the pure function — drive a real turn through the hooks for each
    # provider and assert the emitted LLM span carries Opik's canonical name, so
    # Opik can compute cost. (Whether Opik then costs it is Opik's job, not the
    # plugin's — we stop at the span boundary.)
    kw = dict(task_id="t", session_id="s", turn_id="T1")
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="some-model",
        provider=hermes_provider,
        **kw,
    )
    plugin.on_post_llm_call(
        api_call_count=1,
        assistant_tool_call_count=0,
        usage={"input_tokens": 10, "output_tokens": 5},
        model="some-model",
        provider=hermes_provider,
        **kw,
    )
    llm = [s for s in plugin._fake.traces[0].spans if s.type == "llm"]
    assert llm, "expected an llm span"
    assert llm[0].create_kwargs.get("provider") == opik_provider


def test_unknown_provider_reaches_span_unchanged_through_hooks(plugin):
    # A custom/user-defined Hermes provider is passed through so the span still
    # records it (Opik just won't cost it) — no dropping/mangling in the path.
    kw = dict(task_id="t", session_id="s", turn_id="T1")
    plugin.on_pre_llm_request(
        api_call_count=1,
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        provider="my-custom-provider",
        **kw,
    )
    plugin.on_post_llm_call(
        api_call_count=1,
        assistant_tool_call_count=0,
        usage={"input_tokens": 1, "output_tokens": 1},
        model="m",
        provider="my-custom-provider",
        **kw,
    )
    llm = [s for s in plugin._fake.traces[0].spans if s.type == "llm"]
    assert llm and llm[0].create_kwargs.get("provider") == "my-custom-provider"
