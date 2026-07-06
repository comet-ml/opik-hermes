"""Map Hermes provider identifiers to Opik's canonical provider names.

Opik only computes/attributes LLM cost when a span's ``provider`` matches one of
its recognized providers (``opik.types.LLMProvider``: openai, anthropic, groq,
bedrock, google_ai, google_vertexai, anthropic_vertexai). Hermes emits its own
identifiers (e.g. ``openai-api``, ``codex_responses``, ``anthropic_messages``),
which Opik does not recognize — so cost silently isn't calculated.

``to_opik_provider`` translates the Hermes name to the Opik canonical one for the
value written to the span's ``provider`` field. Anything already canonical, or
unrecognized (custom / user-defined Hermes providers), passes through unchanged
so cost calc degrades gracefully rather than erroring.
"""

from __future__ import annotations

from typing import Optional

# Hermes identifier -> Opik canonical (opik.types.LLMProvider values: openai,
# anthropic, groq, bedrock, google_ai, google_vertexai, anthropic_vertexai).
# Hermes provider strings are lowercased and
# stripped before lookup. Unlisted values fall through unchanged.
_HERMES_TO_OPIK = {
    # OpenAI family (chat completions + Responses/codex modes)
    "openai-api": "openai",
    "openai": "openai",
    "openai_responses": "openai",
    "codex_responses": "openai",
    "azure-openai": "openai",
    "azure_openai": "openai",
    # Anthropic
    "anthropic-api": "anthropic",
    "anthropic": "anthropic",
    "anthropic_messages": "anthropic",
    # Google
    "google": "google_ai",
    "google-ai": "google_ai",
    "google_ai": "google_ai",
    "gemini": "google_ai",
    "google-vertexai": "google_vertexai",
    "google_vertexai": "google_vertexai",
    "vertexai": "google_vertexai",
    "vertex": "google_vertexai",
    # Groq
    "groq": "groq",
    # AWS Bedrock
    "bedrock": "bedrock",
    "bedrock-runtime": "bedrock",
    "bedrock_converse": "bedrock",
    "bedrock_anthropic": "bedrock",
}


def to_opik_provider(provider: Optional[str]) -> Optional[str]:
    """Return the Opik canonical provider name for a Hermes provider string.

    - Known Hermes identifier -> its Opik canonical name (enables Opik cost calc).
    - Already-canonical or unrecognized -> returned unchanged (graceful: Opik
      simply won't compute cost for an unknown provider, same as before).
    - ``None`` / empty -> ``None``.
    """
    if not provider:
        return None
    key = provider.strip().lower()
    if not key:
        return None
    if key in _HERMES_TO_OPIK:
        return _HERMES_TO_OPIK[key]
    return provider
