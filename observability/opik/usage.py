"""Token-usage and cost mapping, bridging Hermes' pricing to Opik's schema."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import debug


def opik_usage_from_canonical(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
    reasoning: int = 0,
) -> dict[str, int]:
    """Build an Opik-friendly usage dict using OpenAI-style key names.

    Opik surfaces input/output/total tokens when the usage dict carries the
    OpenAI keys ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.
    Cache and reasoning tokens are passed through under their canonical names
    for completeness.
    """
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write:
        usage["cache_creation_input_tokens"] = cache_write
    if reasoning:
        usage["reasoning_tokens"] = reasoning
    return usage


def usage_and_cost(
    response: Any, *, provider: str, api_mode: str, model: str, base_url: str
) -> tuple[dict[str, int], Optional[float]]:
    """Return (opik_usage_dict, total_cost_usd) from a provider response.

    Reuses Hermes' own ``agent.usage_pricing`` for canonical token counts and
    USD cost, then maps to the OpenAI-style keys Opik expects so input/output/
    total tokens render in the UI. ``total_cost`` is passed to Opik directly
    (it takes priority over Opik's own usage-derived estimate).
    """
    usage_details: Dict[str, int] = {}
    total_cost: Optional[float] = None
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return usage_details, total_cost

    try:
        from agent.usage_pricing import estimate_usage_cost, normalize_usage

        canonical = normalize_usage(raw_usage, provider=provider, api_mode=api_mode)
        usage_details = opik_usage_from_canonical(
            input_tokens=canonical.input_tokens,
            output_tokens=canonical.output_tokens,
            cache_read=canonical.cache_read_tokens,
            cache_write=canonical.cache_write_tokens,
            reasoning=canonical.reasoning_tokens,
        )
        cost = estimate_usage_cost(
            model, canonical, provider=provider, base_url=base_url, api_key=""
        )
        if cost.amount_usd is not None:
            total_cost = float(cost.amount_usd)
    except Exception as exc:  # pragma: no cover - fail-open
        debug(f"usage normalization failed: {exc}")

    return usage_details, total_cost


def cost_from_usage_dict(
    usage: dict, *, provider: str, base_url: str, model: str
) -> Optional[float]:
    """Best-effort USD cost from a pre-built usage summary dict (post_api_request).

    Returns ``None`` when Hermes' pricing module is unavailable or the model is
    unknown — callers treat that as "no cost recorded".
    """
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        cu = CanonicalUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0)
            or usage.get("completion_tokens", 0),
            cache_read_tokens=usage.get("cache_read_tokens", 0),
            cache_write_tokens=usage.get("cache_write_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
        )
        cost = estimate_usage_cost(
            model, cu, provider=provider, base_url=base_url, api_key=""
        )
        if cost.amount_usd is not None:
            return float(cost.amount_usd)
    except Exception:
        pass
    return None
