"""Opik client factory + process-wide cache.

The client is constructed lazily on first use and cached. Activation of the
plugin is controlled by the Hermes plugin system; this module only handles the
runtime-availability gate (is the ``opik`` SDK installed and constructible).
"""

from __future__ import annotations

from typing import Any, Optional

from .config import logger, project_name

try:
    import opik as _opik_sdk
except Exception:  # pragma: no cover - fail-open when optional dep is missing
    _opik_sdk = None

_CLIENT: Any = None

# Sentinel: "get_client() has tried and failed". Lets us short-circuit every
# subsequent hook call without re-attempting SDK init. Runtime callers cannot
# reset the cache; if an operator fixes a misconfigured credential they must
# restart the process.
_INIT_FAILED = object()


def get_client() -> Optional[Any]:
    """Return a cached Opik client, or ``None`` if unavailable.

    The result is cached: on the first call we try to construct a client, and
    every subsequent call returns that client (or fast-returns ``None`` if
    init failed).

    A local open-source Opik needs no API key, so — unlike the Langfuse plugin
    — we do NOT gate on credentials here. The Opik SDK reads OPIK_API_KEY /
    OPIK_URL_OVERRIDE / OPIK_WORKSPACE from the environment itself.
    """
    global _CLIENT
    if _CLIENT is _INIT_FAILED:
        return None
    if _CLIENT is not None:
        return _CLIENT

    if _opik_sdk is None:
        _CLIENT = _INIT_FAILED
        return None

    try:
        _CLIENT = _opik_sdk.Opik(project_name=project_name())
    except Exception as exc:  # pragma: no cover - fail-open
        logger.warning("Could not initialize Opik client: %s", exc)
        _CLIENT = _INIT_FAILED
        return None

    return _CLIENT


def set_client(client: Any) -> None:
    """Install a client instance directly (used by tests)."""
    global _CLIENT
    _CLIENT = client
