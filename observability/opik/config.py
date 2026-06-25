"""Environment-driven configuration helpers for the Opik Hermes plugin."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("hermes_plugins.opik")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(*names: str) -> bool:
    for name in names:
        value = _env(name).lower()
        if value:
            return value in {"1", "true", "yes", "on"}
    return False


def _debug_enabled() -> bool:
    return _env_bool("HERMES_OPIK_DEBUG")


def _debug(message: str) -> None:
    if _debug_enabled():
        logger.info("Opik tracing: %s", message)


def _project_name() -> str:
    return _env("OPIK_PROJECT_NAME") or "hermes"


def _tags() -> list[str]:
    raw = _env("HERMES_OPIK_TAGS")
    extra = [t.strip() for t in raw.split(",") if t.strip()] if raw else []
    return ["hermes", *extra]
