"""Environment-driven configuration helpers for the Opik Hermes plugin."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("hermes_plugins.opik")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_bool(*names: str) -> bool:
    for name in names:
        value = env(name).lower()
        if value:
            return value in {"1", "true", "yes", "on"}
    return False


def debug_enabled() -> bool:
    return env_bool("HERMES_OPIK_DEBUG")


def debug(message: str) -> None:
    if debug_enabled():
        logger.info("Opik tracing: %s", message)


def project_name() -> str:
    return env("OPIK_PROJECT_NAME") or "hermes"


def tags() -> list[str]:
    raw = env("HERMES_OPIK_TAGS")
    extra = [t.strip() for t in raw.split(",") if t.strip()] if raw else []
    return ["hermes", *extra]
