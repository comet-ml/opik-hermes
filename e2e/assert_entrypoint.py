"""Assert the installed opik-hermes exposes a loadable Hermes plugin entry point.

Single source of truth for the entry-point contract — used by ci.yml,
publish.yml (both install the built wheel first), and e2e/Dockerfile.wheel
(installed into Hermes' venv). It reproduces exactly what Hermes' plugin loader
does, so a green run here means the plugin will actually load in Hermes:

  ep = entry_points(group="hermes_agent.plugins")["opik"]
  loaded = ep.load()                       # Hermes does this
  getattr(loaded, "register")              # ...then this

The entry point must therefore resolve to the MODULE (``opik = opik_hermes``),
NOT the function (``opik_hermes:register``): the latter makes ep.load() return
the function, ``getattr(function, "register")`` is None, and the plugin silently
registers no hooks (the 0.1.0 bug this guards against).

Run in the environment where opik-hermes is installed:
    python e2e/assert_entrypoint.py
"""

from __future__ import annotations

import sys
import types
from importlib.metadata import entry_points, version

GROUP = "hermes_agent.plugins"
NAME = "opik"


def main() -> None:
    eps = [e for e in entry_points(group=GROUP) if e.name == NAME]
    if not eps:
        sys.exit(
            f"no {NAME!r} entry point in group {GROUP!r} — is opik-hermes installed "
            "in this environment?"
        )
    ep = eps[0]

    # Exactly what Hermes' loader does: load the entry point, then look for
    # `register` on the result. This is why the entry point must point at the
    # MODULE (opik = opik_hermes), not the function (opik_hermes:register).
    loaded = ep.load()
    if not isinstance(loaded, types.ModuleType):
        sys.exit(
            f"entry point {NAME!r} loaded {type(loaded).__name__}, not a module "
            f"(value {ep.value!r}). Set 'opik = opik_hermes' in pyproject "
            f'[project.entry-points."{GROUP}"] so Hermes can getattr(module, "register").'
        )
    if not callable(getattr(loaded, "register", None)):
        sys.exit(f"loaded module {loaded.__name__!r} has no callable register()")

    print(
        f"OK: opik-hermes {version('opik-hermes')} — {NAME} -> {loaded.__name__}.register"
    )


if __name__ == "__main__":
    main()
