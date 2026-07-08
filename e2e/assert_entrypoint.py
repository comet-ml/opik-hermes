"""Assert a built wheel exposes the Hermes plugin entry point correctly.

Single source of truth for the entry-point contract, shared by the `build` job
(ci.yml) and the release publish (publish.yml) so the check lives in one place.

Hermes loads a plugin via ``ep.load()`` then ``getattr(<result>, "register")``,
expecting a MODULE. So the entry point must resolve to ``opik = opik_hermes``
(the module), NOT ``opik_hermes:register`` (the function) — the latter loads the
function, ``getattr(function, "register")`` is None, and the plugin silently
registers no hooks.

Usage: python e2e/assert_entrypoint.py [wheel-glob]   (default: dist/*.whl)
"""

from __future__ import annotations

import configparser
import glob
import sys
import zipfile

EXPECTED = "opik_hermes"


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "dist/*.whl"
    wheels = glob.glob(pattern)
    if not wheels:
        sys.exit(f"no wheel matched {pattern!r}")
    wheel = wheels[0]

    zf = zipfile.ZipFile(wheel)
    names = zf.namelist()
    assert "opik_hermes/__init__.py" in names, "plugin module missing from wheel"

    ep_file = next((n for n in names if n.endswith("entry_points.txt")), None)
    assert ep_file, "entry_points.txt missing from wheel"
    body = zf.read(ep_file).decode()

    cp = configparser.ConfigParser()
    cp.read_string(body)
    assert "hermes_agent.plugins" in cp, "hermes_agent.plugins group missing"
    val = cp["hermes_agent.plugins"].get("opik", "").strip()
    assert val == EXPECTED, (
        f"opik entry point must be {EXPECTED!r} (the module) so Hermes can "
        f"getattr(module, 'register'); got {val!r}. Check pyproject "
        f'[project.entry-points."hermes_agent.plugins"].'
    )
    print(f"OK: {wheel} — entry point opik -> {val}")


if __name__ == "__main__":
    main()
