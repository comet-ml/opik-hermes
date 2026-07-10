#!/usr/bin/env bash
# Print the [project].version from pyproject.toml (no leading v).
#
# Shared by release.yml (drift guard: version vs last git tag) and publish.yml
# (guard: version vs the build's tag ref). Both need to read the version the
# same robust way — the comparison each performs differs and stays in the
# caller. Keeping the read in one place means the TOML parsing can't drift
# between the two workflows.
#
# Usage: scripts/pyproject-version.sh [path-to-pyproject.toml]   # default: ./pyproject.toml
set -euo pipefail

FILE="${1:-pyproject.toml}"
python3 - "$FILE" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    print(tomllib.load(fh)["project"]["version"])
PY
