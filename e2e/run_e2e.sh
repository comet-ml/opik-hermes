#!/usr/bin/env bash
# End-to-end test: latest Hermes (official image) + our plugin, driven against
# a mock LLM and mock Opik, asserting the captured spans. No real keys, no
# external network for the agent — everything on one private docker network.
#
# Run from the repo root:  bash e2e/run_e2e.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NET="opik-hermes-e2e"
HERMES_IMAGE="${HERMES_IMAGE:-nousresearch/hermes-agent:latest}"
WORK="$(mktemp -d)"
JOURNAL_DIR="$WORK/journal"
HERMES_HOME="$WORK/hermes"
mkdir -p "$JOURNAL_DIR" "$HERMES_HOME/plugins"

cleanup() {
  docker rm -f e2e-mock-llm e2e-mock-opik e2e-hermes >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> pulling $HERMES_IMAGE"
docker pull -q "$HERMES_IMAGE" >/dev/null

# Build a derived image with opik baked into Hermes' venv (build-time has
# internet; the run then needs none). This is what a real deployment does.
E2E_IMAGE="opik-hermes-e2e:local"
echo "==> building $E2E_IMAGE (opik baked in)"
docker build -q --build-arg HERMES_IMAGE="$HERMES_IMAGE" \
  -f "$REPO_ROOT/e2e/Dockerfile" -t "$E2E_IMAGE" "$REPO_ROOT" >/dev/null

echo "==> private network (no internet; only mocks reachable)"
docker network create "$NET" >/dev/null

# --- mock servers (unprivileged python:slim, our scripts mounted) ----------
echo "==> mock-opik + mock-llm"
docker run -d --name e2e-mock-opik --network "$NET" \
  -e MOCK_OPIK_JOURNAL=/journal/opik-journal.jsonl -e MOCK_OPIK_PORT=5173 \
  -v "$REPO_ROOT/e2e/mock_opik_server.py:/srv/s.py:ro" \
  -v "$JOURNAL_DIR:/journal" \
  python:3.12-slim python /srv/s.py >/dev/null

docker run -d --name e2e-mock-llm --network "$NET" \
  -e MOCK_LLM_PORT=18790 \
  -v "$REPO_ROOT/e2e/mock_llm_server.py:/srv/s.py:ro" \
  python:3.12-slim python /srv/s.py >/dev/null

# --- Hermes home: config + .env + plugin ----------------------------------
# Model points at the mock LLM (container DNS name); Opik points at mock-opik.
cat > "$HERMES_HOME/config.yaml" <<YAML
model:
  default: gpt-5
  provider: openai-api
  base_url: http://e2e-mock-llm:18790/v1
providers: {}
plugins:
  enabled:
    - opik
agent:
  max_turns: 4
terminal:
  backend: local
YAML

cat > "$HERMES_HOME/.env" <<ENV
OPENAI_API_KEY=mock-key
OPENAI_BASE_URL=http://e2e-mock-llm:18790/v1
OPIK_URL_OVERRIDE=http://e2e-mock-opik:5173/api
OPIK_PROJECT_NAME=hermes-e2e
HERMES_OPIK_DEBUG=true
ENV

# Plugin as a user plugin.
cp -R "$REPO_ROOT/observability/opik" "$HERMES_HOME/plugins/opik"

# --- run one turn in the Hermes container ----------------------------------
# Install opik into the image's venv at container start (read-only-venv caveat
# does not apply here — we install before invoking hermes, in the same shell),
# enable the plugin, then run a single chat turn.
echo "==> running one Hermes turn"
# Hard timeout guard so a stray interactive prompt can never wedge CI; stdin
# from /dev/null so any prompt gets EOF instead of blocking. `timeout` is
# present on Linux CI runners but not stock macOS — fall back to no wrapper
# locally (the /dev/null stdin still prevents the prompt-hang).
if command -v timeout >/dev/null 2>&1; then TIMEOUT="timeout 180"; else TIMEOUT=""; fi
# opik is already baked into $E2E_IMAGE; the plugin is enabled via config.yaml.
# No runtime install (the isolated network has no internet). Do NOT run
# `hermes plugins enable` (it prompts and would hang with no TTY). stdin from
# /dev/null so any stray prompt gets EOF instead of blocking.
$TIMEOUT docker run --rm --name e2e-hermes --network "$NET" \
  -e HERMES_UID=0 -e HERMES_GID=0 \
  -v "$HERMES_HOME:/opt/data" \
  "$E2E_IMAGE" \
  sh -c '
    hermes chat -q "Compute 2 to the power 10 and report the number." \
      --provider openai-api --model gpt-5 2>&1 | tail -20
  ' < /dev/null || echo "(hermes turn exited non-zero / timed out; assertion judges from the journal)"

# Give the SDK background flush a moment to POST to mock-opik.
sleep 3

# Preserve the journal outside the temp dir (trap cleans WORK) for debugging.
cp "$JOURNAL_DIR/opik-journal.jsonl" /tmp/opik-e2e-journal.jsonl 2>/dev/null || true
echo "==> journal saved to /tmp/opik-e2e-journal.jsonl ($(wc -l < "$JOURNAL_DIR/opik-journal.jsonl" 2>/dev/null || echo 0) rows)"

echo "==> asserting journal"
MOCK_OPIK_JOURNAL="$JOURNAL_DIR/opik-journal.jsonl" python3 "$REPO_ROOT/e2e/assert_journal.py"
