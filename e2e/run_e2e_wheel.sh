#!/usr/bin/env bash
# End-to-end test of the PIP-INSTALL path: build the wheel from source, install
# it into the latest Hermes image, and drive a turn against a mock LLM + mock
# Opik — asserting the plugin was discovered via its hermes_agent.plugins ENTRY
# POINT (no plugin directory copied in) and produced the expected spans.
#
# Complements run_e2e.sh (which tests the directory-install path). Both install
# methods are supported, so CI exercises both. No real keys; the agent has no
# internet at run time.
#
# Run from the repo root:  bash e2e/run_e2e_wheel.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NET="opik-hermes-e2e-wheel"
HERMES_IMAGE="${HERMES_IMAGE:-nousresearch/hermes-agent:latest}"
WORK="$(mktemp -d)"
JOURNAL_DIR="$WORK/journal"
HERMES_HOME="$WORK/hermes"
CTX="$WORK/ctx"
mkdir -p "$JOURNAL_DIR" "$HERMES_HOME" "$CTX"

cleanup() {
  docker rm -f e2e-w-mock-llm e2e-w-mock-opik e2e-w-hermes >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> pulling $HERMES_IMAGE"
docker pull -q "$HERMES_IMAGE" >/dev/null

# --- build the wheel from source into the docker build context ---------------
echo "==> building opik-hermes wheel from source"
python3 -m venv "$WORK/.venv"
"$WORK/.venv/bin/pip" install -q -U build
"$WORK/.venv/bin/python" -m build --wheel --outdir "$CTX/dist" "$REPO_ROOT" >/dev/null
cp "$REPO_ROOT/e2e/Dockerfile.wheel" "$CTX/Dockerfile"
# Stage the shared entry-point check so the Dockerfile can COPY it in (the
# build-time guard uses the same single-source-of-truth script as CI).
mkdir -p "$CTX/e2e"
cp "$REPO_ROOT/e2e/assert_entrypoint.py" "$CTX/e2e/assert_entrypoint.py"
echo "==> wheel: $(ls "$CTX/dist")"

# --- build the Hermes image with the wheel pip-installed (ep assert at build) -
E2E_IMAGE="opik-hermes-e2e-wheel:local"
echo "==> building $E2E_IMAGE (opik-hermes pip-installed)"
docker build -q --build-arg HERMES_IMAGE="$HERMES_IMAGE" \
  -f "$CTX/Dockerfile" -t "$E2E_IMAGE" "$CTX" >/dev/null

echo "==> private network (no internet; only mocks reachable)"
docker network create "$NET" >/dev/null

echo "==> mock-opik + mock-llm"
docker run -d --name e2e-w-mock-opik --network "$NET" \
  -e MOCK_OPIK_JOURNAL=/journal/opik-journal.jsonl -e MOCK_OPIK_PORT=5173 \
  -v "$REPO_ROOT/e2e/mock_opik_server.py:/srv/s.py:ro" \
  -v "$JOURNAL_DIR:/journal" \
  python:3.12-slim python /srv/s.py >/dev/null

docker run -d --name e2e-w-mock-llm --network "$NET" \
  -e MOCK_LLM_PORT=18790 \
  -v "$REPO_ROOT/e2e/mock_llm_server.py:/srv/s.py:ro" \
  python:3.12-slim python /srv/s.py >/dev/null

# --- Hermes home: config + .env, NO plugin dir (entry-point discovery only) --
cat > "$HERMES_HOME/config.yaml" <<YAML
model:
  default: gpt-5
  provider: openai-api
  base_url: http://e2e-w-mock-llm:18790/v1
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
OPENAI_BASE_URL=http://e2e-w-mock-llm:18790/v1
OPIK_URL_OVERRIDE=http://e2e-w-mock-opik:5173/api
OPIK_PROJECT_NAME=hermes-e2e-wheel
HERMES_OPIK_DEBUG=true
ENV
# NOTE: deliberately NO `cp observability/opik` here — the plugin must be found
# via the pip entry point, which is the whole point of this path.

echo "==> running one Hermes turn (plugin from pip entry point)"
if command -v timeout >/dev/null 2>&1; then TIMEOUT="timeout 180"; else TIMEOUT=""; fi
$TIMEOUT docker run --rm --name e2e-w-hermes --network "$NET" \
  -e HERMES_UID=0 -e HERMES_GID=0 \
  -v "$HERMES_HOME:/opt/data" \
  "$E2E_IMAGE" \
  sh -c '
    hermes chat -q "Compute 2 to the power 10 and report the number." \
      --provider openai-api --model gpt-5 2>&1 | tail -20
  ' < /dev/null || echo "(hermes turn exited non-zero / timed out; assertion judges from the journal)"

sleep 3
cp "$JOURNAL_DIR/opik-journal.jsonl" /tmp/opik-e2e-wheel-journal.jsonl 2>/dev/null || true
echo "==> journal saved ($(wc -l < "$JOURNAL_DIR/opik-journal.jsonl" 2>/dev/null || echo 0) rows)"

echo "==> asserting journal"
MOCK_OPIK_JOURNAL="$JOURNAL_DIR/opik-journal.jsonl" python3 "$REPO_ROOT/e2e/assert_journal.py"
