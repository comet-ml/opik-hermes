#!/usr/bin/env bash
# End-to-end test against a REAL Opik instance (not the mock).
#
# Brings up Opik's own open-source backend (+ its deps: mysql, clickhouse,
# redis, minio) via Opik's published compose, runs latest Hermes + our plugin
# against a mock LLM, and asserts the trace/spans landed by querying the REAL
# Opik REST API. The LLM is still mocked (no model cost / keys); only Opik is
# real. Heavier + slower than run_e2e.sh — meant for cron/dispatch, not every PR.
#
# Env:
#   OPIK_REPO   path to a checkout of comet-ml/opik (has deployment/docker-compose)
#   HERMES_IMAGE  default nousresearch/hermes-agent:latest
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES_IMAGE="${HERMES_IMAGE:-nousresearch/hermes-agent:latest}"
NET="opik-hermes-e2e-real"
PROJECT="hermes-e2e-real"
WORK="$(mktemp -d)"
HERMES_HOME="$WORK/hermes"; mkdir -p "$HERMES_HOME/plugins"

# Opik compose: use a provided checkout, else shallow-clone it.
if [ -n "${OPIK_REPO:-}" ] && [ -d "$OPIK_REPO/deployment/docker-compose" ]; then
  OPIK_COMPOSE_DIR="$OPIK_REPO/deployment/docker-compose"
else
  echo "==> cloning comet-ml/opik (shallow, for its compose)"
  git clone --depth 1 https://github.com/comet-ml/opik.git "$WORK/opik" >/dev/null 2>&1
  OPIK_COMPOSE_DIR="$WORK/opik/deployment/docker-compose"
fi
OPIK_PROJECT="opikhermese2e"
# Compose invocation with our no-host-ports override layered on top, so the
# E2E Opik never collides with an Opik already running on the host (all
# comms are container-to-container). Run from the Opik compose dir.
OVERRIDE="$REPO_ROOT/e2e/opik-no-ports.override.yaml"
opik_compose() {
  ( cd "$OPIK_COMPOSE_DIR" && docker compose -p "$OPIK_PROJECT" \
      -f docker-compose.yaml -f "$OVERRIDE" "$@" )
}

cleanup() {
  docker rm -f e2e-real-llm e2e-real-hermes >/dev/null 2>&1 || true
  opik_compose down -v >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> starting real Opik backend + its deps via Opik compose (no host ports)"
# Start the backend AND its dependencies explicitly. `up backend` alone does
# not reliably pull depends_on across Opik's layered compose overrides, which
# left Redis/MySQL/ClickHouse down and the backend 500ing on every request.
# clickhouse-init and mc are one-shot initializers ClickHouse/MinIO need.
opik_compose up -d \
  mysql redis clickhouse clickhouse-init minio mc backend

# Wait for the backend health-check (compose marks it healthy on /health-check).
echo "==> waiting for Opik backend to be healthy"
BE_CID=""
for _ in $(seq 1 120); do
  BE_CID=$(opik_compose ps -q backend 2>/dev/null || true)
  if [ -n "$BE_CID" ]; then
    st=$(docker inspect -f '{{.State.Health.Status}}' "$BE_CID" 2>/dev/null || echo starting)
    [ "$st" = "healthy" ] && { echo "backend healthy"; break; }
  fi
  sleep 5
done
[ -n "$BE_CID" ] || { echo "FAIL: backend never started"; exit 1; }

# The Opik backend joins a compose-managed network; attach our Hermes + mock
# LLM to the SAME network so Hermes can reach the backend by service name.
OPIK_NET=$(docker inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$BE_CID" | awk '{print $1}')
echo "==> Opik backend on network: $OPIK_NET (container: $(docker inspect -f '{{.Name}}' "$BE_CID" | sed s:^/::))"
BE_NAME=$(docker inspect -f '{{.Name}}' "$BE_CID" | sed 's:^/::')

echo "==> mock LLM on the Opik network"
docker run -d --name e2e-real-llm --network "$OPIK_NET" -e MOCK_LLM_PORT=18790 \
  -v "$REPO_ROOT/e2e/mock_llm_server.py:/srv/s.py:ro" \
  python:3.12-slim python /srv/s.py >/dev/null

# Hermes home: model -> mock LLM; Opik -> the REAL backend by service name:8080.
cat > "$HERMES_HOME/config.yaml" <<YAML
model:
  default: gpt-5
  provider: openai-api
  base_url: http://e2e-real-llm:18790/v1
plugins:
  enabled: [opik]
agent: {max_turns: 4}
terminal: {backend: local}
YAML
cat > "$HERMES_HOME/.env" <<ENV
OPENAI_API_KEY=mock-key
OPENAI_BASE_URL=http://e2e-real-llm:18790/v1
OPIK_URL_OVERRIDE=http://${BE_NAME}:8080
OPIK_PROJECT_NAME=${PROJECT}
HERMES_OPIK_DEBUG=true
ENV
cp -R "$REPO_ROOT/observability/opik" "$HERMES_HOME/plugins/opik"

echo "==> building Hermes+opik image"
docker build -q --build-arg HERMES_IMAGE="$HERMES_IMAGE" \
  -f "$REPO_ROOT/e2e/Dockerfile" -t opik-hermes-e2e:local "$REPO_ROOT" >/dev/null

echo "==> running one Hermes turn (Opik is REAL)"
if command -v timeout >/dev/null 2>&1; then TIMEOUT="timeout 180"; else TIMEOUT=""; fi
HERMES_LOG="$WORK/hermes.log"
$TIMEOUT docker run --rm --name e2e-real-hermes --network "$OPIK_NET" \
  -e HERMES_UID=0 -e HERMES_GID=0 -v "$HERMES_HOME:/opt/data" \
  opik-hermes-e2e:local \
  sh -c 'hermes chat -q "Compute 2 to the power 10 and report the number." --provider openai-api --model gpt-5 2>&1' \
  < /dev/null > "$HERMES_LOG" 2>&1 || echo "(hermes turn non-zero/timeout; assertion judges from Opik)"
tail -15 "$HERMES_LOG" || true

# Upsert-only lifecycle must not trip the SDK batching warning (OPIK-7279).
echo "==> asserting no Opik batching warning in Hermes output"
if grep -Ei "may cause data loss|Calling Trace\.update\(\) shortly after creation" "$HERMES_LOG"; then
  echo "=== E2E FAILED: Opik batching warning present (lifecycle not upsert-only) ==="
  exit 1
fi
echo "  (no batching warning — lifecycle is upsert-only)"

echo "==> letting the SDK flush, then querying the REAL Opik API"
sleep 5

# Assert via the real REST API from a container on the Opik network (hits
# backend:8080 directly). Uses a mounted script (stdlib urllib only — no pip)
# to avoid nested-heredoc quoting pitfalls.
docker run --rm --network "$OPIK_NET" -e BE="$BE_NAME" -e PROJECT="$PROJECT" \
  -v "$REPO_ROOT/e2e/assert_real_opik.py:/assert.py:ro" \
  python:3.12-slim python /assert.py
