# E2E image: latest Hermes + the opik SDK + our plugin, baked in at build time.
#
# Building (not runtime-installing) means the E2E container needs no internet —
# it runs on an isolated docker network with only the mock LLM/Opik reachable.
# This also mirrors how a real deployment ships the plugin (install into the
# same venv Hermes runs in), and sidesteps the read-only-venv-at-runtime limit.
ARG HERMES_IMAGE=nousresearch/hermes-agent:latest
FROM ${HERMES_IMAGE}

# Install the opik SDK into Hermes' own venv (build-time = has network).
# Current Hermes images ship the venv without pip and keep uv internal to their
# package manager, so bootstrap pip from the stdlib; older images without
# ensurepip still have uv on PATH.
RUN { /opt/hermes/.venv/bin/python -m ensurepip \
      && /opt/hermes/.venv/bin/python -m pip install --no-cache-dir opik; } \
    || uv pip install --python /opt/hermes/.venv/bin/python --no-cache-dir opik

# Ship the plugin as a bundled user plugin. /opt/data is the Hermes home
# (mounted at runtime); bundle under /opt/hermes and symlink so a mounted
# /opt/data doesn't shadow it. Simpler: copy into the image's plugins path
# that Hermes also scans. We copy to a build-time staging dir and the entrypoint
# script (run_e2e.sh) mounts the real plugin via /opt/data/plugins anyway, so
# this COPY is a fallback for when /opt/data is not pre-populated.
COPY observability/opik /opt/hermes/e2e-plugin/opik
