# Opik Observability Plugin

Traces Hermes conversations, LLM calls, and tool usage to
[Opik](https://github.com/comet-ml/opik), Comet's open-source LLM
observability platform.

This plugin is **opt-in** — it only loads when you explicitly enable it.

## Enable

```bash
pip install opik-hermes      # also pulls in the `opik` SDK
hermes plugins enable observability/opik
```

## Point hermes at your Opik

Set these in `~/.hermes/.env` (or via `hermes tools`). The Opik Python SDK reads
the `OPIK_*` env vars directly — there is no interactive step required. See
[`.env.example`](.env.example) for a copy-paste template.

### Opik Cloud

```bash
OPIK_URL_OVERRIDE=https://www.comet.com/opik/api
OPIK_API_KEY=your-api-key-here
OPIK_WORKSPACE=default
OPIK_PROJECT_NAME=hermes
```

### Self-hosted / local (open-source Opik) — NO API key required

```bash
OPIK_URL_OVERRIDE=http://localhost:5173/api
OPIK_WORKSPACE=default
OPIK_PROJECT_NAME=hermes
```

### No-auth custom deployment

For an unauthenticated Opik behind your own URL, point at it and omit the key:

```bash
OPIK_URL_OVERRIDE=https://opik.internal.example.com/api
OPIK_WORKSPACE=default
OPIK_PROJECT_NAME=hermes
```

> With **no** config, the SDK defaults to Opik Cloud (`https://www.comet.com/opik/api`)
> and, without an API key, logs a repeated `API key must be specified` warning and
> sends nothing. The plugin still fails open (its hooks never crash Hermes), but the
> SDK is **not** silent about the missing key. Always set `OPIK_URL_OVERRIDE`
> explicitly so traces land where you expect.

### Baked image / container

To get traces flowing on first run with **zero interactive steps**, bake the
target as image environment (Dockerfile `ENV`, compose `environment:`, or your
orchestrator's env config):

- `OPIK_URL_OVERRIDE` — the Opik endpoint (required to route away from Cloud)
- `OPIK_WORKSPACE` — the workspace (`default` for local/open-source)
- `OPIK_API_KEY` — **only** for authenticated (Comet-hosted / auth'd self-hosted) deployments
- `OPIK_PROJECT_NAME` — the project to log under (defaults to `hermes`)

The container starts, Hermes loads the plugin, `opik.Opik()` reads these vars,
and traces flow immediately — no `hermes tools` prompt, no config file.

### Interactive alternative (humans, not containers)

For local setup you can instead run the Opik Python SDK's CLI:

```bash
opik configure
```

It walks you through deployment/URL/key/workspace and writes `~/.opik.config`,
which this plugin's `opik.Opik()` then reads. Explicit `OPIK_*` env vars always
win over the config file, so the baked-image path above overrides it.

Without the `opik` SDK the hooks no-op silently — the plugin fails open.

## Verify

```bash
hermes plugins list                 # observability/opik should show "enabled"
hermes chat -q "hello"              # one-shot turn from the CLI
```

…or open the Hermes web UI at **http://localhost:9119** and use the Chat tab.
Either way, check Opik for a trace named after your message.

## Optional tuning

```bash
HERMES_OPIK_TAGS=production,team-x   # extra tags applied to every trace
HERMES_OPIK_MAX_CHARS=12000          # max chars per field (default: 12000)
HERMES_OPIK_DEBUG=true               # verbose plugin logging
```

## Disable

```bash
hermes plugins disable observability/opik
```
