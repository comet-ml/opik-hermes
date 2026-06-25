# Opik Observability Plugin

Traces Hermes conversations, LLM calls, and tool usage to
[Opik](https://github.com/comet-ml/opik), Comet's open-source LLM
observability platform.

This plugin is **opt-in** — it only loads when you explicitly enable it.

## Enable

```bash
# Manual
pip install opik
hermes plugins enable observability/opik
```

## Configuration

Set these in `~/.hermes/.env` (or via `hermes tools`):

```bash
# Local open-source Opik — NO API key required:
OPIK_URL_OVERRIDE=http://localhost:5174/api
OPIK_PROJECT_NAME=hermes

# Comet-hosted / self-hosted-with-auth Opik:
OPIK_API_KEY=...
OPIK_WORKSPACE=your-workspace
```

Without the `opik` SDK the hooks no-op silently — the plugin fails open.

## Verify

```bash
hermes plugins list                 # observability/opik should show "enabled"
hermes chat -q "hello"              # then check Opik for a "Hermes turn" trace
```

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
