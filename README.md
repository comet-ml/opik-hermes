# opik-hermes

Opik observability for the [Hermes agent](https://github.com/NousResearch/hermes-agent).

Traces Hermes conversations, LLM calls, and tool usage to
[Opik](https://github.com/comet-ml/opik) — Comet's open-source LLM and agent
observability platform. Modeled on Hermes' bundled Langfuse observability
plugin, swapped to the Opik Python SDK.

## What it captures

- A **root trace** per agent turn (`Hermes turn`), grouped into Opik threads
  by Hermes `session_id`.
- An **LLM span** (`type=llm`) per API call, with input messages, assistant
  output, model/provider, token usage, and USD cost (via Hermes'
  `agent.usage_pricing`).
- A **tool span** (`type=tool`) per tool call, with arguments and results.

## Install

The plugin reaches Hermes one of two ways:

### As a pip plugin (entry-point)

```bash
pip install opik-hermes        # exposes the `hermes_agent.plugins` entry point
hermes plugins enable observability/opik
```

### As a user plugin (bundled / mounted)

Drop `observability/opik/` into `~/.hermes/plugins/opik/`, then:

```bash
pip install opik
hermes plugins enable observability/opik
```

## Configure

Local open-source Opik needs **no API key**:

```bash
# ~/.hermes/.env
OPIK_URL_OVERRIDE=http://localhost:5174/api
OPIK_PROJECT_NAME=hermes
```

Comet-hosted Opik:

```bash
OPIK_API_KEY=...
OPIK_WORKSPACE=your-workspace
```

See [`observability/opik/README.md`](observability/opik/README.md) for the
full configuration and tuning reference.

## Repository layout

```
opik-hermes/
├── pyproject.toml              # pip package + hermes_agent.plugins entry point
├── observability/opik/         # the plugin, in Hermes' flat bundled-plugin shape
│   ├── plugin.yaml
│   ├── __init__.py             # register(ctx) + hooks (Opik Python SDK)
│   └── README.md
└── tests/
```

The `observability/opik/` directory is byte-compatible with Hermes' bundled
plugin layout, so it can be mounted as a user plugin, vendored into Hermes, or
shipped as the pip package above — all from one source.
