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

Install the package into the same Python environment as Hermes, then enable
the plugin:

```bash
pip install opik-hermes
hermes plugins enable observability/opik
```

`pip install opik-hermes` pulls in the `opik` SDK automatically (it's a
declared dependency) and registers the plugin with Hermes via the
`hermes_agent.plugins` entry point — no manual file copying.

## Configure

Set these in `~/.hermes/.env` (or via `hermes tools`).

Local open-source Opik needs **no API key**:

```bash
# ~/.hermes/.env
OPIK_URL_OVERRIDE=http://localhost:5173/api
OPIK_PROJECT_NAME=hermes
```

Comet-hosted Opik:

```bash
OPIK_API_KEY=...
OPIK_WORKSPACE=your-workspace
```

## Use it

Talk to Hermes however you normally do — every turn is traced to the
`hermes` project (or whatever `OPIK_PROJECT_NAME` you set):

```bash
hermes chat -q "list the files here and tell me how many there are"
```

…or open the Hermes web UI at **http://localhost:9119** and use the Chat tab.
Then open Opik and look for a trace named after your message, with an LLM span
and a span per tool call.

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
