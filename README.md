<h1 align="center" style="border-bottom: none">
  <div>
    <a href="https://www.comet.com/site/products/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=header_img&utm_campaign=opik">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/logo-dark-mode.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg">
        <img alt="Comet Opik logo" src="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg" width="200" />
      </picture>
    </a>
    <br />
    🔭 Opik Observability for Hermes
  </div>
</h1>

<p align="center">
  Plugin for the <a href="https://github.com/NousResearch/hermes-agent">Hermes agent</a> that exports conversation, LLM, and tool traces to <br/>
  <a href="https://www.comet.com/docs/opik/">Opik</a> for observability and monitoring.
</p>

<div align="center">

[![License](https://img.shields.io/github/license/comet-ml/opik-hermes)](./LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/opik-hermes)](https://pypi.org/project/opik-hermes/)
[![Python versions](https://img.shields.io/pypi/pyversions/opik-hermes)](https://pypi.org/project/opik-hermes/)

</div>

## Why this plugin

[Opik](https://github.com/comet-ml/opik) is Comet's open-source LLM and agent
observability, tracing, evaluation, and optimization platform. `opik-hermes`
adds native Opik tracing for [Hermes agent](https://github.com/NousResearch/hermes-agent)
runs — conversations, LLM calls, and tool usage — with no code changes to your
agent. Modeled on Hermes' bundled Langfuse observability plugin, swapped to the
Opik Python SDK.

## What it captures

- A **root trace** per agent turn (`Hermes turn`), grouped into Opik threads
  by Hermes `session_id`.
- An **LLM span** (`type=llm`) per API call, with input messages, assistant
  output, model/provider, token usage, and USD cost (via Hermes'
  `agent.usage_pricing`).
- A **tool span** (`type=tool`) per tool call, with arguments and results.

## Install

Install the package into the same Python environment as Hermes, then enable
the plugin by adding it to `plugins.enabled` in `~/.hermes/config.yaml`:

```bash
pip install opik-hermes
```

```yaml
# ~/.hermes/config.yaml
plugins:
  enabled: [opik]
```

`pip install opik-hermes` pulls in the `opik` SDK automatically (it's a
declared dependency) and registers the plugin with Hermes via the
`hermes_agent.plugins` entry point (name `opik`) — no manual file copying.
Enable it through `plugins.enabled` as above; the entry-point name is `opik`.

> This is the **pip install** path. A directory-based install
> (`hermes plugins install comet-ml/opik-hermes`, enabled via
> `hermes plugins enable observability/opik`) is tracked separately — see the
> follow-up work in the repo's issues.

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
