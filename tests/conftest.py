"""Test harness for the Opik Hermes plugin.

The plugin ships as a flat Hermes-plugin directory (``observability/opik/``)
and is loaded by Hermes via ``spec_from_file_location`` under the module name
``hermes_plugins.opik`` — never as a top-level import. We mirror that here:
each test gets a freshly-loaded module instance (so module-global trace state
and the cached client don't leak between tests), wired to a fake Opik client.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

PLUGIN_PATH = (
    Path(__file__).resolve().parent.parent / "observability" / "opik" / "__init__.py"
)


def _load_plugin_module():
    """Load the plugin source as a fresh module instance.

    A unique module name per load avoids importlib's module cache so each test
    starts with clean module-global state (``_TRACE_STATE``, ``_OPIK_CLIENT``).
    """
    name = f"hermes_plugins.opik_test_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(name, PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeSpan:
    """Records update/end calls; supports nested .span() for tool-on-llm cases."""

    def __init__(self, events: list, name: str, type: str, create_kwargs: dict = None):
        self._events = events
        self.name = name
        self.type = type
        self.create_kwargs = create_kwargs or {}
        self.ended = False
        self.updates: list[dict] = []

    def span(self, **kwargs: Any) -> "FakeSpan":
        self._events.append(("span.span", kwargs.get("type"), kwargs.get("name")))
        return FakeSpan(self._events, kwargs.get("name"), kwargs.get("type"), kwargs)

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)
        self._events.append(("span.update", self.name, sorted(kwargs)))

    def end(self, **kwargs: Any) -> None:
        self.ended = True
        self._events.append(("span.end", self.name))


class FakeTrace:
    def __init__(self, events: list, trace_id: str, kwargs: dict):
        self._events = events
        self.id = trace_id
        self.create_kwargs = kwargs
        self.ended = False
        self.updates: list[dict] = []
        self.spans: list[FakeSpan] = []

    def span(self, **kwargs: Any) -> FakeSpan:
        self._events.append(("trace.span", kwargs.get("type"), kwargs.get("name")))
        s = FakeSpan(self._events, kwargs.get("name"), kwargs.get("type"), kwargs)
        self.spans.append(s)
        return s

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)
        self._events.append(("trace.update", sorted(kwargs)))

    def end(self, **kwargs: Any) -> None:
        self.ended = True
        self._events.append(("trace.end",))


class FakeOpik:
    """Minimal stand-in for opik.Opik that records the lifecycle."""

    def __init__(self, **_: Any):
        self.events: list = []
        self.traces: list[FakeTrace] = []
        self.flushed = 0

    def trace(self, **kwargs: Any) -> FakeTrace:
        tid = f"trace-{len(self.traces) + 1}"
        self.events.append(("client.trace", kwargs.get("name")))
        t = FakeTrace(self.events, tid, kwargs)
        self.traces.append(t)
        return t

    def flush(self) -> None:
        self.flushed += 1
        self.events.append(("flush",))


@pytest.fixture
def plugin():
    """A freshly-loaded plugin module with a FakeOpik client already installed.

    The plugin is a package; its submodules load under the same fresh parent so
    module-global state (client cache, trace store) is isolated per test. Tests
    reach cohesive modules via ``plugin.client`` / ``plugin.state`` / etc.
    """
    mod = _load_plugin_module()
    client = FakeOpik()
    mod.client.set_client(client)
    mod._fake = client  # convenience handle for assertions
    return mod


@pytest.fixture
def plugin_no_sdk():
    """A plugin module loaded as if the opik SDK were missing (fail-open path)."""
    mod = _load_plugin_module()
    mod.client._opik_sdk = None
    mod.client.set_client(None)
    return mod
