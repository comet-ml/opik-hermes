"""Assert the mock-Opik journal captured a well-formed Hermes trace.

Reads the JSON-lines journal written by mock_opik_server.py and checks the
plugin produced what we expect from one mocked turn: a root "Hermes turn"
trace, at least one LLM span, and a tool span — none of them "NA" (missing
name/type). Exits non-zero with a readable diff if anything is missing.
"""

from __future__ import annotations

import json
import os
import sys

JOURNAL = os.environ.get("MOCK_OPIK_JOURNAL", "journal/opik-journal.jsonl")


def _load():
    if not os.path.exists(JOURNAL):
        sys.exit(f"FAIL: journal not found at {JOURNAL} — did mock-opik run?")
    rows = []
    with open(JOURNAL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _flatten_items(payload):
    """SDK batches arrive as {'traces': [...]} / {'spans': [...]} or bare lists."""
    if isinstance(payload, dict):
        for key in ("traces", "spans"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


def main() -> None:
    rows = _load()
    traces, spans = [], []
    for r in rows:
        items = _flatten_items(r.get("payload"))
        if r.get("kind") == "traces":
            traces += items
        elif r.get("kind") == "spans":
            spans += items

    print(f"journal: {len(rows)} rows | traces={len(traces)} spans={len(spans)}")

    errors = []

    # 1. A root trace exists and is named (descriptive or the fallback).
    if not traces:
        errors.append("no traces captured")
    else:
        names = [t.get("name") for t in traces]
        if not any(n for n in names):
            errors.append(f"trace(s) have no name: {names}")

    # 2. At least one LLM span and one tool span, all with name+type (no NA).
    llm = [s for s in spans if s.get("type") == "llm"]
    tool = [s for s in spans if s.get("type") == "tool"]
    na = [s for s in spans if not s.get("name") or not s.get("type")]

    if not llm:
        errors.append("no llm spans captured")
    if not tool:
        errors.append("no tool spans captured")
    if na:
        errors.append(f"{len(na)} NA span(s) (missing name/type): {[s.get('id') for s in na]}")

    if errors:
        print("\n=== E2E ASSERTION FAILED ===")
        for e in errors:
            print("  -", e)
        print("\n--- captured span summary ---")
        for s in spans:
            print(f"  type={s.get('type')!r} name={s.get('name')!r}")
        sys.exit(1)

    print("\n=== E2E PASSED ===")
    print(f"  trace: {traces[0].get('name')!r}")
    print(f"  llm spans: {len(llm)} | tool spans: {len(tool)} | NA: 0")


if __name__ == "__main__":
    main()
