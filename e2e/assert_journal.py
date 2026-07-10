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
    traces, spans, updates = [], [], []
    for r in rows:
        kind = r.get("kind")
        items = _flatten_items(r.get("payload"))
        if kind == "traces":
            traces += items
        elif kind == "spans":
            spans += items
        elif kind == "update":
            updates.append(r.get("payload"))

    # Upsert-only lifecycle: the create and the finalize arrive as two SEPARATE
    # trace-batch rows sharing one id. Merge by id, mimicking real Opik's
    # last-write-wins — a key present in a later row overwrites earlier, INCLUDING
    # nulls. This reproduces the OPIK-7279 first-attempt NA-trace bug: a finalize
    # re-send that omits name/thread_id sends them as null and clobbers the
    # create. The finalize must therefore replay the full create payload.
    merged: dict = {}
    for t in traces:
        tid = t.get("id")
        key = tid if tid is not None else id(t)
        merged.setdefault(key, {}).update(t)
    merged_traces = list(merged.values())

    print(
        f"journal: {len(rows)} rows | trace-batches={len(traces)} "
        f"merged-traces={len(merged_traces)} spans={len(spans)} updates={len(updates)}"
    )

    errors = []

    # 0. Upsert-only: finalize must NOT go through trace.update() (a PATCH the
    # mock records as an "update" row). Its presence means the plugin regressed
    # to the forbidden post-create mutation — the source of the SDK's
    # "Calling Trace.update() shortly after creation ... may cause data loss"
    # warning (OPIK-7279).
    if updates:
        errors.append(
            f"{len(updates)} trace.update() PATCH(es) — lifecycle must be "
            "upsert-only (same-id re-send), never trace.update()/end()"
        )

    # 1. A root trace exists and, after the finalize re-send merges in, still
    # carries name + thread_id (not clobbered to an NA trace) AND the finished
    # payload (output + end_time). This is the full OPIK-7279 acceptance shape.
    if not merged_traces:
        errors.append("no traces captured")
    else:
        if not any(t.get("name") for t in merged_traces):
            errors.append(
                "no trace has a name after merge (NA trace — finalize re-send "
                "clobbered it; it must replay the full create payload)"
            )
        if not any(t.get("thread_id") for t in merged_traces):
            errors.append(
                "no trace has a thread_id after merge (session grouping lost — "
                "finalize re-send must replay thread_id)"
            )
        if not any(t.get("output") for t in merged_traces):
            errors.append("no trace carries output (finalize re-send missing)")
        if not any(t.get("end_time") for t in merged_traces):
            errors.append("no trace carries end_time (trace not finalized)")

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
    named = next((t.get("name") for t in merged_traces if t.get("name")), None)
    print(f"  trace: {named!r} (finalized via upsert, no trace.update PATCH)")
    print(f"  llm spans: {len(llm)} | tool spans: {len(tool)} | NA: 0")


if __name__ == "__main__":
    main()
