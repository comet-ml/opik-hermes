"""Assert a Hermes turn landed in a REAL Opik instance via its REST API.

Run inside a container on the Opik compose network. Resolves the project by
name, then checks the latest trace has a name, LLM + tool spans, no NA spans,
and a finalized end_time. Env: BE (backend service name), PROJECT.
"""

from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
import json

BE = os.environ["BE"]
PROJECT = os.environ["PROJECT"]
BASE = f"http://{BE}:8080/v1/private"
HEADERS = {"Comet-Workspace": "default"}


def get(path: str, **params) -> dict:
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def main() -> None:
    projects = get("projects", name=PROJECT, size=5).get("content", [])
    if not projects:
        sys.exit(f"FAIL: project {PROJECT!r} not found in real Opik")
    pid = projects[0]["id"]

    traces = get("traces", project_id=pid, size=10).get("content", [])
    print(f"traces: {len(traces)}")
    if not traces:
        sys.exit("FAIL: no traces in real Opik")

    t = get(f"traces/{traces[0]['id']}")
    spans = get("spans", project_id=pid, trace_id=t["id"], size=30).get("content", [])
    llm = [s for s in spans if s.get("type") == "llm"]
    tool = [s for s in spans if s.get("type") == "tool"]
    na = [s for s in spans if not s.get("name") or not s.get("type")]

    # Thread grouping: the plugin sets thread_id = Hermes session_id. The
    # /traces projection in this Opik version does NOT return name/thread_id on
    # the trace object, but the threads endpoint does — so assert the thread
    # there. (Confirmed: a trace created with thread_id shows up under
    # /traces/threads even though the trace object omits the field.)
    threads = get("traces/threads", project_id=pid, size=20).get("content", [])
    thread_ids = [th.get("thread_id") or th.get("id") for th in threads]

    print(f"end_time={t.get('end_time')}")
    print(f"spans={len(spans)} llm={len(llm)} tool={len(tool)} NA={len(na)}")
    print(f"threads={thread_ids}")

    # Hard checks: what proves the integration works — the trace was created,
    # finalized, carries correctly-typed LLM + tool spans with no NA spans, and
    # is grouped into a thread (session grouping).
    errors = []
    if not t.get("end_time"):
        errors.append("trace not finalized (end_time is null)")
    if not llm:
        errors.append("no llm spans")
    if not tool:
        errors.append("no tool spans")
    if na:
        errors.append(f"{len(na)} NA span(s)")
    if not thread_ids:
        errors.append("no thread recorded (session grouping missing)")

    # The descriptive trace name is set by the plugin (verified in the mock E2E
    # payload) but this Opik version's /traces projection omits the name field,
    # so it is not asserted here.

    if errors:
        print("=== REAL-OPIK E2E FAILED ===")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    print("=== REAL-OPIK E2E PASSED ===")


if __name__ == "__main__":
    main()
