"""Mock Opik backend for E2E.

Stands in for a real Opik server: accepts whatever the opik SDK POSTs and
appends each create/update call to a JSON-lines journal. The E2E assertion
script then reads the journal to verify the plugin produced a Hermes-turn
trace with LLM + tool spans — without any real Opik instance.

Only the handful of endpoints the SDK touches are implemented; everything
else returns 200/204 so the SDK's background flush never errors and never
retries forever.
"""

from __future__ import annotations

import gzip
import json
import os
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

JOURNAL = os.environ.get("MOCK_OPIK_JOURNAL", "/journal/opik-journal.jsonl")
PORT = int(os.environ.get("MOCK_OPIK_PORT", "5173"))


def _record(kind: str, payload: object) -> None:
    os.makedirs(os.path.dirname(JOURNAL) or ".", exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "payload": payload}) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if not length:
            return None
        raw = self.rfile.read(length)
        # The Opik SDK gzip-compresses batch payloads; decompress before parsing.
        enc = (self.headers.get("Content-Encoding") or "").lower()
        if "gzip" in enc or raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except Exception:
                try:
                    raw = zlib.decompress(raw, zlib.MAX_WBITS | 16)
                except Exception:
                    pass
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw": raw.decode("utf-8", "replace")}

    def _ok(self, code: int = 204, body: bytes = b""):
        self.send_response(code)
        if body:
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        # is-alive / project lookups → benign empty answers.
        if "is-alive" in self.path or "ping" in self.path:
            self._ok(200, b'{"status":"ok"}')
            return
        self._ok(200, b'{"content":[],"total":0}')

    def do_POST(self):
        payload = self._read_json()
        path = self.path
        # The SDK batches: /v1/private/traces/batch, /spans/batch, etc.
        if "/traces/batch" in path or path.endswith("/traces"):
            _record("traces", payload)
        elif "/spans/batch" in path or path.endswith("/spans"):
            _record("spans", payload)
        elif "/projects" in path:
            # project create/resolve — echo an id so the SDK is satisfied.
            self._ok(201, b'{"id":"00000000-0000-0000-0000-000000000000"}')
            return
        else:
            _record("other", {"path": path, "payload": payload})
        self._ok(204)

    def do_PUT(self):
        self.do_POST()

    def do_PATCH(self):
        payload = self._read_json()
        _record("update", {"path": self.path, "payload": payload})
        self._ok(204)


if __name__ == "__main__":
    # Truncate any prior journal so each run starts clean.
    os.makedirs(os.path.dirname(JOURNAL) or ".", exist_ok=True)
    open(JOURNAL, "w").close()
    print(f"mock-opik listening on :{PORT}, journal -> {JOURNAL}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
