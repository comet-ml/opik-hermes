"""Mock OpenAI-compatible LLM for E2E.

The Hermes openai-api provider (as configured here) calls the **Chat
Completions** API with streaming: POST /v1/chat/completions, and consumes
`data: {"choices":[{"delta":{...}}]}` SSE chunks terminated by `data: [DONE]`.
It also probes GET /v1/models and POST /api/show (Ollama-style).

Deterministic two-step interaction:
  call 1  -> a tool_call delta  (Hermes runs a tool  -> tool span)
  call 2+ -> a content delta     (turn completes)

That yields one LLM -> tool -> LLM cycle: a root trace with two LLM spans and
one tool span. No real model, no key, no external network.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("MOCK_LLM_PORT", "18790"))
_STATE = {"calls": 0}
_USAGE = {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}


def _tool_call_chunks():
    """Stream a single tool_call, then finish_reason=tool_calls."""
    base = {"id": "chatcmpl-mock1", "object": "chat.completion.chunk", "model": "gpt-5"}
    yield {
        **base,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_mock_1",
                            "type": "function",
                            "function": {
                                "name": "execute_code",
                                "arguments": json.dumps({"code": "print(2**10)"}),
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    }
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": _USAGE}


def _final_chunks():
    """Stream the final assistant content, then finish_reason=stop."""
    base = {"id": "chatcmpl-mock2", "object": "chat.completion.chunk", "model": "gpt-5"}
    yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "The answer is 1024."}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": _USAGE}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        # /v1/models and health probes
        self._json({"object": "list", "data": [{"id": "gpt-5", "object": "model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        print(f"POST {self.path} len={length}", flush=True)

        # Ollama-style model probe.
        if self.path.rstrip("/").endswith("/api/show"):
            self._json({"model": "gpt-5", "details": {"family": "gpt"}})
            return

        # Chat Completions (the path Hermes uses here).
        if "chat/completions" in self.path:
            _STATE["calls"] += 1
            chunks = _tool_call_chunks() if _STATE["calls"] == 1 else _final_chunks()
            wants_stream = b'"stream": true' in body or b'"stream":true' in body
            if wants_stream:
                self._sse(chunks)
            else:
                # Non-streaming fallback: assemble one completion object.
                msgs = list(chunks)
                self._json(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "model": "gpt-5",
                        "choices": [{"index": 0, "message": msgs[0]["choices"][0]["delta"], "finish_reason": msgs[-1]["choices"][0]["finish_reason"]}],
                        "usage": _USAGE,
                    }
                )
            return

        # Anything else → benign 200.
        self._json({"ok": True})


if __name__ == "__main__":
    print(f"mock-llm listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
