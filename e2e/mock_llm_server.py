"""Mock OpenAI-compatible LLM for E2E.

Serves **both** wire protocols the Hermes openai-api provider may pick, because
which one it uses is upstream's choice and has changed under us before:

  * **Chat Completions** — POST /v1/chat/completions, consuming
    `data: {"choices":[{"delta":{...}}]}` SSE chunks terminated by `data: [DONE]`.
  * **Responses** — POST /v1/responses, consuming typed SSE events
    (`response.output_item.added/done`, `response.output_text.delta`) and
    terminated by `response.completed`. Hermes calls this its `codex_responses`
    api_mode; the openai-api provider declares that transport, and a rolling
    `nousresearch/hermes-agent:latest` rebuild started honoring it (2026-08-02),
    which is what made a chat-only mock fail with "Codex Responses stream did
    not emit a terminal response".

Both are kept so the suite passes against old and new Hermes images alike.

It also probes GET /v1/models and POST /api/show (Ollama-style).

Deterministic two-step interaction, identical on either protocol:
  call 1  -> a tool call      (Hermes runs a tool  -> tool span)
  call 2+ -> assistant text   (turn completes)

That yields one LLM -> tool -> LLM cycle: a root trace with two LLM spans and
one tool span. No real model, no key, no external network.

Unknown POST routes return 501 rather than a benign 200: a silent 200 makes an
unimplemented protocol look like a hung stream, which is exactly how the
2026-08-02 breakage disguised itself.
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


_RESPONSES_USAGE = {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}


def _responses_tool_events():
    """Emit a function_call item, then response.completed.

    Hermes marks `has_tool_calls` from the `output_item.added` event and
    collects the executable item from `output_item.done`; both are required.
    """
    item = {
        "id": "fc_mock_1",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_mock_1",
        "name": "execute_code",
        "arguments": json.dumps({"code": "print(2**10)"}),
    }
    yield "response.output_item.added", {"output_index": 0, "item": item}
    yield "response.output_item.done", {"output_index": 0, "item": item}
    yield "response.completed", {
        "response": {
            "id": "resp_mock1",
            "status": "completed",
            "output": [item],
            "usage": _RESPONSES_USAGE,
        }
    }


def _responses_final_events():
    """Emit assistant text deltas, then response.completed."""
    text = "The answer is 1024."
    item = {
        "id": "msg_mock_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    yield "response.output_item.added", {
        "output_index": 0,
        "item": {**item, "status": "in_progress", "content": []},
    }
    yield "response.output_text.delta", {"output_index": 0, "delta": text}
    yield "response.output_item.done", {"output_index": 0, "item": item}
    yield "response.completed", {
        "response": {
            "id": "resp_mock2",
            "status": "completed",
            "output": [item],
            "usage": _RESPONSES_USAGE,
        }
    }


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

    def _sse_typed(self, events):
        """SSE for the Responses API: named events carrying a `type` field.

        Unlike Chat Completions there is no `[DONE]` sentinel — the stream ends
        at the terminal `response.completed` event.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for seq, (event_type, payload) in enumerate(events):
            frame = {"type": event_type, "sequence_number": seq, **payload}
            self.wfile.write(f"event: {event_type}\n".encode())
            self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
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

        # Responses API (codex_responses api_mode).
        if self.path.rstrip("/").endswith("/responses"):
            _STATE["calls"] += 1
            events = (
                _responses_tool_events()
                if _STATE["calls"] == 1
                else _responses_final_events()
            )
            wants_stream = b'"stream": true' in body or b'"stream":true' in body
            if wants_stream:
                self._sse_typed(events)
            else:
                # Non-streaming fallback: the terminal event's response object
                # is already the full non-streamed body.
                terminal = list(events)[-1][1]
                self._json({"object": "response", "model": "gpt-5", **terminal["response"]})
            return

        # Chat Completions (the other wire Hermes may pick).
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

        # An unimplemented *inference* route must fail loudly. Returning 200
        # here would let Hermes hang until its retries expire and surface as a
        # vague stream error, hiding the real cause (a protocol we don't serve).
        print(f"UNIMPLEMENTED inference route: {self.path}", flush=True)
        self._json(
            {"error": {"message": f"mock-llm does not implement {self.path}", "type": "not_implemented"}},
            code=501,
        )


if __name__ == "__main__":
    print(f"mock-llm listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
