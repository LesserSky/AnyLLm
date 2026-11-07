"""A tiny mock LLM provider used by the test suite.

It implements the OpenAI, Anthropic, and Gemini streaming shapes so the whole
client can be exercised without network access or real API keys.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

TEST_KEY = "test-key"


class MockState:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.reject_stream_options = False


def _openai_events() -> list:
    return [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        },
        {
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ]
        },
        {
            "choices": [
                {"index": 0, "delta": {"content": " from mock"}, "finish_reason": None}
            ]
        },
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 1},
            },
        },
        "[DONE]",
    ]


def _anthropic_events() -> list:
    return [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "usage": {"input_tokens": 5, "cache_read_input_tokens": 2},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi "},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "from Claude"},
            },
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


def _gemini_events() -> list:
    return [
        {"candidates": [{"content": {"parts": [{"text": "Gemini "}], "role": "model"}}]},
        {
            "candidates": [
                {"content": {"parts": [{"text": "says hi"}], "role": "model"}}
            ],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
        },
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}], "role": "model"},
                    "finishReason": "STOP",
                }
            ]
        },
    ]


def make_server(state: MockState | None = None):
    state = state or MockState()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):  # silence test output
            pass

        def _json(self, payload, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _unauthorized(self) -> None:
            self._json({"error": {"message": "Incorrect API key provided"}}, 401)

        def _sse(self, events) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for item in events:
                if isinstance(item, tuple):
                    name, payload = item
                    self.wfile.write(f"event: {name}\n".encode("utf-8"))
                else:
                    payload = item
                text = payload if isinstance(payload, str) else json.dumps(payload)
                self.wfile.write(f"data: {text}\n\n".encode("utf-8"))
                self.wfile.flush()

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            state.requests.append({"method": "GET", "path": path, "headers": dict(self.headers)})
            if path == "/openai/v1/models":
                if self.headers.get("Authorization") != f"Bearer {TEST_KEY}":
                    return self._unauthorized()
                self._json(
                    {
                        "object": "list",
                        "data": [{"id": "mock-small"}, {"id": "mock-large"}],
                    }
                )
            elif path == "/anthropic/v1/models":
                if self.headers.get("x-api-key") != TEST_KEY:
                    return self._unauthorized()
                self._json({"data": [{"id": "claude-mock"}], "has_more": False})
            elif path == "/gemini/v1beta/models":
                if self.headers.get("x-goog-api-key") != TEST_KEY:
                    return self._unauthorized()
                self._json(
                    {
                        "models": [
                            {
                                "name": "models/gemini-mock",
                                "supportedGenerationMethods": ["generateContent"],
                            },
                            {
                                "name": "models/embedding-001",
                                "supportedGenerationMethods": ["embedContent"],
                            },
                        ]
                    }
                )
            else:
                self._json({"error": {"message": "not found"}}, 404)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            state.requests.append(
                {
                    "method": "POST",
                    "path": path,
                    "headers": dict(self.headers),
                    "payload": payload,
                }
            )

            if path == "/openai/v1/chat/completions":
                if self.headers.get("Authorization") != f"Bearer {TEST_KEY}":
                    return self._unauthorized()
                if state.reject_stream_options and "stream_options" in payload:
                    return self._json(
                        {"error": {"message": "Unsupported parameter: stream_options"}},
                        400,
                    )
                self._sse(_openai_events())
            elif path == "/bad/v1/chat/completions":
                self._unauthorized()
            elif path == "/anthropic/v1/messages":
                if self.headers.get("x-api-key") != TEST_KEY:
                    return self._unauthorized()
                self._sse(_anthropic_events())
            elif path.startswith("/gemini/v1beta/models/") and path.endswith(
                ":streamGenerateContent"
            ):
                if self.headers.get("x-goog-api-key") != TEST_KEY:
                    return self._unauthorized()
                self._sse(_gemini_events())
            else:
                self._json({"error": {"message": "not found"}}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.state = state  # type: ignore[attr-defined]
    return server


def start_server(state: MockState | None = None):
    server = make_server(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, thread, base
