"""Local SSE server. A pause longer than 5s must not poison the stream."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.adapters import stream_chat
from anyllm.catalog import BUILTINS
from anyllm.errors import NetworkError
from anyllm.seam import RunOptions, make_snapshot, run_request
from anyllm.targets import Target

PAUSE = 6.5


def _event(text: str, *, finish: str | None = None) -> bytes:
    choice = {"delta": {"content": text}}
    if finish:
        choice["finish_reason"] = finish
    return f"data: {json.dumps({'choices': [choice]})}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        return

    def _chunk(self, payload: bytes) -> None:
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        mode = parse_qs(urlsplit(self.path).query).get("mode", ["pause"])[0]
        with self.server.lock:
            self.server.posts += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            if mode == "split":
                # One SSE event, cut in the middle of the text, then the rest.
                self._chunk(b'data: {"choices":[{"delta":{"content":"Hel')
                time.sleep(PAUSE)
                self._chunk(b'lo"},"finish_reason":"stop"}]}\n\n')
            else:
                self._chunk(_event("Hello"))
                time.sleep(PAUSE)
                self._chunk(_event(" there", finish="stop"))
            self._chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            return


class StreamDeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.server.lock = threading.Lock()
        cls.server.posts = 0
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _texts(self, mode: str, *, deadline: float, cancel_event=None) -> str:
        parts: list[str] = []
        for chunk in stream_chat(
            BUILTINS["openai"],
            model="mock",
            messages=[{"role": "user", "content": "hi"}],
            key="sentinel-key",
            base_url=f"{self.base}?mode={mode}",
            timeout=30,
            deadline=deadline,
            cancel_event=cancel_event,
        ):
            if chunk.text:
                parts.append(chunk.text)
        return "".join(parts)

    def test_pause_over_five_seconds_keeps_both_sides_of_the_event(self):
        before = self.server.posts
        started = time.perf_counter()
        text = self._texts("split", deadline=started + 15)
        elapsed = time.perf_counter() - started
        self.assertEqual(text, "Hello")
        self.assertGreater(elapsed, 6)
        self.assertLess(elapsed, 15)
        self.assertEqual(self.server.posts, before + 1)

    def test_complete_event_then_pause_still_finishes_before_the_deadline(self):
        before = self.server.posts
        text = self._texts("pause", deadline=time.perf_counter() + 15)
        self.assertEqual(text, "Hello there")
        self.assertEqual(self.server.posts, before + 1)

    def test_overall_deadline_keeps_text_from_before_the_pause(self):
        before = self.server.posts
        started = time.perf_counter()
        with self.assertRaises(NetworkError) as caught:
            self._texts("pause", deadline=started + 2.5)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 5)
        message = str(caught.exception).lower()
        self.assertIn("deadline", message)
        self.assertIn("not cancelled", message)
        self.assertNotIn("cannot read from timed out object", message)
        result = run_request(
            BUILTINS["openai"],
            Target("openai", "mock", f"{self.base}?mode=pause"),
            make_snapshot([]),
            RunOptions(timeout=30, deadline=time.perf_counter() + 2.5),
            credential="sentinel-key",
            prompt="hi",
        )
        self.assertEqual(result.status, "timeout")
        self.assertIn("Hello", result.text)
        self.assertNotIn("there", result.text)
        self.assertEqual(result.request_count, 1)
        self.assertIn("not cancelled", (result.error or "").lower())
        self.assertEqual(self.server.posts, before + 2)

    def test_cancel_during_pause_keeps_received_text_and_does_not_retry(self):
        before = self.server.posts
        cancel = threading.Event()

        def poke() -> None:
            time.sleep(0.4)
            cancel.set()

        threading.Thread(target=poke, daemon=True).start()
        started = time.perf_counter()
        parts: list[str] = []
        with self.assertRaises(NetworkError) as caught:
            for chunk in stream_chat(
                BUILTINS["openai"],
                model="mock",
                messages=[{"role": "user", "content": "hi"}],
                key="sentinel-key",
                base_url=f"{self.base}?mode=pause",
                timeout=30,
                deadline=started + 15,
                cancel_event=cancel,
            ):
                if chunk.text:
                    parts.append(chunk.text)
        text = "".join(parts)
        self.assertLess(time.perf_counter() - started, 5)
        self.assertIn("Hello", text)
        self.assertNotIn("there", text)
        self.assertIn("not cancelled", str(caught.exception).lower())
        self.assertEqual(self.server.posts, before + 1)


if __name__ == "__main__":
    unittest.main()
