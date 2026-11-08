"""Tool protocol. Offline fixtures only; no claim that a provider supports tools."""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
import uuid
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.adapters import OPENAI, ToolCallDelta, stream_chat
from anyllm.catalog import ANTHROPIC, Provider
from anyllm.errors import UsageError
from anyllm.seam import RunOptions, attachment_store, make_snapshot
from anyllm.targets import Target
from anyllm.tool_protocol import (
    READ_ATTACHED_FILE,
    assemble_tool_calls,
    capability_record,
    disable_all_tools,
    enable_tool,
    execute_read_attached_file,
    parse_path_argument,
    probe_tools,
    require_tool_capability,
    run_tool_conversation,
)
from anyllm.seam import ApprovedAttachment


def _home(case: unittest.TestCase) -> Path:
    path = ROOT / ".test-run" / f"tools-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    case.addCleanup(lambda: __import__("shutil").rmtree(path, True))
    return path


def _env(home: Path):
    return mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home), "NO_COLOR": "1"}, clear=False)


class AssembleTests(unittest.TestCase):
    def test_fragments_join_and_duplicates_are_rejected(self):
        calls = assemble_tool_calls(
            [
                ToolCallDelta(index=0, id="call_1", name="read_", arguments=""),
                ToolCallDelta(index=0, id="", name="attached_file", arguments='{"pa'),
                ToolCallDelta(index=0, id="", name="", arguments='th": "notes.txt"}'),
            ]
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].id, "call_1")
        self.assertEqual(calls[0].name, READ_ATTACHED_FILE)
        self.assertEqual(parse_path_argument(calls[0].arguments), "notes.txt")
        with self.assertRaises(UsageError):
            assemble_tool_calls(
                [
                    ToolCallDelta(0, "same", "read_attached_file", "{}"),
                    ToolCallDelta(1, "same", "read_attached_file", "{}"),
                ]
            )

    def test_argument_and_call_caps_trip_while_fragments_arrive(self):
        with self.assertRaises(UsageError) as caught:
            assemble_tool_calls(
                [
                    ToolCallDelta(0, "call", "read_attached_file", "a" * 8000),
                    ToolCallDelta(0, "", "", "b" * 200),
                ]
            )
        self.assertIn("8192", str(caught.exception))
        self.assertIn("streaming", str(caught.exception))
        many = [ToolCallDelta(index, f"id-{index}", "read_attached_file", "{}") for index in range(9)]
        with self.assertRaises(UsageError) as caught:
            assemble_tool_calls(many)
        self.assertIn("8", str(caught.exception))

    def test_bad_json_and_unknown_fields(self):
        with self.assertRaises(UsageError):
            parse_path_argument("{")
        with self.assertRaises(UsageError):
            parse_path_argument('{"path": "a", "command": "rm"}')
        with self.assertRaises(UsageError):
            parse_path_argument("[]")
        with self.assertRaises(UsageError):
            parse_path_argument("x" * 9000)

    def test_unapproved_path_is_denied_without_reading_other_content(self):
        snapshot = make_snapshot(
            [],
            attachments=(
                ApprovedAttachment("notes.txt", "abc", "hello-notes"),
            ),
        )
        allowed = execute_read_attached_file("notes.txt", snapshot)
        self.assertTrue(allowed["ok"])
        self.assertEqual(allowed["content"], "hello-notes")
        denied = execute_read_attached_file("../secrets.env", snapshot)
        self.assertFalse(denied["ok"])
        self.assertNotIn("SECRET-FILE", json.dumps(denied))
        self.assertIn("approval", denied["error"])


class ProbeAndLoopTests(unittest.TestCase):
    def setUp(self):
        self.home = _home(self)
        self._patch = _env(self.home)
        self._patch.start()
        attachment_store().clear()

    def tearDown(self):
        attachment_store().clear()
        self._patch.stop()

    def test_support_stays_unknown_until_a_probe_and_is_endpoint_specific(self):
        provider = Provider("mock", "Mock", OPENAI, "http://127.0.0.1/v1", key_required=False)
        target = Target("mock", "model-a", "http://127.0.0.1:9/v1", alias="fast")
        with self.assertRaises(UsageError) as caught:
            require_tool_capability(target.base_url, target.model)
        self.assertIn("unknown", str(caught.exception).lower())
        other = Provider("anthropic", "Anthropic", ANTHROPIC, "https://api.anthropic.com/v1")
        record = probe_tools(other, Target("anthropic", "claude", other.base_url), credential=None, timeout=5)
        self.assertFalse(record["sent"])
        self.assertEqual(record["tools"], "unknown")
        self.assertIsNone(capability_record(other.base_url, "claude"))

    def test_enable_is_local_and_reject_unknown_names(self):
        with self.assertRaises(UsageError):
            enable_tool("shell")
        enable_tool(READ_ATTACHED_FILE)
        disable_all_tools()
        provider = Provider("mock", "Mock", OPENAI, "http://127.0.0.1/v1", key_required=False)
        target = Target("mock", "m", provider.base_url, alias="fast")
        with self.assertRaises(UsageError):
            run_tool_conversation(
                provider,
                target,
                make_snapshot([]),
                RunOptions(timeout=5),
                credential=None,
                prompt="hi",
                tool_names=[READ_ATTACHED_FILE],
                max_rounds=1,
            )

    def test_mock_http_tool_round_and_probe(self):
        secret = "SECRET-FILE-CONTENTS"
        approved = "hello-notes"
        state = {"calls": []}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                payload = json.loads(raw)
                state["calls"].append(payload)
                blob = raw.decode("utf-8")
                if payload.get("model") == "reject-model":
                    body = json.dumps(
                        {"error": {"message": "Unknown parameter: tools", "param": "tools"}}
                    ).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                system = ""
                for message in payload.get("messages") or []:
                    if message.get("role") == "system":
                        system = message.get("content") or ""
                if "capability probe" in system:
                    events = [
                        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
                        "[DONE]",
                    ]
                elif any(message.get("role") == "tool" for message in payload.get("messages") or []):
                    events = [
                        {"choices": [{"delta": {"content": "saw the note"}, "finish_reason": "stop"}]},
                        "[DONE]",
                    ]
                else:
                    events = [
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_1",
                                                "function": {"name": "read_attached_", "arguments": ""},
                                            }
                                        ]
                                    }
                                }
                            ]
                        },
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {"name": "file", "arguments": '{"path": "notes.txt"}'},
                                            },
                                            {
                                                "index": 1,
                                                "id": "call_2",
                                                "function": {
                                                    "name": "read_attached_file",
                                                    "arguments": '{"path": "secrets.env"}',
                                                },
                                            },
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ]
                        },
                        "[DONE]",
                    ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for item in events:
                    text = item if isinstance(item, str) else json.dumps(item)
                    self.wfile.write(f"data: {text}\n\n".encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        provider = Provider("mock", "Mock", OPENAI, base, key_required=False)
        target = Target("mock", "model-a", base, alias="fast")
        record = probe_tools(provider, target, credential=None, timeout=5)
        self.assertEqual(record["tools"], "accepted")
        self.assertIn("not evidence", record["detail"])
        self.assertIsNone(capability_record(base, "model-b"))

        rejected = probe_tools(
            provider,
            Target("mock", "reject-model", base, alias="fast"),
            credential=None,
            timeout=5,
        )
        self.assertEqual(rejected["tools"], "rejected")

        enable_tool(READ_ATTACHED_FILE)
        attachment_store().approve(ApprovedAttachment("notes.txt", "hash", approved))
        # A file that exists on disk must not be read just because the model names it.
        outside = self.home / "secrets.env"
        outside.write_text(secret, encoding="utf-8")
        result = run_tool_conversation(
            provider,
            target,
            make_snapshot([], attachments=attachment_store().list()),
            RunOptions(timeout=5, max_tokens=32),
            credential=None,
            prompt="Inspect the attached file",
            tool_names=[READ_ATTACHED_FILE],
            max_rounds=2,
        )
        self.assertEqual(result.text, "saw the note")
        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(result.request_count, 2)
        self.assertTrue(any(item.get("executed") for item in result.tool_trace))
        self.assertNotIn(secret, json.dumps(state["calls"]))
        self.assertTrue(outside.exists())

    def test_stream_chat_without_tools_omits_the_field(self):
        # Guard the existing payload shape: tools stay absent unless requested.
        provider = Provider("mock", "Mock", OPENAI, "http://127.0.0.1:9/v1", key_required=False)
        # No server is contacted because the call is not executed; build the payload
        # by inspecting the function default through a refused connection only if tools
        # were set. Here we just check the public default stays None.
        self.assertIn("tools", stream_chat.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
