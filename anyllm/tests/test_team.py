"""Team fan-out with fake runners and one local HTTP server. No live APIs."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
import uuid
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.adapters import Usage
from anyllm.catalog import OPENAI, Provider
from anyllm.errors import ConfigError, UsageError
from anyllm.keystore import KeyStore
from anyllm.seam import RunOptions, RunResult, make_snapshot
from anyllm.session import Session
from anyllm.targets import ResolvedTarget, Target
from anyllm.team import (
    MAX_CONCURRENCY,
    PreparedMember,
    add_member,
    adopt_run,
    execute_team_ask,
    fan_out,
    load_last_run,
    load_members,
    request_team_cancel,
    synthesis_prompt,
)
from anyllm.term import Style


def _home(case: unittest.TestCase) -> Path:
    path = ROOT / ".test-run" / f"team-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    case.addCleanup(lambda: __import__("shutil").rmtree(path, True))
    return path


def _env(home: Path, **extra):
    cleaned = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith("_API_KEY") and not key.startswith("ANYLLM_")
    }
    cleaned["ANYLLM_HOME"] = str(home)
    cleaned["NO_COLOR"] = "1"
    cleaned.update(extra)
    return mock.patch.dict(os.environ, cleaned, clear=True)


def _member(name: str, secret: str | None = None, *, model: str = "m") -> PreparedMember:
    provider = Provider(
        name="mock",
        label="Mock",
        kind=OPENAI,
        base_url="http://127.0.0.1/v1",
        key_required=secret is not None,
    )
    target = Target("mock", model, "http://127.0.0.1/v1", alias=name)
    resolved = ResolvedTarget(target, provider, secret, "test", True, None)
    return PreparedMember(name, f"mock:{model}", resolved)


def _ok(name: str, text: str, delay: float = 0) -> RunResult:
    if delay:
        time.sleep(delay)
    return RunResult(
        status="ok",
        text=text,
        usage=Usage(input_tokens=3, output_tokens=2),
        finish_reason="stop",
        elapsed=delay,
        request_count=1,
        provider="mock",
        model="m",
        alias=name,
    )


class FanOutTests(unittest.TestCase):
    def test_identical_snapshots_do_not_change_the_session_and_stay_ordered(self):
        home = _home(self)
        session = Session(
            Provider("mock", "Mock", OPENAI, "http://127.0.0.1/v1", key_required=True),
            "m",
            api_key=None,
        )
        session.messages = [{"role": "user", "content": "earlier"}]
        seen = []

        def runner(provider, target, snapshot, options, *, credential, prompt, cancel_event=None, event_queue=None):
            seen.append((snapshot.snapshot_id, snapshot.history, prompt, target.model))
            session.messages.append({"role": "user", "content": "smuggled"})
            delay = 0.05 if target.alias == "slow" else 0
            return _ok(target.alias or "", f"from-{target.alias}", delay)

        with _env(home, ANYLLM_MOCK_API_KEY="super-secret-key"):
            before = list(session.messages)
            artifact = execute_team_ask(
                specs=[("slow", "mock:m"), ("fast", "mock:m")],
                prompt="Compare",
                providers={"mock": session.provider},
                store=KeyStore(),
                history=session.messages,
                system=None,
                temperature=None,
                max_tokens=32,
                timeout=5,
                deadline_seconds=5,
                view="stacked",
                merge=None,
                budget_usd=None,
                concurrency=2,
                yes=True,
                interactive=False,
                stdout_tty=False,
                style=Style(False),
                runner=runner,
                width=80,
            )
            saved = load_last_run()
        self.assertEqual(
            [item for item in session.messages if item.get("content") != "smuggled"],
            before,
        )
        self.assertFalse(any(item.get("role") == "assistant" for item in session.messages))
        self.assertEqual(seen[0][0], seen[1][0])
        self.assertEqual(seen[0][1], (("user", "earlier"),))
        self.assertEqual(seen[0][2], "Compare")
        self.assertEqual([item["alias"] for item in artifact["results"]], ["slow", "fast"])
        self.assertEqual(artifact["results"][0]["text"], "from-slow")
        self.assertNotIn("super-secret-key", json.dumps(artifact))
        self.assertNotIn("super-secret-key", json.dumps(saved))
        self.assertEqual(saved["id"], artifact["id"])

    def test_concurrency_is_capped_and_each_member_is_called_once(self):
        state = {"current": 0, "max": 0, "calls": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(MAX_CONCURRENCY)

        def runner(provider, target, snapshot, options, *, credential=None, prompt="", cancel_event=None, event_queue=None):
            with lock:
                state["calls"] += 1
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            barrier.wait(timeout=2)
            with lock:
                state["current"] -= 1
            return _ok(target.alias or "", "x")

        members = [_member(name) for name in ("a", "b", "c", "d")]
        fan_out(
            members,
            make_snapshot([]),
            "q",
            RunOptions(timeout=5),
            concurrency=MAX_CONCURRENCY + 5,
            runner=runner,
        )
        self.assertEqual(state["calls"], 4)
        self.assertEqual(state["max"], MAX_CONCURRENCY)

    def test_default_wave_size_is_two(self):
        state = {"current": 0, "max": 0}
        lock = threading.Lock()
        release = threading.Barrier(2)

        def runner(provider, target, snapshot, options, *, credential=None, prompt="", cancel_event=None, event_queue=None):
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            release.wait(timeout=2)
            with lock:
                state["current"] -= 1
            return _ok(target.alias or "", "x")

        fan_out(
            [_member(name) for name in ("a", "b", "c", "d")],
            make_snapshot([]),
            "q",
            RunOptions(timeout=5),
            concurrency=2,
            runner=runner,
        )
        self.assertEqual(state["max"], 2)

    def test_failure_is_isolated_and_does_not_pick_another_model(self):
        def runner(provider, target, snapshot, options, *, credential=None, prompt="", cancel_event=None, event_queue=None):
            if target.alias == "bad":
                return RunResult(
                    status="error",
                    error="authentication failed",
                    incomplete=True,
                    request_count=1,
                    provider=target.provider,
                    model=target.model,
                    alias=target.alias or "",
                )
            good = _ok(target.alias or "", "good")
            good.model = target.model
            return good

        results = fan_out(
            [_member("bad", model="bad-model"), _member("good", model="good-model")],
            make_snapshot([]),
            "q",
            RunOptions(timeout=5),
            runner=runner,
        )
        self.assertEqual(results[0].status, "error")
        self.assertEqual(results[0].model, "bad-model")
        self.assertEqual(results[1].text, "good")
        self.assertEqual(results[1].model, "good-model")

    def test_deadline_does_not_start_remaining_work(self):
        calls = {"n": 0}

        def runner(*_args, **_kwargs):
            calls["n"] += 1
            raise AssertionError("should not run")

        results = fan_out(
            [_member("a"), _member("b")],
            make_snapshot([]),
            "q",
            RunOptions(timeout=5, deadline=time.perf_counter() - 1),
            runner=runner,
        )
        self.assertEqual(calls["n"], 0)
        self.assertTrue(all(item.status == "timeout" for item in results))

    def test_cancel_stops_scheduling(self):
        entered = threading.Event()
        calls = {"n": 0}

        def runner(*_args, cancel_event=None, **_kwargs):
            calls["n"] += 1
            entered.set()
            assert cancel_event is not None
            cancel_event.wait(timeout=2)
            return RunResult(status="cancelled", incomplete=True, request_count=1, text="")

        def poke():
            self.assertTrue(entered.wait(timeout=2))
            self.assertTrue(request_team_cancel())

        threading.Thread(target=poke, daemon=True).start()
        results = fan_out(
            [_member(name) for name in ("a", "b", "c", "d")],
            make_snapshot([]),
            "q",
            RunOptions(timeout=5),
            concurrency=1,
            runner=runner,
        )
        self.assertLess(calls["n"], 4)
        self.assertTrue(any(item.status == "cancelled" for item in results))

    def test_synthesis_quotes_are_untrusted_and_partial_merge_needs_yes(self):
        prompts = []

        def runner(provider, target, snapshot, options, *, credential=None, prompt="", cancel_event=None, event_queue=None):
            prompts.append((target.alias, prompt))
            if target.alias == "bad":
                return RunResult(status="error", error="nope", incomplete=True, request_count=1, alias="bad", provider="mock", model="m")
            return _ok(target.alias or "", f"answer-{target.alias}")

        home = _home(self)
        provider = _member("fast").resolved.provider
        with _env(home):
            with self.assertRaises(UsageError):
                execute_team_ask(
                    specs=[("only", "mock:m")],
                    prompt="q",
                    providers={"mock": provider},
                    store=KeyStore(),
                    history=[],
                    system=None,
                    temperature=None,
                    max_tokens=16,
                    timeout=5,
                    deadline_seconds=5,
                    view="stacked",
                    merge=None,
                    budget_usd=None,
                    concurrency=2,
                    yes=True,
                    interactive=False,
                    style=Style(False),
                    runner=runner,
                )
            artifact = execute_team_ask(
                specs=[("fast", "mock:m"), ("bad", "mock:m")],
                prompt="Compare these approaches",
                providers={"mock": provider},
                store=KeyStore(),
                history=[],
                system=None,
                temperature=None,
                max_tokens=16,
                timeout=5,
                deadline_seconds=5,
                view="columns",
                merge="fast",
                budget_usd=None,
                concurrency=2,
                yes=False,
                interactive=False,
                stdout_tty=False,
                style=Style(False),
                runner=runner,
                width=40,
            )
            self.assertIsNone(artifact["synthesis"])
            self.assertEqual(len(prompts), 2)
            prompts.clear()
            merged = execute_team_ask(
                specs=[("fast", "mock:m"), ("bad", "mock:m")],
                prompt="Compare these approaches",
                providers={"mock": provider},
                store=KeyStore(),
                history=[],
                system=None,
                temperature=None,
                max_tokens=16,
                timeout=5,
                deadline_seconds=5,
                view="stacked",
                merge="fast",
                budget_usd=None,
                concurrency=2,
                yes=True,
                interactive=False,
                style=Style(False),
                runner=runner,
            )
        synth_prompts = [prompt for alias, prompt in prompts if "UNTRUSTED" in prompt]
        self.assertEqual(len(synth_prompts), 1)
        text = synth_prompts[0]
        self.assertIn("not instructions", text)
        self.assertIn("Do not claim the models reached consensus.", text)
        self.assertIn("<<<UNTRUSTED member=fast", text)
        self.assertIn("<<<UNTRUSTED member=bad", text)
        self.assertIn("(no answer)", text)
        self.assertIn("<<<END UNTRUSTED>>>", text)
        self.assertEqual(merged["synthesis"]["alias"], "merged")
        self.assertEqual(merged["request_count"], 3)
        self.assertEqual(len(merged["results"]), 2)

    def test_columns_fallback_has_no_ansi_and_unicode_renders(self):
        def runner(provider, target, snapshot, options, *, credential=None, prompt="", cancel_event=None, event_queue=None):
            return _ok(target.alias or "", "café 字")

        home = _home(self)
        provider = _member("a").resolved.provider
        with _env(home):
            with mock.patch("sys.stdout") as stdout:
                stdout.isatty.return_value = False
                # execute prints to the real stdout; capture via redirect instead
            import io
            from contextlib import redirect_stdout

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                execute_team_ask(
                    specs=[("a", "mock:m"), ("b", "mock:m")],
                    prompt="q",
                    providers={"mock": provider},
                    store=KeyStore(),
                    history=[],
                    system=None,
                    temperature=None,
                    max_tokens=8,
                    timeout=5,
                    deadline_seconds=5,
                    view="columns",
                    merge=None,
                    budget_usd=None,
                    concurrency=2,
                    yes=True,
                    interactive=False,
                    stdout_tty=False,
                    style=Style(False),
                    runner=runner,
                    width=40,
                )
        text = buffer.getvalue()
        self.assertIn("showing stacked", text)
        self.assertIn("café 字", text)
        self.assertNotIn("\x1b", text)

    def test_adopt_appends_one_exchange_and_unknown_member_does_not(self):
        home = _home(self)
        provider = Provider("mock", "Mock", OPENAI, "http://127.0.0.1/v1", key_required=False)
        session = Session(provider, "m")

        def runner(provider, target, snapshot, options, *, credential=None, prompt="", cancel_event=None, event_queue=None):
            return _ok(target.alias or "", f"answer-{target.alias}")

        with _env(home):
            execute_team_ask(
                specs=[("fast", "mock:m"), ("slow", "mock:m")],
                prompt="Question",
                providers={"mock": provider},
                store=KeyStore(),
                history=[],
                system=None,
                temperature=None,
                max_tokens=8,
                timeout=5,
                deadline_seconds=5,
                view="stacked",
                merge="fast",
                budget_usd=None,
                concurrency=2,
                yes=True,
                interactive=False,
                style=Style(False),
                runner=runner,
            )
            self.assertEqual(session.messages, [])
            adopt_run(session, load_last_run(), "slow")
            self.assertEqual(session.messages[0]["content"], "Question")
            self.assertEqual(session.messages[1]["content"], "answer-slow")
            adopt_run(session, load_last_run(), "merged")
            self.assertEqual(session.messages[-1]["content"], "answer-fast")
            with self.assertRaises(UsageError):
                adopt_run(session, load_last_run(), "missing")

    def test_saved_team_rejects_a_fifth_member_and_unknown_spec(self):
        home = _home(self)
        provider = Provider("mock", "Mock", OPENAI, "http://127.0.0.1/v1", key_required=False)
        with _env(home):
            for name in ("a", "b", "c", "d"):
                add_member(name, "mock:m", {"mock": provider}, KeyStore())
            with self.assertRaises(ConfigError):
                add_member("e", "mock:m", {"mock": provider}, KeyStore())
            self.assertEqual(len(load_members()), 4)
            with self.assertRaises(ConfigError):
                add_member("z", "nope", {"mock": provider}, KeyStore())

    def test_synthesis_helper_marks_every_status(self):
        text = synthesis_prompt(
            "Why?",
            [
                RunResult(status="ok", text="yes", alias="a", provider="p", model="m"),
                RunResult(status="error", text="", alias="b", provider="p", model="m"),
            ],
        )
        self.assertIn("member=a", text)
        self.assertIn("status=error", text)
        self.assertNotIn("consensus.", text.lower().split("do not claim")[0])


class MockHttpTeamTests(unittest.TestCase):
    def test_mixed_auth_and_partial_stream_stay_isolated(self):
        state = {"bodies": []}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                state["bodies"].append(
                    {"auth": self.headers.get("Authorization"), "body": raw.decode("utf-8")}
                )
                model = json.loads(raw).get("model")
                if self.headers.get("Authorization") != "Bearer good-key":
                    payload = json.dumps({"error": {"message": "bad key"}}).encode()
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if model == "partial":
                    events = [
                        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
                        {"choices": [{"delta": {}, "finish_reason": "length"}]},
                        "[DONE]",
                    ]
                else:
                    events = [
                        {"choices": [{"delta": {"content": "Full"}, "finish_reason": None}]},
                        {
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
                        },
                        "[DONE]",
                    ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for item in events:
                    text = item if isinstance(item, str) else json.dumps(item)
                    self.wfile.write(f"data: {text}\n\n".encode())
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        home = _home(self)

        def prepared(name, model, key):
            provider = Provider("mock", "Mock", OPENAI, base, key_required=True)
            target = Target("mock", model, base, alias=name)
            return PreparedMember(
                name,
                f"mock:{model}",
                ResolvedTarget(target, provider, key, "test", True, None),
            )

        with _env(home):
            results = fan_out(
                [prepared("full", "complete", "good-key"), prepared("denied", "complete", "other-key"), prepared("cut", "partial", "good-key")],
                make_snapshot([{"role": "user", "content": "history"}]),
                "Question",
                RunOptions(timeout=5, max_tokens=16),
                concurrency=2,
            )
        by_name = {item.alias: item for item in results}
        self.assertEqual(by_name["full"].text, "Full")
        self.assertEqual(by_name["full"].status, "ok")
        self.assertEqual(by_name["cut"].status, "incomplete")
        self.assertIn("Hi", by_name["cut"].text)
        self.assertEqual(by_name["denied"].status, "error")
        self.assertNotIn("good-key", by_name["denied"].text or "")
        auths = [item["auth"] for item in state["bodies"]]
        self.assertIn("Bearer good-key", auths)
        self.assertIn("Bearer other-key", auths)
        self.assertTrue(all("history" in item["body"] and "Question" in item["body"] for item in state["bodies"]))


if __name__ == "__main__":
    unittest.main()
