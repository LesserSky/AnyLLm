"""Offline end-to-end tests for AnyLLM."""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib import error

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for entry in (str(ROOT), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from anyllm.adapters import Chunk, Usage, list_models, stream_chat
from anyllm.catalog import ANTHROPIC, BUILTINS, GEMINI, OPENAI, Provider, load_providers
from anyllm.cli import _interactive_setup, _session_from_saved, cmd_config_set
from anyllm.config import load_config, validate_max_tokens, validate_temperature
from anyllm.errors import APIError, AuthError, ConfigError, NetworkError, UsageError
from anyllm.keystore import KeyStore, accepts_generic_api_key, resolve_key
from anyllm.repl import Repl
from anyllm.session import (
    INTERRUPTED_MARK,
    Session,
    load_session_data,
    save_session,
    session_base_url,
)
from anyllm.term import Style
from anyllm.transport import iter_sse, join_url, open_stream, request_json

from mock_api import TEST_KEY, MockState, start_server


def _read_stream(generator):
    chunks = list(generator)
    text = "".join(chunk.text for chunk in chunks)
    reasoning = "".join(chunk.reasoning for chunk in chunks)
    usage = Usage()
    for chunk in chunks:
        if chunk.usage is not None:
            usage.update(chunk.usage)
    finish = next((chunk.finish_reason for chunk in chunks if chunk.finish_reason), None)
    return text, reasoning, usage, finish, chunks


def new_temp_dir(case: unittest.TestCase) -> Path:
    """A writable test folder.

    ``tempfile`` creates 0700 directories, which the DSH Windows sandbox
    rejects; a normally-created directory under the workspace works.
    """
    base = ROOT / ".test-run"
    base.mkdir(exist_ok=True)
    path = base / f"home-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    case.addCleanup(shutil.rmtree, path, True)
    return path


def tearDownModule() -> None:
    shutil.rmtree(ROOT / ".test-run", ignore_errors=True)


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = MockState()
        cls.server, cls.thread, cls.base = start_server(cls.state)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def temp_home(self) -> Path:
        return new_temp_dir(self)

    def tearDown(self) -> None:
        state = self.state
        state.reject_stream_options = False
        state.stream_options_status = 400
        state.stream_options_body = None
        state.openai_failures.clear()
        state.get_failures.clear()
        state.reset_midstream = False

    def _posts_since(self, start: int) -> list[dict]:
        return [
            item
            for item in self.state.requests[start:]
            if item.get("method") == "POST"
        ]

    def run_cli(self, args, home: Path, *, input_text=None, extra_env=None):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("ANYLLM_") and not key.endswith("_API_KEY")
        }
        env.update(
            {
                "ANYLLM_HOME": str(home),
                "NO_COLOR": "1",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, "-m", "anyllm", *args],
            cwd=str(ROOT),
            env=env,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=90,
        )


class AdapterTests(ServerTestCase):
    def test_openai_stream(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )
        text, _reasoning, usage, finish, _chunks = _read_stream(
            stream_chat(
                provider,
                model="mock",
                messages=[{"role": "user", "content": "hi"}],
                system="be brief",
                key=TEST_KEY,
            )
        )
        self.assertEqual(text, "Hello from mock")
        self.assertEqual(usage.input_tokens, 7)
        self.assertEqual(usage.output_tokens, 3)
        self.assertEqual(usage.cached_tokens, 1)
        self.assertEqual(usage.total, 10)
        self.assertEqual(finish, "stop")

        payload = self.state.requests[-1]["payload"]
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "be brief"})
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})

    def test_openai_retries_without_stream_options(self):
        self.state.reject_stream_options = True
        try:
            provider = Provider(
                name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
            )
            text, _reasoning, _usage, finish, _chunks = _read_stream(
                stream_chat(
                    provider,
                    model="mock",
                    messages=[{"role": "user", "content": "hi"}],
                    key=TEST_KEY,
                )
            )
        finally:
            self.state.reject_stream_options = False
        self.assertEqual(text, "Hello from mock")
        self.assertEqual(finish, "stop")

    def test_anthropic_stream(self):
        provider = Provider(
            name="mock",
            label="Mock",
            kind=ANTHROPIC,
            base_url=f"{self.base}/anthropic/v1",
            auth_header="x-api-key",
            auth_prefix="",
        )
        text, _reasoning, usage, finish, _chunks = _read_stream(
            stream_chat(
                provider,
                model="claude-mock",
                messages=[{"role": "user", "content": "hi"}],
                system="be brief",
                key=TEST_KEY,
            )
        )
        self.assertEqual(text, "Hi from Claude")
        self.assertEqual(usage.input_tokens, 5)
        self.assertEqual(usage.output_tokens, 4)
        self.assertEqual(usage.cached_tokens, 2)
        self.assertEqual(finish, "end_turn")

        payload = self.state.requests[-1]["payload"]
        self.assertEqual(payload["system"], "be brief")
        self.assertEqual(payload["max_tokens"], 4096)

    def test_gemini_stream(self):
        provider = Provider(
            name="mock",
            label="Mock",
            kind=GEMINI,
            base_url=f"{self.base}/gemini/v1beta",
            auth_header="x-goog-api-key",
            auth_prefix="",
        )
        text, _reasoning, usage, finish, _chunks = _read_stream(
            stream_chat(
                provider,
                model="gemini-mock",
                messages=[{"role": "user", "content": "hi"}],
                system="be brief",
                key=TEST_KEY,
            )
        )
        self.assertEqual(text, "Gemini says hi")
        self.assertEqual(usage.input_tokens, 4)
        self.assertEqual(usage.output_tokens, 2)
        self.assertEqual(finish, "stop")

        payload = self.state.requests[-1]["payload"]
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "be brief"}]})
        self.assertEqual(payload["contents"][0]["role"], "user")

    def test_auth_failure(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/bad/v1"
        )
        with self.assertRaises(AuthError) as caught:
            _read_stream(
                stream_chat(
                    provider,
                    model="mock",
                    messages=[{"role": "user", "content": "hi"}],
                    key=TEST_KEY,
                )
            )
        self.assertIn("authentication failed", str(caught.exception).lower())

    def test_list_models(self):
        openai = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )
        self.assertEqual(
            list_models(openai, key=TEST_KEY), ["mock-large", "mock-small"]
        )
        anthropic = Provider(
            name="mock",
            label="Mock",
            kind=ANTHROPIC,
            base_url=f"{self.base}/anthropic/v1",
            auth_header="x-api-key",
            auth_prefix="",
        )
        self.assertEqual(list_models(anthropic, key=TEST_KEY), ["claude-mock"])
        gemini = Provider(
            name="mock",
            label="Mock",
            kind=GEMINI,
            base_url=f"{self.base}/gemini/v1beta",
            auth_header="x-goog-api-key",
            auth_prefix="",
        )
        self.assertEqual(list_models(gemini, key=TEST_KEY), ["gemini-mock"])


class SseTests(unittest.TestCase):
    def test_crlf_comments_and_multiline_data(self):
        body = (
            b": keepalive\r\n\r\n"
            b"data: {\"a\": 1}\r\n\r\n"
            b"data: line1\r\ndata: line2\r\n\r\n"
            b"data: [DONE]\r\n\r\n"
        )

        class FakeResponse:
            def __init__(self, raw):
                self._stream = io.BytesIO(raw)

            def read(self, size):
                return self._stream.read(size)

        events = list(iter_sse(FakeResponse(body)))
        self.assertEqual(events, ['{"a": 1}', "line1\nline2"])

    def test_join_url_preserves_base_query(self):
        self.assertEqual(
            join_url(
                "https://example.test/v1?api-version=2024-01-01",
                "chat/completions",
            ),
            "https://example.test/v1/chat/completions?api-version=2024-01-01",
        )


class KeyStoreTests(unittest.TestCase):
    def test_roundtrip_and_env_precedence(self):
        tmp = new_temp_dir(self)
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(tmp)}):
            store = KeyStore()
            scheme = store.set("openai", "sk-test-abcdef")
            expected = "dpapi" if os.name == "nt" else "plain"
            self.assertEqual(scheme, expected)
            self.assertEqual(store.get("openai"), "sk-test-abcdef")
            self.assertEqual(store.masked("openai"), "sk-t...cdef")
            self.assertTrue(store.remove("openai"))
            self.assertIsNone(store.get("openai"))

            store.set("openai", "saved-key")
            provider = Provider(
                name="openai",
                label="OpenAI",
                kind=OPENAI,
                base_url="https://api.openai.com/v1",
                env_vars=("OPENAI_API_KEY",),
            )
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
                resolution = resolve_key(provider, store)
            self.assertEqual(resolution.value, "env-key")
            self.assertEqual(resolution.env_var, "OPENAI_API_KEY")


class ConfigTests(unittest.TestCase):
    def test_roundtrip(self):
        tmp = new_temp_dir(self)
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(tmp)}):
            config = load_config()
            self.assertIsNone(config.get("provider"))
            config.set("provider", "openai")
            config.set("temperature", 0.3)
            config.save()
            reloaded = load_config()
            self.assertEqual(reloaded.get("provider"), "openai")
            self.assertEqual(reloaded.get("temperature"), 0.3)


class CatalogTests(unittest.TestCase):
    def test_openscale_is_a_builtin_openai_compatible_provider(self):
        provider = BUILTINS["openscale"]
        self.assertEqual(provider.base_url, "https://api.openscale.so/v1")
        self.assertEqual(provider.kind, OPENAI)
        self.assertEqual(provider.env_vars, ("OPENSCALE_API_KEY",))
        self.assertIn("deepseek/deepseek-v4.1-flash", provider.models)
        self.assertEqual(provider.auth_header, "Authorization")


class SetupTests(ServerTestCase):
    def test_first_run_prompts_for_provider_and_key(self):
        home = self.temp_home()
        args = SimpleNamespace(provider=None, model=None, base_url=None)

        class TtyBuffer(io.StringIO):
            def isatty(self):
                return True

        output = TtyBuffer()
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}, clear=True):
            config = load_config()
            providers = load_providers(config)
            store = KeyStore()
            with (
                mock.patch("sys.stdin", new=TtyBuffer()),
                redirect_stdout(output),
                mock.patch("builtins.input", return_value=""),
                mock.patch("anyllm.cli.getpass.getpass", return_value="sk-first-run"),
            ):
                _interactive_setup(args, config, providers, store, Style(False))
            self.assertEqual(load_config().get("provider"), "openai")
            self.assertEqual(KeyStore().get("openai"), "sk-first-run")

        self.assertIn("Welcome to AnyLLM", output.getvalue())
        self.assertIn("No API key found for openai", output.getvalue())


class SessionTests(ServerTestCase):
    def test_session_persists_messages_and_totals(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )
        home = self.temp_home()
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}):
            session = Session(provider, "mock", api_key=TEST_KEY)
            turn = session.send("hi")
            self.assertEqual(turn.text, "Hello from mock")
            self.assertEqual(turn.usage.total, 10)
            self.assertEqual(len(session.messages), 2)
            self.assertEqual(session.turns, 1)

            save_session(session, "demo")
            data = load_session_data("demo")
            self.assertEqual(data["provider"], "mock")
            self.assertEqual(data["model"], "mock")
            self.assertEqual(data["messages"], session.messages)
            self.assertEqual(data["usage"]["total"], 10)


class ReplTests(ServerTestCase):
    def test_scripted_conversation(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )
        home = self.temp_home()
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}):
            session = Session(provider, "mock", api_key=TEST_KEY)
            repl = Repl(
                session,
                Style(False),
                {"mock": provider},
                KeyStore(),
                load_config(),
            )
            lines = iter(
                [
                    "hi",
                    "/usage",
                    "/save demo",
                    "/sessions",
                    "/model other-model",
                    "/info",
                    "/clear",
                    "/usage",
                    "/exit",
                ]
            )
            output = io.StringIO()
            with (
                mock.patch("builtins.input", side_effect=lambda prompt="": next(lines)),
                redirect_stdout(output),
            ):
                code = repl.run()
        text = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Hello from mock", text)
        self.assertIn("Session usage", text)
        self.assertIn("saved →", text)
        self.assertIn("Saved sessions", text)
        self.assertIn("model → other-model", text)
        self.assertIn("provider:  mock", text)
        self.assertIn("conversation cleared", text)
        self.assertIn("no requests in this session yet", text)
        self.assertIn("bye", text)
        self.assertTrue((home / "sessions" / "demo.json").exists())


class CliTests(ServerTestCase):
    def test_ask_json(self):
        home = self.temp_home()
        result = self.run_cli(
            [
                "ask",
                "--provider",
                "openai",
                "--base-url",
                f"{self.base}/openai/v1",
                "--model",
                "mock",
                "--json",
                "hi",
            ],
            home,
            extra_env={"OPENAI_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["text"], "Hello from mock")
        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(payload["usage"]["total"], 10)
        self.assertEqual(payload["finish_reason"], "stop")

    def test_ask_streams_plain_text(self):
        home = self.temp_home()
        result = self.run_cli(
            [
                "ask",
                "--provider",
                "openai",
                "--base-url",
                f"{self.base}/openai/v1",
                "--model",
                "mock",
                "hi",
            ],
            home,
            extra_env={"OPENAI_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "Hello from mock\n")

    def test_ask_reads_stdin(self):
        home = self.temp_home()
        result = self.run_cli(
            [
                "ask",
                "--provider",
                "openai",
                "--base-url",
                f"{self.base}/openai/v1",
                "--model",
                "mock",
            ],
            home,
            input_text="what is this?\n",
            extra_env={"OPENAI_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "Hello from mock\n")

    def test_chat_with_piped_stdin_answers_once(self):
        home = self.temp_home()
        result = self.run_cli(
            [
                "chat",
                "--provider",
                "openai",
                "--base-url",
                f"{self.base}/openai/v1",
                "--model",
                "mock",
            ],
            home,
            input_text="hi\n",
            extra_env={"OPENAI_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "Hello from mock\n")

    def test_ask_without_a_key_explains_how_to_add_one(self):
        home = self.temp_home()
        result = self.run_cli(["ask", "--provider", "openai", "hi"], home)
        self.assertEqual(result.returncode, 1)
        self.assertIn("No API key found for openai", result.stderr)
        self.assertIn("anyllm keys set openai", result.stderr)

    def test_ask_reports_auth_error(self):
        home = self.temp_home()
        result = self.run_cli(
            ["ask", "--provider", "openai", "--base-url", f"{self.base}/bad/v1", "hi"],
            home,
            extra_env={"OPENAI_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("authentication failed", result.stderr.lower())

    def test_models_command(self):
        home = self.temp_home()
        result = self.run_cli(
            ["models", "openai", "--base-url", f"{self.base}/openai/v1"],
            home,
            extra_env={"OPENAI_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["mock-large", "mock-small"])

    def test_custom_provider_end_to_end(self):
        home = self.temp_home()
        added = self.run_cli(
            [
                "providers",
                "add",
                "lab",
                "--base-url",
                f"{self.base}/openai/v1",
                "--model",
                "mock",
            ],
            home,
        )
        self.assertEqual(added.returncode, 0, added.stderr)

        keyed = self.run_cli(
            ["keys", "set", "lab", "--stdin"], home, input_text=TEST_KEY + "\n"
        )
        self.assertEqual(keyed.returncode, 0, keyed.stderr)

        result = self.run_cli(
            ["ask", "--provider", "lab", "--json", "hi"], home
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["text"], "Hello from mock")

        keys_file = home / "keys.json"
        self.assertTrue(keys_file.exists())
        stored = json.loads(keys_file.read_text(encoding="utf-8"))
        expected = "dpapi" if os.name == "nt" else "plain"
        self.assertEqual(stored["keys"]["lab"]["scheme"], expected)
        self.assertNotIn(TEST_KEY, keys_file.read_text(encoding="utf-8"))

        doctor = self.run_cli(["doctor", "--test"], home)
        self.assertEqual(doctor.returncode, 0, doctor.stderr)
        self.assertIn("ok", doctor.stdout)


class BaseUrlTests(ServerTestCase):
    def test_saved_session_keeps_custom_base_url(self):
        provider = Provider(
            name="mock",
            label="Mock",
            kind=OPENAI,
            base_url="https://default.example/v1",
        )
        custom = f"{self.base}/openai/v1?api-version=2024-10-21"
        home = self.temp_home()
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}):
            store = KeyStore()
            store.set("mock", TEST_KEY)
            session = Session(
                provider,
                "mock",
                api_key=TEST_KEY,
                base_url=custom,
                temperature=0.0,
                max_tokens=8,
            )
            session.messages = [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
            save_session(session, "demo")
            saved = load_session_data("demo")
            self.assertEqual(saved["base_url"], custom)

            repl = Repl(
                Session(provider, "other", api_key="stale", base_url="https://default.example/v1"),
                Style(False),
                {"mock": provider},
                store,
                load_config(),
            )
            with redirect_stdout(io.StringIO()):
                repl._cmd_load("demo")
            self.assertEqual(repl.session.base_url, custom)
            self.assertEqual(repl.session.model, "mock")
            self.assertEqual(repl.session.temperature, 0.0)
            self.assertEqual(repl.session.max_tokens, 8)
            self.assertEqual(repl.session.messages[0]["content"], "hi")

            args = SimpleNamespace(
                model=None,
                system=None,
                temperature=None,
                max_tokens=None,
                base_url=None,
                reasoning=False,
                timeout=5,
            )
            with redirect_stdout(io.StringIO()):
                resumed = _session_from_saved("demo", args, {"mock": provider}, store, Style(False))
            self.assertEqual(resumed.base_url, custom)

            args.base_url = "http://127.0.0.1:9/v1"
            with redirect_stdout(io.StringIO()):
                overridden = _session_from_saved(
                    "demo", args, {"mock": provider}, store, Style(False)
                )
            self.assertEqual(overridden.base_url, "http://127.0.0.1:9/v1")

    def test_old_session_without_base_url_uses_provider_default(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url="https://default.example/v1"
        )
        for saved in (None, "", "   "):
            with self.subTest(saved=saved):
                data = {"provider": "mock", "model": "mock"}
                if saved is not None:
                    data["base_url"] = saved
                self.assertEqual(session_base_url(data, provider), provider.base_url)
        self.assertEqual(
            session_base_url({}, provider, " https://override.example/v1/ "),
            "https://override.example/v1",
        )

    def test_invalid_saved_base_url_is_rejected(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url="https://default.example/v1"
        )
        with self.assertRaises(ConfigError):
            session_base_url({"base_url": "ftp://files.example/v1"}, provider)
        with self.assertRaises(ConfigError):
            session_base_url({"base_url": 5}, provider)
        with self.assertRaises(ConfigError):
            session_base_url({"base_url": "not a url"}, provider, None)

    def test_chat_load_uses_saved_base_url(self):
        home = self.temp_home()
        payload = {
            "version": 1,
            "provider": "openai",
            "model": "mock",
            "base_url": f"{self.base}/openai/v1",
            "temperature": 0.2,
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "ok"},
            ],
        }
        directory = home / "sessions"
        directory.mkdir(parents=True)
        (directory / "demo.json").write_text(json.dumps(payload), encoding="utf-8")
        before = len(self.state.requests)
        result = self.run_cli(
            ["chat", "--load", "demo", "--timeout", "5"],
            home,
            input_text="hi\n",
            extra_env={"OPENAI_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Hello from mock", result.stdout)
        self.assertIn("resumed demo", result.stdout)
        posts = self._posts_since(before)
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["path"], "/openai/v1/chat/completions")
        roles = [item["role"] for item in posts[0]["payload"]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_chat_load_rejects_bad_sampling_before_request(self):
        home = self.temp_home()
        payload = {
            "version": 1,
            "provider": "openai",
            "model": "mock",
            "base_url": f"{self.base}/openai/v1",
            "temperature": 9,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "earlier"}],
        }
        directory = home / "sessions"
        directory.mkdir(parents=True)
        (directory / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
        before = len(self.state.requests)
        result = self.run_cli(
            ["chat", "--load", "bad", "--timeout", "5"],
            home,
            input_text="hi\n",
            extra_env={"OPENAI_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("temperature", result.stderr.lower())
        self.assertEqual(self._posts_since(before), [])


class SamplingValidationTests(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(validate_temperature(0), 0.0)
        self.assertEqual(validate_temperature(2), 2.0)
        self.assertEqual(validate_temperature("0.25"), 0.25)
        self.assertIsNone(validate_temperature(None))
        self.assertEqual(validate_max_tokens(1), 1)
        self.assertEqual(validate_max_tokens("16"), 16)
        self.assertIsNone(validate_max_tokens(None))
        for value in (-0.1, 2.1, "hot", [], {}):
            with self.subTest(temperature=value):
                with self.assertRaises(ConfigError):
                    validate_temperature(value)
        for value in (0, -3, "0", "no", [], {}):
            with self.subTest(max_tokens=value):
                with self.assertRaises(ConfigError):
                    validate_max_tokens(value)

    def test_config_set_still_validates(self):
        tmp = new_temp_dir(self)
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(tmp)}, clear=True):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(ConfigError):
                    cmd_config_set(SimpleNamespace(key="temperature", value="9"), Style(False))
                with self.assertRaises(ConfigError):
                    cmd_config_set(SimpleNamespace(key="max_tokens", value="0"), Style(False))
                cmd_config_set(SimpleNamespace(key="temperature", value="0"), Style(False))
                cmd_config_set(SimpleNamespace(key="max_tokens", value="1024"), Style(False))
            config = load_config()
            self.assertEqual(config.get("temperature"), 0.0)
            self.assertEqual(config.get("max_tokens"), 1024)


class GenericKeyTests(unittest.TestCase):
    def _store(self) -> KeyStore:
        return KeyStore(new_temp_dir(self) / "keys.json")

    def _env(self, **extra):
        cleaned = {
            key: value
            for key, value in os.environ.items()
            if not key.endswith("_API_KEY") and not key.startswith("ANYLLM_")
        }
        cleaned.update(extra)
        return mock.patch.dict(os.environ, cleaned, clear=True)

    def test_builtin_hosts_do_not_receive_generic_key(self):
        store = self._store()
        with self._env(ANYLLM_API_KEY="generic-secret"):
            for name, provider in BUILTINS.items():
                if name == "custom":
                    continue
                with self.subTest(provider=name):
                    resolution = resolve_key(provider, store)
                    self.assertNotEqual(resolution.value, "generic-secret")
                    self.assertNotEqual(resolution.env_var, "ANYLLM_API_KEY")
            custom = BUILTINS["custom"]
            resolution = resolve_key(custom, store)
            self.assertEqual(resolution.value, "generic-secret")
            self.assertEqual(resolution.env_var, "ANYLLM_API_KEY")
            self.assertTrue(accepts_generic_api_key(custom))
            self.assertFalse(accepts_generic_api_key(BUILTINS["openai"]))
            self.assertFalse(accepts_generic_api_key(BUILTINS["groq"]))
            self.assertFalse(accepts_generic_api_key(BUILTINS["ollama"]))
            self.assertFalse(accepts_generic_api_key(BUILTINS["anthropic"]))
            self.assertFalse(accepts_generic_api_key(BUILTINS["google"]))

    def test_user_openai_provider_uses_generic_key_last(self):
        store = self._store()
        lab = Provider(
            name="lab",
            label="Lab",
            kind=OPENAI,
            base_url="http://127.0.0.1:9/v1",
            env_vars=(),
        )
        gate = Provider(
            name="gate",
            label="Gate",
            kind=ANTHROPIC,
            base_url="https://gate.example/v1",
            env_vars=(),
            auth_header="x-api-key",
            auth_prefix="",
        )
        self.assertTrue(accepts_generic_api_key(lab))
        self.assertFalse(accepts_generic_api_key(gate))
        with self._env(ANYLLM_API_KEY="generic-secret"):
            self.assertEqual(resolve_key(lab, store).value, "generic-secret")
            self.assertIsNone(resolve_key(gate, store).value)
            explicit = Provider(
                name="gate",
                label="Gate",
                kind=ANTHROPIC,
                base_url="https://gate.example/v1",
                env_vars=("ANYLLM_API_KEY",),
                auth_header="x-api-key",
                auth_prefix="",
            )
            self.assertEqual(resolve_key(explicit, store).value, "generic-secret")

        with self._env(ANYLLM_LAB_API_KEY="specific", ANYLLM_API_KEY="generic-secret"):
            resolution = resolve_key(lab, store)
            self.assertEqual(resolution.value, "specific")
            self.assertEqual(resolution.env_var, "ANYLLM_LAB_API_KEY")

        store.set("openai", "saved-openai")
        store.set("lab", "saved-lab")
        with self._env(ANYLLM_API_KEY="generic-secret"):
            openai = resolve_key(BUILTINS["openai"], store)
            self.assertEqual(openai.value, "saved-openai")
            self.assertTrue(openai.stored)
            lab_resolution = resolve_key(lab, store)
            self.assertEqual(lab_resolution.value, "generic-secret")
            self.assertEqual(lab_resolution.env_var, "ANYLLM_API_KEY")


class GenericKeyCliTests(ServerTestCase):
    def test_generic_key_is_not_sent_to_openai_or_groq(self):
        home = self.temp_home()
        before = len(self.state.requests)
        for provider in ("openai", "groq", "anthropic"):
            with self.subTest(provider=provider):
                result = self.run_cli(
                    [
                        "ask",
                        "--provider",
                        provider,
                        "--base-url",
                        f"{self.base}/openai/v1",
                        "--model",
                        "mock",
                        "--timeout",
                        "5",
                        "hi",
                    ],
                    home,
                    extra_env={"ANYLLM_API_KEY": TEST_KEY},
                )
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("No API key", result.stderr)
        self.assertEqual(len(self.state.requests), before)

    def test_generic_key_works_for_added_openai_provider(self):
        home = self.temp_home()
        added = self.run_cli(
            [
                "providers",
                "add",
                "lab",
                "--base-url",
                f"{self.base}/openai/v1",
                "--model",
                "mock",
            ],
            home,
        )
        self.assertEqual(added.returncode, 0, added.stderr)
        result = self.run_cli(
            ["ask", "--provider", "lab", "--json", "hi"],
            home,
            extra_env={"ANYLLM_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["text"], "Hello from mock")
        self.assertEqual(
            self.state.requests[-1]["headers"].get("Authorization"),
            f"Bearer {TEST_KEY}",
        )

    def test_saved_openai_key_wins_over_generic_env(self):
        home = self.temp_home()
        keyed = self.run_cli(
            ["keys", "set", "openai", "--stdin"],
            home,
            input_text="saved-not-the-mock-key\n",
        )
        self.assertEqual(keyed.returncode, 0, keyed.stderr)
        before = len(self.state.requests)
        result = self.run_cli(
            [
                "ask",
                "--provider",
                "openai",
                "--base-url",
                f"{self.base}/openai/v1",
                "--model",
                "mock",
                "--timeout",
                "5",
                "hi",
            ],
            home,
            extra_env={"ANYLLM_API_KEY": TEST_KEY},
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("authentication failed", result.stderr.lower())
        posts = self._posts_since(before)
        self.assertEqual(len(posts), 1)
        self.assertNotIn(TEST_KEY, posts[0]["headers"].get("Authorization", ""))
        self.assertIn("saved-not-the-mock-key", posts[0]["headers"].get("Authorization", ""))


class ProviderCommandTests(ServerTestCase):
    def test_switch_preserves_override_and_clears_stale_model(self):
        home = self.temp_home()
        env = {
            "ANYLLM_HOME": str(home),
            "OPENAI_API_KEY": "sk-openai",
            "GROQ_API_KEY": "groq-key",
            "ANYLLM_API_KEY": "generic-secret",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            session = Session(
                BUILTINS["openai"],
                "gpt-custom",
                base_url="https://proxy.example/v1",
                api_key="old",
            )
            session.messages = [{"role": "user", "content": "keep me"}]
            repl = Repl(session, Style(False), dict(BUILTINS), KeyStore(), load_config())
            output = io.StringIO()

            with redirect_stdout(output):
                with self.assertRaises(ConfigError) as caught:
                    repl._cmd_provider("custom")
            self.assertIn("no base URL", str(caught.exception))
            self.assertIn("unchanged", caught.exception.hint or "")
            self.assertNotIn("API key", str(caught.exception))
            self.assertEqual(repl.session.provider.name, "openai")
            self.assertEqual(repl.session.base_url, "https://proxy.example/v1")
            self.assertEqual(repl.session.model, "gpt-custom")
            self.assertEqual(repl.session.api_key, "old")
            self.assertEqual(repl.session.messages[0]["content"], "keep me")

            with redirect_stdout(output):
                repl._cmd_provider("openai")
            self.assertEqual(repl.session.provider.name, "openai")
            self.assertEqual(repl.session.base_url, "https://proxy.example/v1")
            self.assertEqual(repl.session.model, "gpt-custom")
            self.assertEqual(repl.session.api_key, "sk-openai")

            held = os.environ.pop("GROQ_API_KEY")
            try:
                with redirect_stdout(output):
                    with self.assertRaises(ConfigError) as missing:
                        repl._cmd_provider("groq")
                self.assertIn("No API key", str(missing.exception))
            finally:
                os.environ["GROQ_API_KEY"] = held
            self.assertEqual(repl.session.provider.name, "openai")
            self.assertEqual(repl.session.base_url, "https://proxy.example/v1")
            self.assertEqual(repl.session.model, "gpt-custom")

            with redirect_stdout(output):
                repl._cmd_provider("groq")
            self.assertEqual(repl.session.provider.name, "groq")
            self.assertEqual(repl.session.base_url, BUILTINS["groq"].base_url)
            self.assertEqual(repl.session.model, BUILTINS["groq"].default_model)
            self.assertEqual(repl.session.api_key, "groq-key")
            self.assertEqual(repl.session.messages[0]["content"], "keep me")
            self.assertIn("model reset from gpt-custom", output.getvalue())

            with redirect_stdout(output):
                repl._cmd_provider("lmstudio")
            self.assertEqual(repl.session.base_url, BUILTINS["lmstudio"].base_url)
            self.assertEqual(repl.session.model, "")
            self.assertIn("cleared the previous model", output.getvalue())
            with self.assertRaises(UsageError):
                repl.session.send("hi")
            self.assertEqual(len(repl.session.messages), 1)

            custom_session = Session(
                BUILTINS["custom"],
                "local-model",
                base_url="http://127.0.0.1:8000/v1",
                api_key="local",
            )
            custom_repl = Repl(
                custom_session, Style(False), dict(BUILTINS), KeyStore(), load_config()
            )
            with redirect_stdout(output):
                custom_repl._cmd_provider("custom")
            self.assertEqual(custom_repl.session.base_url, "http://127.0.0.1:8000/v1")
            self.assertEqual(custom_repl.session.model, "local-model")
            with redirect_stdout(output):
                custom_repl._cmd_provider("openai")
            self.assertEqual(custom_repl.session.base_url, BUILTINS["openai"].base_url)
            self.assertEqual(custom_repl.session.model, BUILTINS["openai"].default_model)


class LoadValidationTests(ServerTestCase):
    def test_repl_load_rejects_bad_sampling_without_replacing_session(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )
        home = self.temp_home()
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}):
            store = KeyStore()
            store.set("mock", TEST_KEY)
            session = Session(
                provider,
                "mock",
                api_key=TEST_KEY,
                base_url=f"{self.base}/openai/v1",
                temperature=0.4,
                max_tokens=20,
            )
            session.messages = [{"role": "user", "content": "hi"}]
            path = save_session(session, "demo")
            repl = Repl(
                Session(provider, "live", api_key=TEST_KEY, temperature=0.1, max_tokens=4),
                Style(False),
                {"mock": provider},
                store,
                load_config(),
            )
            raw = json.loads(path.read_text(encoding="utf-8"))
            for key, value in (
                ("temperature", 9),
                ("temperature", -0.01),
                ("temperature", "hot"),
                ("max_tokens", 0),
                ("max_tokens", -1),
                ("max_tokens", "no"),
            ):
                with self.subTest(key=key, value=value):
                    broken = dict(raw)
                    broken[key] = value
                    path.write_text(json.dumps(broken), encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        repl._cmd_load("demo")
                    self.assertEqual(repl.session.model, "live")
                    self.assertEqual(repl.session.temperature, 0.1)
                    self.assertEqual(repl.session.max_tokens, 4)

            raw["temperature"] = "1.5"
            raw["max_tokens"] = "32"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                repl._cmd_load("demo")
            self.assertEqual(repl.session.temperature, 1.5)
            self.assertEqual(repl.session.max_tokens, 32)
            self.assertEqual(repl.session.model, "mock")


class InterruptTests(ServerTestCase):
    def test_ctrl_c_marks_partial_answer_and_empty_interrupt_drops_turn(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )
        session = Session(provider, "mock", api_key=TEST_KEY)

        def interrupted(*_args, **_kwargs):
            yield Chunk(text="Hello")
            yield Chunk(text=" there")
            raise KeyboardInterrupt

        with mock.patch("anyllm.session.stream_chat", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                session.send("hi")
        self.assertEqual(session.turns, 0)
        self.assertIsNone(session.usage.total)
        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": f"Hello there\n{INTERRUPTED_MARK}",
                    "interrupted": True,
                },
            ],
        )

        def already_marked(*_args, **_kwargs):
            yield Chunk(text=f"Hello there\n{INTERRUPTED_MARK}")
            raise KeyboardInterrupt

        session.messages.clear()
        with mock.patch("anyllm.session.stream_chat", already_marked):
            with self.assertRaises(KeyboardInterrupt):
                session.send("again")
        self.assertEqual(
            session.messages[1]["content"], f"Hello there\n{INTERRUPTED_MARK}"
        )

        home = self.temp_home()
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}):
            save_session(session, "partial")
            stored = load_session_data("partial")
        self.assertTrue(stored["messages"][1]["interrupted"])
        self.assertIn(INTERRUPTED_MARK, stored["messages"][1]["content"])

        follow = Session(provider, "mock", api_key=TEST_KEY, messages=list(session.messages))
        follow.send("continue")
        payload = self.state.requests[-1]["payload"]
        self.assertIn(
            f"Hello there\n{INTERRUPTED_MARK}",
            [item["content"] for item in payload["messages"]],
        )
        self.assertNotIn("interrupted", payload["messages"][1])

        blank = Session(provider, "mock", api_key=TEST_KEY)

        def reasoning_only(*_args, **_kwargs):
            yield Chunk(reasoning="hmm")
            raise KeyboardInterrupt

        with mock.patch("anyllm.session.stream_chat", reasoning_only):
            with self.assertRaises(KeyboardInterrupt):
                blank.send("hi")
        self.assertEqual(blank.messages, [])

        def whitespace(*_args, **_kwargs):
            yield Chunk(text=" \n")
            raise KeyboardInterrupt

        with mock.patch("anyllm.session.stream_chat", whitespace):
            with self.assertRaises(KeyboardInterrupt):
                blank.send("hi")
        self.assertEqual(blank.messages, [])


class StreamOptionTests(ServerTestCase):
    def _provider(self) -> Provider:
        return Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )

    def _chat(self):
        return stream_chat(
            self._provider(),
            model="mock",
            messages=[{"role": "user", "content": "hi"}],
            key=TEST_KEY,
        )

    def test_stream_options_rejection_variants_retry_once(self):
        cases = [
            (
                400,
                {
                    "error": {
                        "message": "Unrecognized request argument supplied: stream_options",
                        "param": "stream_options",
                    }
                },
            ),
            (
                400,
                {
                    "error": {
                        "message": "Unknown parameter",
                        "param": "stream_options",
                        "code": "unknown_parameter",
                    }
                },
            ),
            (422, {"error": {"message": "stream options are not supported"}}),
            (400, {"error": {"message": "stream-options: unsupported"}}),
            (
                422,
                {
                    "detail": [
                        {
                            "loc": ["body", "stream_options"],
                            "msg": "Extra inputs are not permitted",
                            "type": "extra_forbidden",
                        }
                    ]
                },
            ),
            (
                400,
                {
                    "error": {
                        "message": "Additional properties are not allowed ('stream_options' was unexpected)"
                    }
                },
            ),
            (400, {"error": "unknown_parameter: stream_options"}),
            (
                400,
                {
                    "error": {
                        "message": "Unknown parameter: 'include_usage'",
                        "param": "include_usage",
                    }
                },
            ),
        ]
        for status, body in cases:
            with self.subTest(status=status, body=body):
                self.state.reject_stream_options = True
                self.state.stream_options_status = status
                self.state.stream_options_body = body
                start = len(self.state.requests)
                with mock.patch("anyllm.transport.time.sleep"):
                    text, _reasoning, _usage, finish, _chunks = _read_stream(self._chat())
                self.assertEqual(text, "Hello from mock")
                self.assertEqual(finish, "stop")
                posts = self._posts_since(start)
                self.assertEqual(len(posts), 2)
                self.assertIn("stream_options", posts[0]["payload"])
                self.assertNotIn("stream_options", posts[1]["payload"])
                self.state.reject_stream_options = False

    def test_stream_options_fallback_does_not_hide_other_errors(self):
        cases = [
            (
                400,
                {
                    "error": {
                        "message": "Unknown parameter: temperature",
                        "param": "temperature",
                        "code": "unknown_parameter",
                    }
                },
            ),
            (
                400,
                {"error": {"message": "This model's maximum context length was exceeded"}},
            ),
            (
                400,
                {
                    "error": {
                        "message": "The stream_options field was ignored because the model is unknown"
                    }
                },
            ),
            (
                500,
                {
                    "error": {
                        "message": "Unsupported parameter: stream_options",
                        "param": "stream_options",
                    }
                },
            ),
            (
                409,
                {
                    "error": {
                        "message": "Unsupported parameter: stream_options",
                        "param": "stream_options",
                    }
                },
            ),
            (400, {"error": {"message": "bad request"}}),
        ]
        for status, body in cases:
            with self.subTest(status=status, body=body):
                self.state.openai_failures = [(status, body)]
                start = len(self.state.requests)
                with mock.patch("anyllm.transport.time.sleep"):
                    with self.assertRaises(APIError) as caught:
                        _read_stream(self._chat())
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(len(self._posts_since(start)), 1)

    def test_stream_options_fallback_does_not_loop(self):
        body = {"error": {"message": "Unsupported parameter: stream_options"}}
        self.state.openai_failures = [(400, body), (400, body)]
        start = len(self.state.requests)
        with mock.patch("anyllm.transport.time.sleep"):
            with self.assertRaises(APIError):
                _read_stream(self._chat())
        self.assertEqual(len(self._posts_since(start)), 2)


class CompletionRetryTests(ServerTestCase):
    def _provider(self) -> Provider:
        return Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )

    def test_post_retries_safe_statuses_only(self):
        provider = self._provider()
        for status in (408, 425, 429, 503):
            with self.subTest(status=status):
                self.state.openai_failures = [(status, {"error": {"message": "try again"}})]
                start = len(self.state.requests)
                with mock.patch("anyllm.transport.time.sleep"):
                    text, *_rest = _read_stream(
                        stream_chat(
                            provider,
                            model="mock",
                            messages=[{"role": "user", "content": "hi"}],
                            key=TEST_KEY,
                        )
                    )
                self.assertEqual(text, "Hello from mock")
                self.assertEqual(len(self._posts_since(start)), 2)
        for status in (409, 500, 502, 504):
            with self.subTest(status=status):
                self.state.openai_failures = [(status, {"error": {"message": "do not replay"}})]
                start = len(self.state.requests)
                with mock.patch("anyllm.transport.time.sleep"):
                    with self.assertRaises(APIError) as caught:
                        _read_stream(
                            stream_chat(
                                provider,
                                model="mock",
                                messages=[{"role": "user", "content": "hi"}],
                                key=TEST_KEY,
                            )
                        )
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(len(self._posts_since(start)), 1)

    def test_get_retries_server_errors_but_not_conflict(self):
        provider = self._provider()
        self.state.get_failures = [(500, {"error": {"message": "boom"}})]
        start = len(self.state.requests)
        with mock.patch("anyllm.transport.time.sleep"):
            names = list_models(provider, key=TEST_KEY)
        self.assertEqual(names, ["mock-large", "mock-small"])
        gets = [item for item in self.state.requests[start:] if item["method"] == "GET"]
        self.assertEqual(len(gets), 2)

        self.state.get_failures = [(409, {"error": {"message": "conflict"}})]
        start = len(self.state.requests)
        with mock.patch("anyllm.transport.time.sleep"):
            with self.assertRaises(APIError) as caught:
                list_models(provider, key=TEST_KEY)
        self.assertEqual(caught.exception.status, 409)
        gets = [item for item in self.state.requests[start:] if item["method"] == "GET"]
        self.assertEqual(len(gets), 1)

        self.state.get_failures = [(502, {"error": {"message": "gateway"}})]
        start = len(self.state.requests)
        with mock.patch("anyllm.transport.time.sleep"):
            self.assertEqual(list_models(provider, key=TEST_KEY), ["mock-large", "mock-small"])
        gets = [item for item in self.state.requests[start:] if item["method"] == "GET"]
        self.assertEqual(len(gets), 2)

    def test_midstream_failure_does_not_repost(self):
        provider = self._provider()

        class Broken:
            def __init__(self):
                self.reads = 0

            def read(self, _n=-1):
                self.reads += 1
                if self.reads == 1:
                    return b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
                raise ConnectionResetError(104, "Connection reset by peer")

            def close(self):
                pass

        calls = {"n": 0}

        def fake_open(*_args, **_kwargs):
            calls["n"] += 1
            return Broken()

        with mock.patch("anyllm.adapters.open_stream", fake_open):
            with self.assertRaises(NetworkError):
                _read_stream(
                    stream_chat(
                        provider,
                        model="mock",
                        messages=[{"role": "user", "content": "hi"}],
                        key=TEST_KEY,
                    )
                )
        self.assertEqual(calls["n"], 1)

        self.state.reset_midstream = True
        start = len(self.state.requests)
        with self.assertRaises(NetworkError):
            _read_stream(
                stream_chat(
                    provider,
                    model="mock",
                    messages=[{"role": "user", "content": "hi"}],
                    key=TEST_KEY,
                )
            )
        self.assertEqual(len(self._posts_since(start)), 1)


class TransportRetryTests(unittest.TestCase):
    def _ok(self, body: bytes = b'{"ok": true}'):
        class Response:
            def __init__(self):
                self._body = io.BytesIO(body)
                self.status = 200

            def read(self, n=-1):
                return self._body.read(n)

            def close(self):
                pass

        return Response()

    def _http_error(self, status: int, payload=None, headers=None):
        body = json.dumps(payload or {"error": {"message": "nope"}}).encode()
        hdrs = Message()
        for key, value in (headers or {}).items():
            hdrs[key] = str(value)
        return error.HTTPError(
            "http://127.0.0.1/v1/chat/completions",
            status,
            "error",
            hdrs,
            io.BytesIO(body),
        )

    def test_post_retries_connection_refused_not_later_network_failures(self):
        calls = {"n": 0}

        def refused_then_ok(req, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise error.URLError(ConnectionRefusedError(111, "Connection refused"))
            return self._ok()

        with (
            mock.patch("anyllm.transport.request.urlopen", refused_then_ok),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            response = open_stream("http://127.0.0.1:9/v1/chat/completions", {}, {"x": 1})
            response.close()
        self.assertEqual(calls["n"], 3)

        failures = [
            error.URLError(socket.timeout("timed out")),
            error.URLError(ConnectionResetError(104, "Connection reset by peer")),
            error.URLError(ssl.SSLError("certificate verify failed")),
            TimeoutError("timed out"),
            ConnectionResetError(104, "Connection reset by peer"),
            error.URLError(OSError(104, "Connection reset by peer")),
        ]
        for exc in failures:
            with self.subTest(exc=repr(exc)):
                seen = {"n": 0}

                def fail(req, timeout=None, exc=exc):
                    seen["n"] += 1
                    raise exc

                with (
                    mock.patch("anyllm.transport.request.urlopen", fail),
                    mock.patch("anyllm.transport.time.sleep"),
                ):
                    with self.assertRaises(NetworkError):
                        open_stream(
                            "http://127.0.0.1:9/v1/chat/completions",
                            {},
                            {"x": 1},
                            retries=2,
                        )
                self.assertEqual(seen["n"], 1)

    def test_dns_failure_retries_because_nothing_was_sent(self):
        calls = {"n": 0}

        def flaky(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise error.URLError(socket.gaierror(8, "nodename nor servname provided"))
            return self._ok()

        with (
            mock.patch("anyllm.transport.request.urlopen", flaky),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            response = open_stream("http://example.test/v1/chat/completions", {}, {"x": 1})
            response.close()
        self.assertEqual(calls["n"], 2)

    def test_real_refused_connection_is_retried(self):
        calls = {"n": 0}

        def refuse(*_args, **_kwargs):
            calls["n"] += 1
            raise ConnectionRefusedError(111, "Connection refused")

        with (
            mock.patch("socket.create_connection", refuse),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            with self.assertRaises(NetworkError) as caught:
                open_stream("http://127.0.0.1:9/v1/chat/completions", {}, {"x": 1}, retries=2)
        self.assertEqual(calls["n"], 3)
        self.assertIn("refused", str(caught.exception).lower())

    def test_retry_after_cap_and_conflict(self):
        calls = {"n": 0}

        def too_long(req, timeout=None):
            calls["n"] += 1
            raise self._http_error(429, headers={"Retry-After": "30"})

        with (
            mock.patch("anyllm.transport.request.urlopen", too_long),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            with self.assertRaises(Exception):
                open_stream("http://127.0.0.1/v1/chat/completions", {}, {"x": 1}, retries=2)
        self.assertEqual(calls["n"], 1)

        calls["n"] = 0

        def short_wait(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise self._http_error(429, headers={"Retry-After": "1"})
            return self._ok()

        with (
            mock.patch("anyllm.transport.request.urlopen", short_wait),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            response = open_stream("http://127.0.0.1/v1/chat/completions", {}, {"x": 1})
            response.close()
        self.assertEqual(calls["n"], 2)

        calls["n"] = 0

        def conflict(req, timeout=None):
            calls["n"] += 1
            raise self._http_error(409, headers={"Retry-After": "0"})

        with (
            mock.patch("anyllm.transport.request.urlopen", conflict),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            with self.assertRaises(APIError) as caught:
                open_stream("http://127.0.0.1/v1/chat/completions", {}, {"x": 1}, retries=2)
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(calls["n"], 1)

    def test_get_retries_500_post_does_not_and_read_errors_are_terminal(self):
        calls = {"n": 0}

        def flaky_get(req, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise self._http_error(500)
            return self._ok(b'{"data": []}')

        with (
            mock.patch("anyllm.transport.request.urlopen", flaky_get),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            payload = request_json("http://example.test/v1/models", {}, method="GET", retries=2)
        self.assertEqual(payload, {"data": []})
        self.assertEqual(calls["n"], 3)

        calls["n"] = 0

        def post_500(req, timeout=None):
            calls["n"] += 1
            raise self._http_error(500)

        with (
            mock.patch("anyllm.transport.request.urlopen", post_500),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            with self.assertRaises(APIError) as caught:
                request_json(
                    "http://example.test/v1/chat/completions",
                    {},
                    {"model": "m"},
                    method="POST",
                    retries=2,
                )
        self.assertEqual(caught.exception.status, 500)
        self.assertEqual(calls["n"], 1)

        calls["n"] = 0

        class Resetting:
            def read(self, n=-1):
                raise ConnectionResetError(104, "Connection reset by peer")

            def close(self):
                pass

        def accepted_then_reset(req, timeout=None):
            calls["n"] += 1
            return Resetting()

        with (
            mock.patch("anyllm.transport.request.urlopen", accepted_then_reset),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            with self.assertRaises(NetworkError):
                request_json(
                    "http://example.test/v1/chat/completions",
                    {},
                    {"model": "m"},
                    method="POST",
                    retries=2,
                )
        self.assertEqual(calls["n"], 1)

        calls["n"] = 0
        with (
            mock.patch("anyllm.transport.request.urlopen", accepted_then_reset),
            mock.patch("anyllm.transport.time.sleep"),
        ):
            with self.assertRaises(NetworkError):
                request_json("http://example.test/v1/models", {}, method="GET", retries=2)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
