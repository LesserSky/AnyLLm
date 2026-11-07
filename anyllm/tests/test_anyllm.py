"""Offline end-to-end tests for AnyLLM."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for entry in (str(ROOT), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from anyllm.adapters import Usage, list_models, stream_chat
from anyllm.catalog import ANTHROPIC, BUILTINS, GEMINI, OPENAI, Provider, load_providers
from anyllm.cli import _interactive_setup
from anyllm.config import load_config
from anyllm.errors import AuthError
from anyllm.keystore import KeyStore, resolve_key
from anyllm.repl import Repl
from anyllm.session import Session, load_session_data, save_session
from anyllm.term import Style
from anyllm.transport import iter_sse, join_url

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


if __name__ == "__main__":
    unittest.main()
