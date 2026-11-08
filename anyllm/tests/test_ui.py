"""UI, color fallback, and piped-output tests for AnyLLM."""

from __future__ import annotations

import io
import json
import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for entry in (str(ROOT), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from anyllm.adapters import Usage
from anyllm.catalog import OPENAI, Provider
from anyllm.config import load_config
from anyllm.errors import AnyLLMError, ConfigError, NetworkError, RateLimitError
from anyllm.keystore import KeyStore
from anyllm.repl import Repl, StreamPrinter
from anyllm.session import Session
from anyllm.term import (
    GREY,
    RESET,
    Style,
    _windows_vt_supported,
    color_enabled,
    format_error,
    format_reply_status,
    format_table,
    redact_endpoint,
    startup_banner,
)

from mock_api import TEST_KEY
from test_anyllm import ServerTestCase, new_temp_dir


class Tty(io.StringIO):
    def isatty(self):
        return True


class Pipe(io.StringIO):
    def isatty(self):
        return False


def _without_color_env():
    """Drop color overrides so a test starts from a neutral environment."""
    os.environ.pop("NO_COLOR", None)
    os.environ.pop("FORCE_COLOR", None)


class ColorEnabledTests(unittest.TestCase):
    def test_tty_is_colored_and_pipe_is_plain(self):
        with mock.patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
            _without_color_env()
            self.assertTrue(color_enabled(stream=Tty()))
            self.assertFalse(color_enabled(stream=Pipe()))

    def test_no_color_flag_and_env_win_over_force_and_tty(self):
        tty = Tty()
        with mock.patch.dict(os.environ, {"FORCE_COLOR": "1", "TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            self.assertFalse(color_enabled(True, stream=tty))
            os.environ["NO_COLOR"] = "1"
            self.assertFalse(color_enabled(stream=tty))
            # Any non-empty NO_COLOR disables color, including "0".
            os.environ["NO_COLOR"] = "0"
            self.assertFalse(color_enabled(stream=tty))

    def test_empty_no_color_does_not_disable(self):
        with mock.patch.dict(
            os.environ,
            {"NO_COLOR": "", "TERM": "xterm-256color"},
            clear=False,
        ):
            os.environ.pop("FORCE_COLOR", None)
            self.assertTrue(color_enabled(stream=Tty()))

    def test_force_color_overrides_pipe_and_dumb_terminal(self):
        with mock.patch.dict(os.environ, {"FORCE_COLOR": "1", "TERM": "dumb"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            self.assertTrue(color_enabled(stream=Pipe()))
        with mock.patch.dict(os.environ, {"TERM": "dumb"}, clear=False):
            _without_color_env()
            self.assertFalse(color_enabled(stream=Tty()))

    def test_old_windows_console_is_plain_until_vt_works(self):
        tty = Tty()
        with mock.patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            _without_color_env()
            with (
                mock.patch("anyllm.term.os.name", "nt"),
                mock.patch("anyllm.term._windows_vt_supported", return_value=False),
            ):
                self.assertFalse(color_enabled(stream=tty))
            with (
                mock.patch("anyllm.term.os.name", "nt"),
                mock.patch("anyllm.term._windows_vt_supported", return_value=True),
            ):
                self.assertTrue(color_enabled(stream=tty))
                self.assertFalse(color_enabled(stream=Pipe()))

    def test_non_windows_skips_the_vt_probe(self):
        if os.name == "nt":
            self.skipTest("VT probe is the Windows path")
        self.assertTrue(_windows_vt_supported())


class StyleTests(unittest.TestCase):
    def test_disabled_style_returns_text_unchanged(self):
        style = Style(False)
        for sample in ("you> ", "error: boom", ""):
            self.assertEqual(style.bold(sample), sample)
            self.assertEqual(style.muted(sample), sample)
            self.assertEqual(style.reasoning(sample), sample)
            self.assertEqual(style.error(sample), sample)
            self.assertEqual(style.prompt(sample, readline=True), sample)

    def test_enabled_style_wraps_and_resets(self):
        style = Style(True)
        painted = style.accent_bold("anyllm")
        self.assertTrue(painted.startswith("\x1b["))
        self.assertIn("anyllm", painted)
        self.assertTrue(painted.endswith(RESET))
        self.assertIn("thinking", style.reasoning("thinking"))
        self.assertIn("\x1b[2;3m", style.reasoning("thinking"))

    def test_readline_prompt_hides_escapes_but_keeps_the_label(self):
        style = Style(True)
        prompt = style.prompt("you> ", readline=True)
        self.assertIn("you> ", prompt)
        self.assertTrue(prompt.startswith("\001\x1b"))
        self.assertIn("\001" + RESET + "\002", prompt)
        visible = prompt.replace("\001", "").replace("\002", "")
        self.assertIn("you> ", visible)
        plain = style.prompt("you> ", readline=False)
        self.assertEqual(plain, style.accent_bold("you> "))
        self.assertNotIn("\001", plain)


class RedactEndpointTests(unittest.TestCase):
    def test_plain_urls_are_unchanged(self):
        for url in (
            "https://api.openai.com/v1",
            "http://127.0.0.1:11434/v1",
            "https://example.openai.azure.com/openai/deployments/dep?api-version=2024-10-21",
        ):
            self.assertEqual(redact_endpoint(url), url)

    def test_userinfo_and_secret_query_values_are_hidden(self):
        shown = redact_endpoint(
            "https://user:supersecret@example.test:8443/v1?api_key=supersecret&api-version=2024-01-01"
        )
        self.assertNotIn("supersecret", shown)
        self.assertNotIn("user:supersecret", shown)
        self.assertIn("example.test:8443", shown)
        self.assertIn("api-version=2024-01-01", shown)
        self.assertIn("***", shown)

    def test_known_secret_in_the_path_is_hidden(self):
        secret = "sk-live-abcdef"
        shown = redact_endpoint(f"https://example.test/{secret}/v1", secret)
        self.assertNotIn(secret, shown)
        self.assertIn("example.test", shown)
        self.assertIn("***", shown)

    def test_short_token_is_not_stripped_from_the_host(self):
        url = "https://api.openai.com/v1"
        self.assertEqual(redact_endpoint(url, "v1"), url)
        self.assertEqual(redact_endpoint(url, "api"), url)

    def test_missing_url(self):
        self.assertEqual(redact_endpoint(None), "(not set)")
        self.assertEqual(redact_endpoint("  "), "(not set)")


class BannerAndStatusTests(unittest.TestCase):
    def test_plain_banner_lists_provider_model_and_endpoint(self):
        lines = startup_banner(
            Style(False),
            version="1.0.0",
            provider="openai",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1",
        )
        text = "\n".join(lines)
        self.assertIn("anyllm 1.0.0", text)
        self.assertIn("openai/gpt-4o-mini", text)
        self.assertIn("https://api.openai.com/v1", text)
        self.assertIn("Type /help for commands, /exit to quit.", text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("─", text)

    def test_color_banner_keeps_the_same_facts(self):
        text = "\n".join(
            startup_banner(
                Style(True),
                version="1.0.0",
                provider="openai",
                model="gpt-4o-mini",
                endpoint="https://api.openai.com/v1",
            )
        )
        self.assertIn("anyllm", text)
        self.assertIn("1.0.0", text)
        self.assertIn("openai", text)
        self.assertIn("gpt-4o-mini", text)
        self.assertIn("https://api.openai.com/v1", text)
        self.assertIn("\x1b[", text)

    def test_status_includes_tokens_only_when_usage_exists(self):
        usage = Usage(input_tokens=7, output_tokens=3, cached_tokens=1, reasoning_tokens=2)
        self.assertEqual(
            format_reply_status(Style(False), 1.0, usage),
            "  1.0s | 7 in | 3 out | 2 reasoning | 1 cached",
        )
        self.assertEqual(format_reply_status(Style(False), 0.4, Usage()), "  0.4s")
        colored = format_reply_status(Style(True), 1.0, usage)
        self.assertIn("1.0s · 7 in · 3 out · 2 reasoning · 1 cached", colored)
        self.assertIn("\x1b[", colored)

    def test_plain_table_is_aligned_ascii(self):
        lines = format_table(
            Style(False),
            ["PROVIDER", "KIND", "KEY"],
            [["openai", "openai", "saved"], ["ollama", "openai", "not needed"]],
        )
        text = "\n".join(lines)
        self.assertTrue(lines[0].startswith("PROVIDER"))
        self.assertIn("KIND", lines[0])
        self.assertIn("openai", text)
        self.assertIn("not needed", text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("─", text)

    def test_color_table_accents_names_and_status(self):
        lines = format_table(
            Style(True),
            ["PROVIDER", "KEY"],
            [["openai", "saved"], ["lab", "missing"]],
        )
        text = "\n".join(lines)
        self.assertIn("openai", text)
        self.assertIn("saved", text)
        self.assertIn("missing", text)
        self.assertIn("\x1b[", text)
        self.assertIn("─", text)


class ErrorFormatTests(unittest.TestCase):
    def test_plain_error_preserves_message_and_hint(self):
        exc = ConfigError(
            "No API key found for openai.",
            hint="Run `anyllm keys set openai`, or set OPENAI_API_KEY.",
        )
        self.assertEqual(
            format_error(Style(False), exc),
            [
                "error: No API key found for openai.",
                "  Run `anyllm keys set openai`, or set OPENAI_API_KEY.",
            ],
        )

    def test_color_error_keeps_the_same_text(self):
        exc = AnyLLMError("authentication failed: bad key", hint="Check the key.")
        text = "\n".join(format_error(Style(True), exc))
        self.assertIn("error: authentication failed: bad key", text)
        self.assertIn("  Check the key.", text)
        self.assertIn("\x1b[", text)

    def test_rate_limit_and_network_use_the_warning_color(self):
        for exc in (
            RateLimitError("rate limited: slow down", hint="Wait and retry."),
            NetworkError("could not reach example.test", hint="Check the base URL."),
        ):
            text = "\n".join(format_error(Style(True), exc))
            self.assertIn("error: " + exc.message, text)
            self.assertIn("  " + exc.hint, text)
            self.assertIn("\x1b[33m", text)


class StreamPrinterTests(unittest.TestCase):
    def test_plain_stream_is_the_answer_only(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            printer = StreamPrinter(Style(False), spinner=False)
            printer.begin()
            printer.reasoning("secret thought")
            printer.text("Hello from mock")
            printer.end()
        self.assertEqual(buf.getvalue(), "Hello from mock\n")

    def test_reasoning_is_labeled_and_separate_from_the_answer(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            printer = StreamPrinter(Style(False), show_reasoning=True, spinner=False)
            printer.begin()
            printer.reasoning("hmm")
            printer.text("yes")
            printer.end()
        self.assertEqual(buf.getvalue(), "reasoning:\nhmm\nyes\n")

    def test_color_reasoning_is_dim_italic_and_the_answer_is_not(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            printer = StreamPrinter(Style(True), show_reasoning=True, spinner=False)
            printer.reasoning("hidden thought")
            printer.text("visible")
            printer.end()
        raw = buf.getvalue()
        self.assertIn("reasoning:\n", raw)
        self.assertIn("\x1b[2;3mhidden thought" + RESET, raw)
        self.assertIn("visible\n", raw)
        self.assertNotIn("\x1b[2;3mvisible", raw)

    def test_code_fences_are_grey_only_when_color_is_on(self):
        plain = io.StringIO()
        with redirect_stdout(plain):
            printer = StreamPrinter(Style(False), spinner=False)
            printer.text("say ```code``` now")
            printer.end()
        self.assertEqual(plain.getvalue(), "say ```code``` now\n")

        colored = io.StringIO()
        with redirect_stdout(colored):
            printer = StreamPrinter(Style(True), spinner=False)
            printer.text("say ```code``` now")
            printer.end()
        raw = colored.getvalue()
        self.assertIn(GREY + "code" + RESET, raw)
        self.assertIn("say ", raw)
        self.assertTrue(raw.endswith(" now\n"))

    def test_spinner_is_cleared_before_the_first_token(self):
        buf = io.StringIO()
        printer = StreamPrinter(Style(True), spinner=True)
        try:
            with redirect_stdout(buf):
                printer.begin()
                time.sleep(0.2)
                printer.text("Hello")
                printer.end()
        finally:
            printer._clear_indicator()
        raw = buf.getvalue()
        self.assertTrue(raw.endswith("\x1b[2KHello\n"), repr(raw))
        self.assertNotIn("Hello", raw[: raw.rfind("\x1b[2K")])

    def test_ascii_spinner_leaves_no_escape_and_no_leftover_glyph(self):
        buf = io.StringIO()
        printer = StreamPrinter(Style(False), spinner=True)
        try:
            with redirect_stdout(buf):
                printer.begin()
                time.sleep(0.2)
                printer.text("Hello")
                printer.end()
        finally:
            printer._clear_indicator()
        raw = buf.getvalue()
        self.assertNotIn("\x1b", raw)
        self.assertTrue(raw.endswith("Hello\n"), repr(raw))
        self.assertNotIn("Hello", raw[: raw.rfind("Hello")])


class ReplPresentationTests(unittest.TestCase):
    def _repl(self, style, provider, session, home):
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}):
            return Repl(session, style, {"mock": provider}, KeyStore(), load_config())

    def test_banner_info_and_help_hide_secrets_and_stay_plain(self):
        secret = "supersecretvalue"
        provider = Provider(
            name="mock",
            label="Mock",
            kind=OPENAI,
            base_url=(
                f"https://user:{secret}@example.test/v1"
                f"?api_key={secret}&api-version=2024-01-01"
            ),
        )
        home = new_temp_dir(self)
        session = Session(provider, "demo-model", api_key=secret)
        repl = self._repl(Style(False), provider, session, home)
        lines = iter(["/info", "/help", "/keys", "/exit"])
        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}),
            mock.patch("builtins.input", side_effect=lambda prompt="": next(lines)),
            redirect_stdout(buf),
        ):
            self.assertEqual(repl.run(), 0)
        text = buf.getvalue()
        self.assertNotIn(secret, text)
        self.assertNotIn("user:" + secret, text)
        self.assertIn("anyllm ", text)
        self.assertIn("mock/demo-model", text)
        self.assertIn("example.test", text)
        self.assertIn("api-version=2024-01-01", text)
        self.assertIn("***", text)
        self.assertIn("provider:  mock", text)
        self.assertIn("Commands", text)
        self.assertIn("/provider [NAME]", text)
        self.assertIn("/exit", text)
        self.assertIn("bye", text)
        self.assertEqual(repl._prompt(), "you> ")
        self.assertNotIn("\x1b", text)

    def test_color_prompt_names_the_user(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url="https://example.test/v1"
        )
        home = new_temp_dir(self)
        session = Session(provider, "demo-model", api_key="supersecretvalue")
        repl = self._repl(Style(True), provider, session, home)
        prompt = repl._prompt()
        self.assertIn("you> ", prompt)
        self.assertNotIn("supersecretvalue", prompt)


class ReplStreamTests(ServerTestCase):
    def test_reply_status_shows_time_and_tokens_without_color(self):
        provider = Provider(
            name="mock", label="Mock", kind=OPENAI, base_url=f"{self.base}/openai/v1"
        )
        home = self.temp_home()
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}):
            session = Session(provider, "mock", api_key=TEST_KEY)
            repl = Repl(
                session, Style(False), {"mock": provider}, KeyStore(), load_config()
            )
            lines = iter(["hi", "/exit"])
            buf = io.StringIO()
            with (
                mock.patch("builtins.input", side_effect=lambda prompt="": next(lines)),
                redirect_stdout(buf),
            ):
                self.assertEqual(repl.run(), 0)
        text = buf.getvalue()
        self.assertIn("Hello from mock", text)
        self.assertRegex(text, r"\d+\.\ds \| 7 in \| 3 out \| 1 cached")
        self.assertNotIn("tokens n/a", text)
        self.assertNotIn(TEST_KEY, text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\r", text)


class CliPresentationTests(ServerTestCase):
    def test_piped_ask_is_plain_even_when_color_is_forced(self):
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
            extra_env={
                "OPENAI_API_KEY": TEST_KEY,
                "NO_COLOR": "",
                "FORCE_COLOR": "1",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "Hello from mock\n")
        self.assertNotIn("\x1b", result.stdout)
        self.assertNotIn("\x1b", result.stderr)
        self.assertNotIn(TEST_KEY, result.stdout)
        self.assertNotIn(TEST_KEY, result.stderr)
        # Explicit --base-url still sends the provider key and says so.
        # The warning is one plain line, not color and not the secret.
        self.assertIn("still sends the openai key", result.stderr)
        self.assertEqual(result.stderr.count("\n"), 1)

    def test_piped_ask_no_stream_and_chat_stay_plain(self):
        home = self.temp_home()
        common = [
            "--provider",
            "openai",
            "--base-url",
            f"{self.base}/openai/v1",
            "--model",
            "mock",
        ]
        once = self.run_cli(
            ["ask", *common, "--no-stream", "hi"],
            home,
            extra_env={"OPENAI_API_KEY": TEST_KEY, "NO_COLOR": "", "FORCE_COLOR": "1"},
        )
        self.assertEqual(once.stdout, "Hello from mock\n")
        self.assertNotIn("\x1b", once.stdout)
        self.assertNotIn("\x1b", once.stderr)
        chat = self.run_cli(
            ["chat", *common],
            home,
            input_text="hi\n",
            extra_env={"OPENAI_API_KEY": TEST_KEY, "NO_COLOR": ""},
        )
        self.assertEqual(chat.returncode, 0, chat.stderr)
        self.assertEqual(chat.stdout, "Hello from mock\n")
        self.assertNotIn("\x1b", chat.stdout)
        self.assertNotIn("|", chat.stdout)
        self.assertNotIn("·", chat.stdout)

    def test_no_color_flag_keeps_errors_and_hints_plain(self):
        home = self.temp_home()
        result = self.run_cli(
            ["--no-color", "ask", "--provider", "openai", "hi"],
            home,
            extra_env={"NO_COLOR": "", "FORCE_COLOR": "1"},
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("error: No API key found for openai.", result.stderr)
        self.assertIn("anyllm keys set openai", result.stderr)
        self.assertNotIn("\x1b", result.stderr)
        self.assertNotIn("\x1b", result.stdout)

    def test_providers_list_is_a_plain_table_when_piped(self):
        home = self.temp_home()
        result = self.run_cli(
            ["providers", "list"],
            home,
            extra_env={"NO_COLOR": ""},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PROVIDER", result.stdout)
        self.assertIn("DEFAULT MODEL", result.stdout)
        self.assertIn("openai", result.stdout)
        self.assertIn("anthropic", result.stdout)
        self.assertIn("gpt-4o-mini", result.stdout)
        self.assertNotIn("\x1b", result.stdout)
        self.assertNotIn("─", result.stdout)

    def test_models_pipe_stays_one_name_per_line(self):
        home = self.temp_home()
        result = self.run_cli(
            ["models", "openai", "--base-url", f"{self.base}/openai/v1"],
            home,
            extra_env={"OPENAI_API_KEY": TEST_KEY, "NO_COLOR": "", "FORCE_COLOR": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["mock-large", "mock-small"])
        self.assertNotIn("\x1b", result.stdout)
        self.assertNotIn("MODEL", result.stdout)

    def test_provider_show_redacts_secrets_in_the_endpoint(self):
        home = self.temp_home()
        secret = "supersecretvalue"
        added = self.run_cli(
            [
                "providers",
                "add",
                "lab",
                "--base-url",
                f"https://user:{secret}@example.test/v1?api_key={secret}&api-version=2024-01-01",
                "--no-key",
            ],
            home,
        )
        self.assertEqual(added.returncode, 0, added.stderr)
        self.assertNotIn(secret, added.stdout)
        self.assertIn("api-version=2024-01-01", added.stdout)
        shown = self.run_cli(["providers", "show", "lab"], home)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertNotIn(secret, shown.stdout)
        self.assertIn("example.test", shown.stdout)
        self.assertIn("***", shown.stdout)
        self.assertIn("api-version=2024-01-01", shown.stdout)
        self.assertNotIn("\x1b", shown.stdout)

    def test_ask_json_stays_parseable(self):
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
            extra_env={"OPENAI_API_KEY": TEST_KEY, "NO_COLOR": "", "FORCE_COLOR": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("\x1b", result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["text"], "Hello from mock")
        self.assertEqual(payload["usage"]["total"], 10)


if __name__ == "__main__":
    unittest.main()
