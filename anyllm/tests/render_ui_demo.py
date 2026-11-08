"""Print a reproducible plain and color transcript of the AnyLLM terminal UI.

Run from the ``anyllm`` directory:

    python3 tests/render_ui_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.adapters import Usage
from anyllm.errors import ConfigError
from anyllm.repl import StreamPrinter
from anyllm.term import Style, format_error, format_reply_status, format_table, startup_banner


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _banner(style: Style) -> None:
    for line in startup_banner(
        style,
        version="1.0.0",
        provider="openai",
        model="gpt-4o-mini",
        endpoint="https://api.openai.com/v1",
    ):
        print(line)


def _reply(style: Style) -> None:
    print(style.prompt("you> ") + "why is the sky blue?")
    printer = StreamPrinter(style, show_reasoning=True, spinner=False)
    printer.reasoning("Sunlight scatters more at short wavelengths.")
    printer.text("Blue light is scattered toward your eyes.")
    printer.end()
    usage = Usage(input_tokens=20, output_tokens=14, reasoning_tokens=6)
    print(format_reply_status(style, 1.1, usage))
    print()
    print(style.prompt("you> ") + "/exit")
    print(style.muted("bye"))


def _tables(style: Style) -> None:
    print(style.bold("Commands"))
    rows = [
        ("/help", "show this list"),
        ("/model NAME", "switch model"),
        ("/provider [NAME]", "switch provider, keeping the conversation"),
        ("/reasoning on|off", "show or hide the model's thinking stream"),
        ("/exit", "quit"),
    ]
    for line in format_table(style, ["COMMAND", "ACTION"], rows):
        print("  " + line)
    print()
    for line in format_table(
        style,
        ["PROVIDER", "KIND", "KEY", "DEFAULT MODEL"],
        [
            ("openai", "openai", "saved", "gpt-4o-mini"),
            ("anthropic", "anthropic", "missing", "claude-sonnet-4-5"),
            ("ollama", "openai", "not needed", "llama3.2"),
            ("lab *", "openai", "saved", "my-model"),
        ],
    ):
        print(line)
    print()
    for line in format_table(
        style,
        ["NAME", "TARGET", "SIZE", "UPDATED"],
        [("notes", "openai/gpt-4o-mini", "4 msg", "2026-11-08 09:15")],
    ):
        print(line)


def _errors(style: Style) -> None:
    missing = ConfigError(
        "No API key found for openai.",
        hint="Run `anyllm keys set openai`, or set OPENAI_API_KEY.",
    )
    for line in format_error(style, missing):
        print(line)


def main() -> None:
    print("AnyLLM terminal UI capture")
    print("Regenerate: python3 tests/render_ui_demo.py")
    print("Plain mode is what NO_COLOR, a pipe, or an old Windows console shows.")
    _section("Plain session")
    _banner(Style(False))
    _reply(Style(False))
    _section("Plain tables, help, and error")
    _tables(Style(False))
    print()
    _errors(Style(False))
    _section("Color session (ANSI; cat this file in a terminal to see it)")
    _banner(Style(True))
    _reply(Style(True))
    _section("Color tables and error")
    _tables(Style(True))
    print()
    _errors(Style(True))
    print()


if __name__ == "__main__":
    main()
