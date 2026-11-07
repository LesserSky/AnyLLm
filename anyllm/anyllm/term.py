"""Terminal helpers: UTF-8 output, ANSI color, and Windows console setup."""

from __future__ import annotations

import os
import shutil
import sys

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
GREY = "\x1b[90m"

# ENABLE_VIRTUAL_TERMINAL_PROCESSING
_ENABLE_VT = 0x0004


def _enable_windows_vt() -> None:
    """Turn on ANSI escape processing for the current Windows console."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | _ENABLE_VT)
    except Exception:
        # Styling is cosmetic; never fail because a console is unusual.
        pass


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
            if encoding != "utf8" and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def setup() -> None:
    """Prepare the terminal once at startup."""
    _enable_windows_vt()
    _reconfigure_streams()


def color_enabled(no_color: bool = False, stream=None) -> bool:
    """Decide whether ANSI styling should be emitted."""
    if no_color:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None and force not in ("", "0", "false"):
        return True
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except OSError:
        return default


class Style:
    """Apply ANSI styles only when color is enabled."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)

    def _wrap(self, text: str, code: str) -> str:
        if not self.enabled or not text:
            return text
        return f"{code}{text}{RESET}"

    def bold(self, text: str) -> str:
        return self._wrap(text, BOLD)

    def dim(self, text: str) -> str:
        return self._wrap(text, DIM)

    def red(self, text: str) -> str:
        return self._wrap(text, RED)

    def green(self, text: str) -> str:
        return self._wrap(text, GREEN)

    def yellow(self, text: str) -> str:
        return self._wrap(text, YELLOW)

    def blue(self, text: str) -> str:
        return self._wrap(text, BLUE)

    def magenta(self, text: str) -> str:
        return self._wrap(text, MAGENTA)

    def cyan(self, text: str) -> str:
        return self._wrap(text, CYAN)

    def grey(self, text: str) -> str:
        return self._wrap(text, GREY)
