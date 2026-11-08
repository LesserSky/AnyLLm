"""Terminal theme: UTF-8 output, ANSI color, and plain fallback.

Color is used only for an interactive terminal that can render ANSI. Output
stays plain when ``--no-color`` is set, ``NO_COLOR`` is non-empty, ``TERM`` is
``dumb``, stdout is not a TTY, or a Windows console cannot enable virtual
terminal processing. ``FORCE_COLOR`` overrides the TTY / dumb-terminal checks
but not ``NO_COLOR`` or ``--no-color``.

The palette is the standard 16 colors plus dim and italic, so it does not
depend on 256-color or truecolor support:

- accent cyan: names, prompt, table keys
- bold cyan: the product name
- dim: endpoints, hints, timing
- dim italic: model reasoning, kept distinct from the answer
- bright black: fenced code
- red / yellow / green: errors, warnings, success
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.parse
from collections.abc import Sequence

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
GREY = "\x1b[90m"

# Combined SGR codes. Nested wraps are avoided because each wrap emits RESET.
ACCENT = "\x1b[36m"
ACCENT_BOLD = "\x1b[1;36m"
ERROR = "\x1b[1;31m"
HINT = "\x1b[2;33m"
REASONING = "\x1b[2;3m"

# ENABLE_VIRTUAL_TERMINAL_PROCESSING
_ENABLE_VT = 0x0004
_VT_READY: bool | None = None

# Query keys whose values are credentials. ``api-version`` is not in this set.
_SECRET_QUERY = {
    "key",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "secret",
    "password",
    "auth",
    "authorization",
    "sig",
    "signature",
    "client_secret",
}

_STATUS_ROLES = {
    "missing": "warn",
    "no key": "warn",
    "saved": "ok",
    "not needed": "muted",
    "no key needed": "muted",
}

_WARN_ERRORS = {"RateLimitError", "NetworkError", "UsageError"}


def _enable_windows_vt() -> bool:
    """Turn on ANSI processing. Return False when this console cannot do it."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        supported = False
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue
            updated = mode.value | _ENABLE_VT
            if not kernel32.SetConsoleMode(handle, updated):
                continue
            check = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(check)) and (
                check.value & _ENABLE_VT
            ):
                supported = True
        return supported
    except Exception:
        # Styling is cosmetic; never fail because a console is unusual.
        return False


def _windows_vt_supported() -> bool:
    """Cache the VT probe. On non-Windows platforms ANSI is not a console flag."""
    global _VT_READY
    if _VT_READY is None:
        _VT_READY = _enable_windows_vt()
    return _VT_READY


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
    _windows_vt_supported()
    _reconfigure_streams()


def is_tty(stream=None) -> bool:
    """Return whether ``stream`` is an interactive terminal."""
    target = sys.stdout if stream is None else stream
    try:
        return bool(target.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _force_color() -> bool:
    force = os.environ.get("FORCE_COLOR")
    if force is None:
        return False
    return force.strip().lower() not in ("", "0", "false", "no", "off")


def color_enabled(no_color: bool = False, stream=None) -> bool:
    """Decide whether ANSI styling should be emitted.

    ``NO_COLOR`` follows https://no-color.org/: any non-empty value disables
    color and wins over ``FORCE_COLOR``. An empty ``NO_COLOR`` does not.
    """
    if no_color:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if _force_color():
        return True
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    if os.name == "nt" and not _windows_vt_supported():
        return False
    return is_tty(stream)


def width(default: int = 80) -> int:
    try:
        columns = shutil.get_terminal_size((default, 24)).columns
    except OSError:
        return default
    return columns if columns > 0 else default


def _sensitive_query_key(key: str) -> bool:
    name = key.strip().lower().replace("-", "_")
    if name in _SECRET_QUERY:
        return True
    return name.endswith(("_key", "_token", "_secret", "_password"))


def redact_endpoint(url: str | None, *secrets: str | None) -> str:
    """Show an endpoint without userinfo, key query values, or a known secret.

    ``api-version`` and other non-secret query parameters are preserved.
    Secrets shorter than 8 characters are not stripped as substrings, so a
    short token cannot erase part of the host; sensitive query names are
    still redacted at any length.
    """
    if url is None or not str(url).strip():
        return "(not set)"
    raw = str(url).strip()
    parts = urllib.parse.urlsplit(raw)
    netloc = parts.netloc
    query = parts.query
    changed = False
    if parts.username or parts.password:
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parts.port:
            netloc = f"{host}:{parts.port}"
        else:
            netloc = host
        changed = True
    if query:
        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
        rebuilt: list[tuple[str, str]] = []
        for key, value in pairs:
            if _sensitive_query_key(key):
                rebuilt.append((key, "***"))
                changed = True
            else:
                rebuilt.append((key, value))
        if changed:
            query = urllib.parse.urlencode(rebuilt, safe="*")
    if changed:
        raw = urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))
    for secret in secrets:
        if not secret:
            continue
        token = str(secret)
        if len(token) < 8:
            continue
        if token in raw:
            raw = raw.replace(token, "***")
    return raw


class Style:
    """Apply ANSI styles only when color is enabled."""

    def __init__(self, enabled: bool = True, *, no_color: bool = False) -> None:
        self.enabled = bool(enabled)
        self.no_color = bool(no_color)

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

    def accent(self, text: str) -> str:
        return self._wrap(text, ACCENT)

    def accent_bold(self, text: str) -> str:
        return self._wrap(text, ACCENT_BOLD)

    def muted(self, text: str) -> str:
        return self.dim(text)

    def hint(self, text: str) -> str:
        return self._wrap(text, HINT)

    def reasoning(self, text: str) -> str:
        return self._wrap(text, REASONING)

    def code(self, text: str) -> str:
        return self.grey(text)

    def error(self, text: str) -> str:
        return self._wrap(text, ERROR)

    def ok(self, text: str) -> str:
        return self.green(text)

    def warn(self, text: str) -> str:
        return self.yellow(text)

    def prompt(self, text: str, *, readline: bool = False) -> str:
        """Color a prompt. Mark escapes so GNU readline keeps the cursor aligned."""
        if not self.enabled:
            return text
        if not readline:
            return self.accent_bold(text)
        return f"\001{ACCENT_BOLD}\002{text}\001{RESET}\002"


def startup_banner(
    style: Style,
    *,
    version: str,
    provider: str,
    model: str,
    endpoint: str,
) -> list[str]:
    """Concise startup block: product, provider, model, and endpoint.

    Plain mode is ASCII so redirected logs and old consoles stay readable.
    Color mode adds a short rule and the accent palette.
    """
    endpoint = endpoint or "(not set)"
    if not style.enabled:
        return [
            "",
            f"anyllm {version}",
            f"{provider}/{model}",
            endpoint,
            "Type /help for commands, /exit to quit.",
            "",
        ]
    bar = "─" * 32
    return [
        "",
        style.accent(bar),
        style.accent_bold("anyllm") + style.muted(f"  {version}"),
        style.bold(provider) + style.muted("  ·  ") + style.accent(model),
        style.muted(endpoint),
        style.muted("Type /help for commands, /exit to quit."),
        style.accent(bar),
        "",
    ]


def format_reply_status(style: Style, elapsed: float, usage) -> str:
    """One compact status line. Token counts appear only when usage exists."""
    separator = " · " if style.enabled else " | "
    pieces = [f"{elapsed:.1f}s"]
    if usage is not None and _usage_present(usage):
        if getattr(usage, "input_tokens", None) is not None:
            pieces.append(f"{usage.input_tokens} in")
        if getattr(usage, "output_tokens", None) is not None:
            pieces.append(f"{usage.output_tokens} out")
        reasoning = getattr(usage, "reasoning_tokens", None)
        if reasoning:
            pieces.append(f"{reasoning} reasoning")
        cached = getattr(usage, "cached_tokens", None)
        if cached:
            pieces.append(f"{cached} cached")
    return style.muted("  " + separator.join(pieces))


def _usage_present(usage) -> bool:
    return any(
        getattr(usage, name, None) is not None
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
        )
    )


def format_error(style: Style, exc) -> list[str]:
    """Color an error without changing its message or hint text.

    The first line always contains the contiguous prefix ``error: `` followed
    by ``exc.message``. A hint is a following line indented by two spaces,
    matching the historical plain layout.
    """
    name = type(exc).__name__
    painter = style.warn if name in _WARN_ERRORS else style.error
    message = getattr(exc, "message", None)
    if message is None:
        message = str(exc)
    lines = [painter("error: " + message)]
    hint = getattr(exc, "hint", None)
    if hint:
        lines.append(style.hint("  " + hint))
    return lines


def format_table(
    style: Style,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> list[str]:
    """Aligned text table. A dim rule and accent names appear only in color."""
    header_list = [str(header) for header in headers]
    if not header_list:
        return []
    normalized: list[list[str]] = []
    width_count = len(header_list)
    for row in rows:
        cells = ["" if cell is None else str(cell) for cell in row]
        if len(cells) < width_count:
            cells.extend([""] * (width_count - len(cells)))
        normalized.append(cells[:width_count])
    widths = [len(header) for header in header_list]
    for cells in normalized:
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], len(cell))

    def paint(index: int, raw: str, shown: str) -> str:
        if not style.enabled:
            return shown
        if index == 0:
            return style.accent(shown)
        role = _STATUS_ROLES.get(raw)
        if role == "warn":
            return style.warn(shown)
        if role == "ok":
            return style.ok(shown)
        if role == "muted":
            return style.muted(shown)
        return shown

    header_shown = [header_list[i].ljust(widths[i]) for i in range(width_count)]
    lines = [style.bold("  ".join(header_shown).rstrip())]
    if style.enabled and normalized:
        lines.append(style.muted("  ".join("─" * column for column in widths)))
    for cells in normalized:
        last = len(cells) - 1
        painted: list[str] = []
        for index, cell in enumerate(cells):
            shown = cell if index == last else cell.ljust(widths[index])
            painted.append(paint(index, cell, shown))
        lines.append("  ".join(painted).rstrip())
    return lines
