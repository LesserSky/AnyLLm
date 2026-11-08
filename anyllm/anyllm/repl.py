"""Interactive chat REPL and streaming terminal rendering."""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime

from . import __version__
from .adapters import Usage, list_models
from .catalog import get_provider
from .config import ensure_config_dir, history_path
from .errors import AnyLLMError, ConfigError
from .keystore import KeyStore, mask_key, resolve_key
from .session import (
    Session,
    Turn,
    finish_notice,
    latest_session_name,
    list_sessions,
    load_session_data,
    save_session,
)
from .term import (
    format_error,
    format_reply_status,
    format_table,
    is_tty,
    redact_endpoint,
    startup_banner,
)


class StreamPrinter:
    """Render streamed text, with light styling for fenced code blocks.

    A spinner runs until the first token when ``spinner`` is true. Callers
    leave it off for pipes so stdout stays the answer text alone.
    """

    def __init__(self, style, show_reasoning: bool = False, spinner: bool = True) -> None:
        self.style = style
        self.show_reasoning = show_reasoning
        self.spinner = bool(spinner)
        self._ansi = bool(style.enabled)
        self.in_code = False
        self.wrote_text = False
        self.wrote_reasoning = False
        self._indicator = False
        self._last = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def begin(self) -> None:
        if not self.spinner or self._thread is not None:
            return
        self._stop.clear()
        self._indicator = True
        self._thread = threading.Thread(
            target=self._spin, name="anyllm-spinner", daemon=True
        )
        self._thread.start()

    def _spin(self) -> None:
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if self._ansi else "|/-\\"
        index = 0
        while not self._stop.is_set():
            frame = frames[index % len(frames)]
            index += 1
            with self._lock:
                if self._stop.is_set():
                    return
                try:
                    if self._ansi:
                        sys.stdout.write("\r" + self.style.accent(frame) + "\x1b[K")
                    else:
                        sys.stdout.write("\r" + frame)
                    sys.stdout.flush()
                except Exception:
                    return
            if self._stop.wait(0.08):
                return

    def _write(self, value: str) -> None:
        with self._lock:
            sys.stdout.write(value)
            sys.stdout.flush()

    def _clear_indicator(self) -> None:
        if not self._indicator and self._thread is None:
            return
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if self._indicator:
            with self._lock:
                try:
                    if self._ansi:
                        sys.stdout.write("\r\x1b[2K")
                    else:
                        sys.stdout.write("\r \r")
                    sys.stdout.flush()
                except Exception:
                    pass
            self._indicator = False

    def text(self, value: str) -> None:
        if not value:
            return
        self._clear_indicator()
        if self.show_reasoning and self.wrote_reasoning and not self.wrote_text:
            if self._last and self._last != "\n":
                self._write("\n")
        if self.style.enabled:
            self._write_styled(value)
        else:
            self._write(value)
        self.wrote_text = True
        self._last = value[-1]

    def reasoning(self, value: str) -> None:
        if not value:
            return
        self._clear_indicator()
        if not self.show_reasoning:
            self.wrote_reasoning = True
            return
        if not self.wrote_reasoning:
            # A plain label, then dim italic text, so thinking is not the answer.
            self._write(self.style.muted("reasoning:\n"))
        self._write(self.style.reasoning(value))
        self.wrote_reasoning = True
        self._last = value[-1]

    def _write_styled(self, value: str) -> None:
        while value:
            index = value.find("```")
            if index < 0:
                self._emit(value)
                return
            self._emit(value[:index])
            self._emit("```", code=True)
            value = value[index + 3 :]
            self.in_code = not self.in_code

    def _emit(self, value: str, code: bool | None = None) -> None:
        if not value:
            return
        in_code = self.in_code if code is None else code
        if in_code:
            value = self.style.code(value)
        self._write(value)

    def end(self) -> None:
        self._clear_indicator()
        if self._last and self._last != "\n":
            self._write("\n")


def _format_when(value: str | None) -> str:
    if not value:
        return "?"
    try:
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone()
        return stamp.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


class Repl:
    def __init__(self, session: Session, style, providers, store: KeyStore, config) -> None:
        self.session = session
        self.style = style
        self.providers = providers
        self.store = store
        self.config = config
        self.running = True
        self._readline = None
        self._setup_readline()

    # -- session plumbing ---------------------------------------------------

    def _setup_readline(self) -> None:
        try:
            import readline
        except ImportError:
            return
        try:
            path = history_path()
            if path.exists():
                readline.read_history_file(str(path))
            readline.set_history_length(500)
            self._readline = readline
        except (OSError, ValueError):
            self._readline = None

    def _save_history(self) -> None:
        if self._readline is None:
            return
        try:
            ensure_config_dir()
            path = history_path()
            self._readline.write_history_file(str(path))
            if os.name != "nt":
                os.chmod(path, 0o600)
        except (OSError, ValueError):
            pass

    def _prompt(self) -> str:
        return self.style.prompt("you> ", readline=self._readline is not None)

    def _read_input(self) -> str:
        prompt = self._prompt()
        line = input(prompt)
        while line.rstrip().endswith("\\"):
            line = line.rstrip()[:-1] + "\n"
            continuation = self.style.prompt("... ", readline=self._readline is not None)
            line += input(continuation)
        return line

    # -- output helpers -----------------------------------------------------

    def say(self, text: str = "") -> None:
        print(text)

    def error(self, exc: AnyLLMError) -> None:
        for line in format_error(self.style, exc):
            print(line)

    def _banner(self) -> None:
        endpoint = redact_endpoint(self.session.base_url, self.session.api_key)
        for line in startup_banner(
            self.style,
            version=__version__,
            provider=self.session.provider.name,
            model=self.session.model,
            endpoint=endpoint,
        ):
            print(line)

    # -- main loop ----------------------------------------------------------

    def run(self) -> int:
        self._banner()
        try:
            while self.running:
                try:
                    line = self._read_input()
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print()
                    continue
                if not line.strip():
                    continue
                if line.startswith("//"):
                    self._send(line[1:])
                    continue
                if line.startswith("/"):
                    self._command(line)
                    continue
                self._send(line)
        finally:
            self._save_history()
        return 0

    def _send(self, text: str) -> None:
        printer = StreamPrinter(
            self.style,
            show_reasoning=self.session.show_reasoning,
            spinner=is_tty(),
        )
        printer.begin()
        try:
            turn = self.session.send(
                text, on_text=printer.text, on_reasoning=printer.reasoning
            )
        except KeyboardInterrupt:
            printer.end()
            print(self.style.dim("(interrupted)"))
            return
        except AnyLLMError as exc:
            printer.end()
            self.error(exc)
            return
        printer.end()
        print(self._stats(turn))
        notice = finish_notice(turn.finish_reason)
        if notice:
            print(self.style.yellow("! " + notice))
        print()

    def _stats(self, turn: Turn) -> str:
        return format_reply_status(self.style, turn.elapsed, turn.usage)

    # -- commands -----------------------------------------------------------

    def _command(self, line: str) -> None:
        raw = line[1:].strip()
        if not raw:
            print(self.style.dim("  type /help for the command list"))
            return
        parts = raw.split(None, 1)
        name = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""
        handler = getattr(self, f"_cmd_{name}", None)
        if handler is None:
            print(self.style.yellow(f"unknown command /{name}"))
            print(self.style.dim("  /help lists commands; //text sends a message starting with a slash"))
            return
        try:
            handler(rest)
        except AnyLLMError as exc:
            self.error(exc)
        except (ValueError, TypeError) as exc:
            self.error(ConfigError(str(exc)))

    def _cmd_help(self, _rest: str) -> None:
        rows = [
            ("/help", "show this list"),
            ("/model NAME", "switch model"),
            ("/models [TEXT]", "list models available to this key"),
            ("/provider [NAME]", "switch provider, keeping the conversation"),
            ("/system [TEXT]", "show or set the system prompt ('-' clears)"),
            ("/temp [VALUE]", "show or set temperature ('off' clears)"),
            ("/max [N]", "show or set maximum response tokens ('off' clears)"),
            ("/reasoning on|off", "show or hide the model's thinking stream"),
            ("/clear", "start a new conversation"),
            ("/usage", "session token totals"),
            ("/info", "connection details"),
            ("/save [NAME]", "save this conversation"),
            ("/load [NAME]", "load a saved conversation (latest by default)"),
            ("/sessions", "list saved conversations"),
            ("/keys", "show which key this provider uses"),
            ("/exit", "quit"),
        ]
        print(self.style.bold("Commands"))
        for line in format_table(self.style, ["COMMAND", "ACTION"], rows):
            print("  " + line)
        print()
        print(self.style.muted("  End a line with \\ to continue on the next line."))
        print(self.style.muted("  //text sends a message that starts with a slash."))
        print()

    def _cmd_model(self, rest: str) -> None:
        if not rest:
            print(f"model: {self.style.bold(self.session.model)}")
            print(self.style.dim("  change it with /model NAME, or list with /models"))
            return
        self.session.model = rest
        print(self.style.dim(f"model → {rest} (this conversation only)"))

    def _cmd_models(self, rest: str) -> None:
        key = resolve_key(self.session.provider, self.store)
        if self.session.provider.key_required and not key.value:
            raise ConfigError(
                f"No API key for {self.session.provider.name}.",
                hint=f"Run `anyllm keys set {self.session.provider.name}`.",
            )
        print(self.style.dim("fetching models…"))
        models = list_models(
            self.session.provider,
            key=key.value,
            base_url=self.session.base_url,
            timeout=min(self.session.timeout, 60.0),
        )
        if rest:
            needle = rest.lower()
            models = [name for name in models if needle in name.lower()]
        if not models:
            print(self.style.yellow("no matching models were returned"))
            return
        shown = models[:200]
        print(self.style.bold(f"Models ({len(models)})"))
        for line in format_table(self.style, ["MODEL"], [[name] for name in shown]):
            print(line)
        if len(models) > 200:
            print(self.style.dim(f"  … and {len(models) - 200} more"))
        print()

    def _cmd_provider(self, rest: str) -> None:
        if not rest:
            print(f"provider: {self.style.bold(self.session.provider.name)}")
            print(self.style.dim("  available:"))
            rows = []
            for name in sorted(self.providers):
                provider = self.providers[name]
                key = resolve_key(provider, self.store)
                if not provider.key_required:
                    status = "no key needed"
                elif key.value:
                    status = key.display
                else:
                    status = "no key"
                label = f"{name} *" if name == self.session.provider.name else name
                rows.append((label, provider.kind, status))
            for line in format_table(self.style, ["PROVIDER", "KIND", "KEY"], rows):
                print("  " + line)
            print(self.style.dim("  * current provider"))
            print()
            return

        provider = get_provider(rest, self.providers)
        key = resolve_key(provider, self.store)
        if provider.key_required and not key.value:
            raise ConfigError(
                f"No API key for {provider.name}.",
                hint=f"Run `anyllm keys set {provider.name}`, then /provider {provider.name} again.",
            )
        old_model = self.session.model
        self.session.provider = provider
        self.session.base_url = provider.base_url
        self.session.api_key = key.value
        self.session.key_source = key.source
        if provider.default_model:
            self.session.model = provider.default_model
        print(
            self.style.dim(
                f"provider → {provider.name} · model {self.session.model} (history kept; /clear to reset)"
            )
        )
        if not provider.default_model:
            print(
                self.style.dim(
                    f"  no default model for {provider.name}; set one with /model NAME"
                )
            )
        if self.session.model != old_model:
            print(self.style.dim(f"  model reset from {old_model} to {self.session.model}"))

    def _cmd_system(self, rest: str) -> None:
        if not rest:
            if self.session.system:
                print(self.style.bold("system prompt:"))
                print(self.session.system)
            else:
                print(self.style.dim("no system prompt set"))
            return
        if rest == "-":
            self.session.system = None
            print(self.style.dim("system prompt cleared"))
            return
        self.session.system = rest
        print(self.style.dim("system prompt set (this conversation only)"))

    def _cmd_temp(self, rest: str) -> None:
        if not rest:
            value = self.session.temperature
            print(f"temperature: {value if value is not None else 'provider default'}")
            return
        if rest.lower() in ("off", "none", "-", "default"):
            self.session.temperature = None
            print(self.style.dim("temperature → provider default"))
            return
        value = float(rest)
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        self.session.temperature = value
        print(self.style.dim(f"temperature → {value}"))

    def _cmd_max(self, rest: str) -> None:
        if not rest:
            value = self.session.max_tokens
            print(f"max tokens: {value if value is not None else 'provider default'}")
            return
        if rest.lower() in ("off", "none", "-", "default"):
            self.session.max_tokens = None
            print(self.style.dim("max tokens → provider default"))
            return
        value = int(rest)
        if value <= 0:
            raise ValueError("max tokens must be a positive number")
        self.session.max_tokens = value
        print(self.style.dim(f"max tokens → {value}"))

    def _cmd_reasoning(self, rest: str) -> None:
        if not rest:
            state = "on" if self.session.show_reasoning else "off"
            print(f"reasoning display: {state}")
            return
        lowered = rest.lower()
        if lowered in ("on", "true", "1", "yes"):
            self.session.show_reasoning = True
        elif lowered in ("off", "false", "0", "no"):
            self.session.show_reasoning = False
        else:
            raise ValueError("use /reasoning on or /reasoning off")
        print(self.style.dim(f"reasoning display → {'on' if self.session.show_reasoning else 'off'}"))

    def _cmd_clear(self, _rest: str) -> None:
        self.session.messages = []
        self.session.usage = Usage()
        self.session.turns = 0
        print(self.style.dim("conversation cleared"))

    def _cmd_usage(self, _rest: str) -> None:
        usage = self.session.usage
        if usage.total is None and not self.session.turns:
            print(self.style.dim("no requests in this session yet"))
            return
        print(self.style.bold("Session usage"))
        print(f"  turns:   {self.session.turns}")
        print(f"  input:   {usage.input_tokens if usage.input_tokens is not None else '?'}")
        print(f"  output:  {usage.output_tokens if usage.output_tokens is not None else '?'}")
        if usage.reasoning_tokens:
            print(f"  reasoning: {usage.reasoning_tokens}")
        if usage.cached_tokens:
            print(f"  cached:  {usage.cached_tokens}")
        if usage.total is not None:
            print(f"  total:   {usage.total}")
        print()

    def _cmd_info(self, _rest: str) -> None:
        session = self.session
        provider = session.provider
        print(self.style.bold("Connection"))
        print(f"  provider:  {self.style.bold(provider.name)} ({provider.label})")
        print(f"  model:     {self.style.bold(session.model)}")
        endpoint = redact_endpoint(session.base_url, session.api_key)
        print(f"  endpoint:  {self.style.muted(endpoint)}")
        key = resolve_key(provider, self.store)
        if not provider.key_required:
            key_text = "not needed"
        elif key.stored:
            key_text = f"saved ({mask_key(key.value or '')})"
        elif key.env_var:
            key_text = f"environment ({key.env_var})"
        else:
            key_text = "missing"
        print(f"  key:       {key_text}")
        print(f"  system:    {session.system or '(none)'}")
        print(f"  messages:  {len(session.messages)}")
        print(f"  timeout:   {session.timeout:g}s")
        print()

    def _cmd_keys(self, _rest: str) -> None:
        provider = self.session.provider
        key = resolve_key(provider, self.store)
        if not provider.key_required:
            print(self.style.dim(f"{provider.name} does not need an API key"))
            return
        if key.stored:
            print(f"{provider.name}: saved key {mask_key(key.value or '')}")
        elif key.env_var:
            print(f"{provider.name}: from environment variable {key.env_var}")
        else:
            print(self.style.yellow(f"{provider.name}: no key configured"))
            print(self.style.dim(f"  run: anyllm keys set {provider.name}"))

    def _cmd_save(self, rest: str) -> None:
        if not self.session.messages:
            raise ConfigError("Nothing to save yet.")
        path = save_session(self.session, rest or None)
        print(self.style.dim(f"saved → {path}"))

    def _cmd_load(self, rest: str) -> None:
        name = rest or latest_session_name()
        if not name:
            raise ConfigError(
                "No saved sessions yet.",
                hint="Use /save NAME to save this conversation first.",
            )
        data = load_session_data(name)
        provider = get_provider(str(data["provider"]), self.providers)
        key = resolve_key(provider, self.store)
        if provider.key_required and not key.value:
            raise ConfigError(
                f"No API key for {provider.name}.",
                hint=f"Run `anyllm keys set {provider.name}`, then /load {name} again.",
            )
        model = data.get("model") or provider.default_model
        if not model:
            raise ConfigError(f"Session {name!r} has no model.")
        self.session = Session(
            provider,
            model,
            base_url=provider.base_url,
            api_key=key.value,
            key_source=key.source,
            system=data.get("system"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            timeout=self.session.timeout,
            show_reasoning=bool(data.get("show_reasoning")),
            messages=data.get("messages") or [],
            usage=Usage.from_dict(data.get("usage")),
        )
        print(
            self.style.dim(
                f"loaded {name} · {provider.name}/{model} · {len(self.session.messages)} messages"
            )
        )

    def _cmd_sessions(self, _rest: str) -> None:
        entries = list_sessions()
        if not entries:
            print(self.style.dim("no saved sessions"))
            return
        print(self.style.bold("Saved sessions"))
        rows = [
            [
                entry["name"],
                f"{entry['provider']}/{entry['model']}",
                f"{entry['messages']} msg",
                _format_when(entry["updated"]),
            ]
            for entry in entries[:20]
        ]
        for line in format_table(self.style, ["NAME", "TARGET", "SIZE", "UPDATED"], rows):
            print(line)
        print()

    def _cmd_exit(self, _rest: str) -> None:
        self.running = False
        print(self.style.dim("bye"))
