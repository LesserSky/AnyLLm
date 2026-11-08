"""Interactive chat REPL and streaming terminal rendering."""

from __future__ import annotations

import os
import sys
from datetime import datetime

from . import __version__
from .adapters import Usage, list_models
from .catalog import get_provider
from .config import ensure_config_dir, history_path, validate_max_tokens, validate_temperature
from .errors import AnyLLMError, ConfigError
from .keystore import KeyStore, resolve_key
from .targets import (
    describe_live_key,
    missing_endpoint_key,
    rebind_session_credentials,
    resolve_endpoint_key,
    saved_target_alias,
)
from .seam import HELP_ROWS, REPL_HANDLERS, install_repl_features
from .session import (
    Session,
    Turn,
    finish_notice,
    latest_session_name,
    list_sessions,
    load_session_data,
    save_session,
    session_base_url,
)


class StreamPrinter:
    """Render streamed text, with light styling for fenced code blocks."""

    def __init__(self, style, show_reasoning: bool = False, spinner: bool = True) -> None:
        self.style = style
        self.show_reasoning = show_reasoning
        self.spinner = spinner and style.enabled
        self.in_code = False
        self.wrote_text = False
        self.wrote_reasoning = False
        self._indicator = False
        self._last = ""

    def begin(self) -> None:
        if self.spinner:
            sys.stdout.write(self.style.dim("…"))
            sys.stdout.flush()
            self._indicator = True

    def _clear_indicator(self) -> None:
        if self._indicator:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
            self._indicator = False

    def text(self, value: str) -> None:
        if not value:
            return
        self._clear_indicator()
        if self.show_reasoning and self.wrote_reasoning and not self.wrote_text:
            if self._last and self._last != "\n":
                sys.stdout.write("\n")
        if self.style.enabled:
            self._write_styled(value)
        else:
            sys.stdout.write(value)
            sys.stdout.flush()
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
            sys.stdout.write(self.style.dim("thinking\n"))
        sys.stdout.write(self.style.dim(value))
        sys.stdout.flush()
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
            value = self.style.grey(value)
        sys.stdout.write(value)
        sys.stdout.flush()

    def end(self) -> None:
        self._clear_indicator()
        if self._last and self._last != "\n":
            sys.stdout.write("\n")
            sys.stdout.flush()


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
        install_repl_features()
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
        if self.style.enabled:
            return self.style.bold(self.style.cyan("you")) + self.style.cyan(" › ")
        return "you> "

    def _read_input(self) -> str:
        prompt = self._prompt()
        line = input(prompt)
        while line.rstrip().endswith("\\"):
            line = line.rstrip()[:-1] + "\n"
            continuation = self.style.dim("... ") if self.style.enabled else "... "
            line += input(continuation)
        return line

    # -- output helpers -----------------------------------------------------

    def say(self, text: str = "") -> None:
        print(text)

    def error(self, exc: AnyLLMError) -> None:
        print(self.style.red("error: ") + exc.message)
        if exc.hint:
            print(self.style.dim("  " + exc.hint))

    def _banner(self) -> None:
        title = self.style.bold("AnyLLM") + self.style.dim(
            f" {__version__}  ·  {self.session.provider.name}/{self.session.model}"
        )
        print()
        print(title)
        print(self.style.dim("Type /help for commands, /exit to quit."))
        print()

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
        rebind_session_credentials(self.session, self.providers, self.store)
        printer = StreamPrinter(
            self.style,
            show_reasoning=self.session.show_reasoning,
            spinner=True,
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
        pieces = [f"{turn.elapsed:.1f}s"]
        usage = turn.usage
        if usage.input_tokens is not None:
            pieces.append(f"{usage.input_tokens} in")
        if usage.output_tokens is not None:
            pieces.append(f"{usage.output_tokens} out")
        if usage.reasoning_tokens:
            pieces.append(f"{usage.reasoning_tokens} reasoning")
        if usage.total is None:
            pieces.append("tokens n/a")
        return self.style.dim("  " + " · ".join(pieces))

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
        extra = None
        if handler is None:
            extra = REPL_HANDLERS.get(name)
            if extra is None:
                print(self.style.yellow(f"unknown command /{name}"))
                print(self.style.dim("  /help lists commands; //text sends a message starting with a slash"))
                return
        try:
            if extra is not None:
                extra(self, rest)
            else:
                handler(rest)
        except AnyLLMError as exc:
            self.error(exc)
        except (ValueError, TypeError) as exc:
            self.error(ConfigError(str(exc)))

    def _cmd_help(self, _rest: str) -> None:
        rows = [
            ("/model NAME", "switch model"),
            ("/models [TEXT]", "list models available to this key"),
            ("/provider [NAME]", "switch provider, keeping the conversation"),
            ("/system [TEXT]", "show or set the system prompt ('-' clears)"),
            ("/temp [VALUE]", "show or set temperature ('off' clears)"),
            ("/max [N]", "show or set maximum response tokens ('off' clears)"),
            ("/reasoning on|off", "show or hide the model's thinking stream"),
            ("/clear", "start a new conversation"),
            ("/usage [--cost]", "session totals; --cost estimates from saved rates"),
            ("/info", "connection details"),
            ("/save [NAME]", "save this conversation"),
            ("/load [NAME]", "load a saved conversation (latest by default)"),
            ("/sessions", "list saved conversations"),
            ("/keys", "show which key this provider uses"),
            ("/exit", "quit"),
        ]
        rows.extend(HELP_ROWS)
        print(self.style.bold("Commands"))
        for command, description in rows:
            print("  " + self.style.cyan(command.ljust(20)) + description)
        print()
        print(self.style.dim("  End a line with \\ to continue on the next line."))
        print()

    def _cmd_model(self, rest: str) -> None:
        if not rest:
            print(f"model: {self.style.bold(self.session.model)}")
            print(self.style.dim("  change it with /model NAME, or list with /models"))
            return
        self.session.model = rest
        print(self.style.dim(f"model → {rest} (this conversation only)"))

    def _cmd_models(self, rest: str) -> None:
        rebind_session_credentials(self.session, self.providers, self.store)
        print(self.style.dim("fetching models…"))
        models = list_models(
            self.session.provider,
            key=self.session.api_key,
            base_url=self.session.base_url,
            timeout=min(self.session.timeout, 60.0),
        )
        if rest:
            needle = rest.lower()
            models = [name for name in models if needle in name.lower()]
        if not models:
            print(self.style.yellow("no matching models were returned"))
            return
        print(self.style.bold(f"Models ({len(models)})"))
        for name in models[:200]:
            print("  " + name)
        if len(models) > 200:
            print(self.style.dim(f"  … and {len(models) - 200} more"))
        print()

    def _cmd_provider(self, rest: str) -> None:
        if not rest:
            print(f"provider: {self.style.bold(self.session.provider.name)}")
            print(self.style.dim("  available:"))
            for name in sorted(self.providers):
                provider = self.providers[name]
                key = resolve_key(provider, self.store)
                if not provider.key_required:
                    status = "no key needed"
                elif key.value:
                    status = key.display
                else:
                    status = "no key"
                marker = "*" if name == self.session.provider.name else " "
                print(f"  {marker} {name.ljust(12)} {provider.kind.ljust(10)} {status}")
            return

        provider = get_provider(rest, self.providers)
        switching = provider.name != self.session.provider.name
        # A provider with no endpoint (the built-in "custom" provider) cannot
        # be selected from here. Bail out before the session changes.
        if not provider.base_url and (switching or not self.session.base_url):
            raise ConfigError(
                f"Provider {provider.name!r} has no base URL.",
                hint="Add a named endpoint with `anyllm providers add NAME --base-url URL`, "
                "or start the chat with --base-url. The current provider was left unchanged.",
            )
        old_model = self.session.model
        if switching:
            key = resolve_key(provider, self.store)
            if provider.key_required and not key.value:
                raise ConfigError(
                    f"No API key for {provider.name}.",
                    hint=f"Run `anyllm keys set {provider.name}`, then /provider {provider.name} again.",
                )
            self.session.provider = provider
            self.session.api_key = key.value
            self.session.key_source = key.source
            self.session.target_alias = None
            self.session.explicit_endpoint = None
        else:
            # Same provider keeps the current host and target identity.
            # Re-resolve with that identity so a provider key cannot replace a target key.
            self.session.provider = provider
            rebind_session_credentials(self.session, self.providers, self.store)
        if switching:
            # The previous endpoint and model belonged to the previous provider.
            # Keep history, but do not keep a stale model when the new provider
            # has no default of its own.
            self.session.base_url = provider.base_url
            self.session.model = provider.default_model or ""
        shown = self.session.model or "(unset)"
        print(
            self.style.dim(
                f"provider → {provider.name} · model {shown} (history kept; /clear to reset)"
            )
        )
        if not self.session.model:
            print(
                self.style.dim(
                    f"  no default model for {provider.name}; set one with /model NAME"
                )
            )
            if switching and old_model:
                print(self.style.dim(f"  cleared the previous model {old_model}"))
        elif self.session.model != old_model:
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

    def _cmd_usage(self, rest: str) -> None:
        if rest.strip() == "--cost":
            from .budgets import repl_usage_cost

            repl_usage_cost(self)
            return
        if rest.strip():
            raise ConfigError("Usage: /usage or /usage --cost")
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
        print(f"  provider:  {provider.name} ({provider.label})")
        print(f"  model:     {session.model}")
        print(f"  endpoint:  {session.base_url}")
        print(f"  key:       {describe_live_key(session, self.providers, self.store)}")
        print(f"  system:    {session.system or '(none)'}")
        print(f"  messages:  {len(session.messages)}")
        print(f"  timeout:   {session.timeout:g}s")
        print()

    def _cmd_keys(self, _rest: str) -> None:
        provider = self.session.provider
        label = describe_live_key(self.session, self.providers, self.store)
        if label == "not needed":
            print(self.style.dim(f"{provider.name} does not need an API key"))
            return
        if label.startswith("withheld"):
            print(self.style.yellow(f"{provider.name}: provider key withheld for this host"))
            alias = getattr(self.session, "target_alias", None)
            print(self.style.dim("  " + missing_endpoint_key(provider, self.session.base_url, alias).hint))
            return
        print(f"{provider.name}: {label}")

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
        model = data.get("model") or provider.default_model
        if not isinstance(model, str) or not model.strip():
            raise ConfigError(f"Session {name!r} has no model.")
        # Validate before replacing the live session so a bad file leaves it.
        temperature = validate_temperature(data.get("temperature"))
        max_tokens = validate_max_tokens(data.get("max_tokens"))
        base_url = session_base_url(data, provider)
        alias = saved_target_alias(data)
        credential, source = resolve_endpoint_key(
            provider,
            base_url,
            target_alias=alias,
            store=self.store,
            providers=self.providers,
        )
        if provider.key_required and not credential:
            raise missing_endpoint_key(provider, base_url, alias)
        self.session = Session(
            provider,
            model,
            base_url=base_url,
            api_key=credential,
            key_source=source,
            system=data.get("system"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.session.timeout,
            show_reasoning=bool(data.get("show_reasoning")),
            messages=data.get("messages") or [],
            usage=Usage.from_dict(data.get("usage")),
        )
        self.session.target_alias = alias
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
        for entry in entries[:20]:
            print(
                "  "
                + self.style.cyan(entry["name"].ljust(22))
                + f"{entry['provider']}/{entry['model']}  "
                + self.style.dim(f"{entry['messages']} msg · {_format_when(entry['updated'])}")
            )
        print()

    def _cmd_exit(self, _rest: str) -> None:
        self.running = False
        print(self.style.dim("bye"))
