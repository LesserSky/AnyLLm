"""Command-line interface for AnyLLM."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import sys

from . import __version__
from .adapters import Usage, list_models
from .catalog import (
    KINDS,
    Provider,
    get_provider,
    load_providers,
    normalize_name,
    provider_spec_from_args,
    validate_base_url,
    validate_custom_name,
)
from .config import (
    ALLOWED_KEYS,
    Config,
    config_path,
    keys_path,
    load_config,
    parse_config_value,
)
from .errors import AnyLLMError, ConfigError, UsageError
from .keystore import KeyStore, dpapi_available, mask_key, resolve_key
from .repl import Repl, StreamPrinter
from .session import (
    Session,
    finish_notice,
    latest_session_name,
    load_session_data,
)
from .term import (
    Style,
    color_enabled,
    format_error,
    format_reply_status,
    format_table,
    is_tty,
    redact_endpoint,
    setup as term_setup,
)

EXAMPLES = """\
examples:
  anyllm keys set openai              save an API key (prompted, hidden)
  anyllm chat                         start chatting
  anyllm chat -p anthropic -m claude-haiku-4-5
  anyllm ask "explain git rebase in 3 lines"
  cat error.log | anyllm ask "what is failing here?"
  anyllm ask --json "hi"              machine-readable output
  anyllm models openai                list models for the configured key
  anyllm providers add lab --base-url http://localhost:8000/v1 --model my-model
  anyllm doctor                       check keys and configuration

Provider keys can also come from the environment. AnyLLM checks
ANYLLM_<PROVIDER>_API_KEY, then the provider's usual variable (for example
OPENAI_API_KEY), then ANYLLM_API_KEY, then the saved key.
"""


# -- argument parsing -------------------------------------------------------


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-p", "--provider", metavar="NAME", help="provider name")
    parser.add_argument("-m", "--model", metavar="NAME", help="model name")
    parser.add_argument("-s", "--system", metavar="TEXT", help="system prompt")
    parser.add_argument("--base-url", metavar="URL", help="override the provider's base URL")
    parser.add_argument("--temperature", type=float, metavar="N", help="sampling temperature")
    parser.add_argument("--max-tokens", type=int, metavar="N", help="maximum response tokens")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help="network timeout (default: 120)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="show the model's thinking stream when the provider supports it",
    )
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anyllm",
        description="Chat with any LLM API key from your terminal.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"anyllm {__version__}")
    parser.add_argument(
        "--no-color",
        action="store_true",
        dest="no_color_global",
        help="disable ANSI colors",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    chat = sub.add_parser(
        "chat",
        help="start an interactive conversation",
        description="Start an interactive conversation. Piped stdin is answered once.",
    )
    _add_model_options(chat)
    chat.add_argument("--load", metavar="NAME", help="load a saved conversation")
    chat.add_argument(
        "--continue",
        dest="continue_last",
        action="store_true",
        help="load the most recently saved conversation",
    )
    chat.set_defaults(func=cmd_chat)

    ask = sub.add_parser(
        "ask",
        help="send one prompt and print the answer",
        description="Send one prompt and print the answer. Reads stdin when no prompt is given.",
    )
    _add_model_options(ask)
    ask.add_argument("prompt", nargs="*", help="prompt text; omit to read stdin")
    ask.add_argument("--json", action="store_true", dest="json_output", help="print JSON")
    ask.add_argument("--no-stream", action="store_true", help="wait for the full answer")
    ask.set_defaults(func=cmd_ask)

    models = sub.add_parser(
        "models",
        help="list the models available to your key",
        description="List model names returned by the provider.",
    )
    models.add_argument("provider", nargs="?", help="provider name")
    models.add_argument("--filter", metavar="TEXT", help="only names containing TEXT")
    models.add_argument("--base-url", metavar="URL", help="override the provider's base URL")
    models.add_argument("--json", action="store_true", dest="json_output")
    models.add_argument("--timeout", type=float, default=60.0, metavar="SECONDS")
    models.add_argument("--no-color", action="store_true")
    models.set_defaults(func=cmd_models)

    providers = sub.add_parser(
        "providers",
        help="list or define providers",
        description="List built-in providers or define your own OpenAI-compatible endpoint.",
    )
    providers_sub = providers.add_subparsers(dest="providers_command", required=True)
    providers_list = providers_sub.add_parser("list", help="list providers")
    providers_list.add_argument("--no-color", action="store_true")
    providers_list.set_defaults(func=cmd_providers_list)

    providers_show = providers_sub.add_parser("show", help="show one provider")
    providers_show.add_argument("name")
    providers_show.add_argument("--no-color", action="store_true")
    providers_show.set_defaults(func=cmd_providers_show)

    providers_add = providers_sub.add_parser(
        "add", help="add a custom OpenAI-compatible provider"
    )
    providers_add.add_argument("name", help="short name, e.g. 'lab'")
    providers_add.add_argument("--base-url", required=True, metavar="URL")
    providers_add.add_argument("--kind", choices=KINDS, default="openai")
    providers_add.add_argument("--label", help="display name")
    providers_add.add_argument(
        "--env", action="append", default=[], metavar="VAR", help="environment variable to check"
    )
    providers_add.add_argument(
        "--model", action="append", default=[], metavar="NAME", help="suggested/default model"
    )
    providers_add.add_argument("--no-key", action="store_true", help="endpoint needs no key")
    providers_add.add_argument(
        "--auth-header",
        default=None,
        metavar="HEADER",
        help="header carrying the key (default depends on --kind)",
    )
    providers_add.add_argument(
        "--auth-prefix",
        default=None,
        metavar="TEXT",
        help="prefix before the key, e.g. 'Bearer ' (default depends on --kind)",
    )
    providers_add.add_argument("--no-color", action="store_true")
    providers_add.set_defaults(func=cmd_providers_add)

    providers_remove = providers_sub.add_parser("remove", help="remove a custom provider")
    providers_remove.add_argument("name")
    providers_remove.add_argument("--no-color", action="store_true")
    providers_remove.set_defaults(func=cmd_providers_remove)

    keys = sub.add_parser(
        "keys",
        help="save, list, or remove API keys",
        description="Manage API keys. Saved keys live in a separate file, encrypted "
        "with Windows DPAPI where available.",
    )
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    keys_set = keys_sub.add_parser("set", help="save a key (prompted, hidden)")
    keys_set.add_argument("provider", nargs="?", help="provider name")
    keys_set.add_argument(
        "--stdin", action="store_true", help="read the key from one line of stdin"
    )
    keys_set.add_argument("--no-color", action="store_true")
    keys_set.set_defaults(func=cmd_keys_set)

    keys_list = keys_sub.add_parser("list", help="list saved keys and environment variables")
    keys_list.add_argument("--no-color", action="store_true")
    keys_list.set_defaults(func=cmd_keys_list)

    keys_remove = keys_sub.add_parser("remove", help="remove a saved key")
    keys_remove.add_argument("provider", nargs="?")
    keys_remove.add_argument("--no-color", action="store_true")
    keys_remove.set_defaults(func=cmd_keys_remove)

    keys_path_parser = keys_sub.add_parser("path", help="print the key file location")
    keys_path_parser.add_argument("--no-color", action="store_true")
    keys_path_parser.set_defaults(func=cmd_keys_path)

    config = sub.add_parser(
        "config",
        help="read or change saved defaults",
        description="Read or change defaults stored in the config file.",
    )
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser("show", help="print the config as JSON")
    config_show.add_argument("--no-color", action="store_true")
    config_show.set_defaults(func=cmd_config_show)

    config_get = config_sub.add_parser("get", help="print one value")
    config_get.add_argument("key")
    config_get.add_argument("--no-color", action="store_true")
    config_get.set_defaults(func=cmd_config_get)

    config_set = config_sub.add_parser("set", help="set one value")
    config_set.add_argument("key")
    config_set.add_argument("value")
    config_set.add_argument("--no-color", action="store_true")
    config_set.set_defaults(func=cmd_config_set)

    config_unset = config_sub.add_parser("unset", help="reset one value to its default")
    config_unset.add_argument("key")
    config_unset.add_argument("--no-color", action="store_true")
    config_unset.set_defaults(func=cmd_config_unset)

    config_path_parser = config_sub.add_parser("path", help="print the config location")
    config_path_parser.add_argument("--no-color", action="store_true")
    config_path_parser.set_defaults(func=cmd_config_path)

    doctor = sub.add_parser(
        "doctor",
        help="check keys, endpoints, and saved settings",
        description="Show where keys come from and, with --test, send a tiny request "
        "to each configured provider.",
    )
    doctor.add_argument("--test", action="store_true", help="send a small test request")
    doctor.add_argument(
        "--provider", action="append", default=[], metavar="NAME", help="test only this provider"
    )
    doctor.add_argument("--timeout", type=float, default=30.0, metavar="SECONDS")
    doctor.add_argument("--no-color", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    term_setup()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["chat"]
    parser = build_parser()
    try:
        args = parser.parse_args(arguments)
    except SystemExit as exc:
        return int(exc.code or 0)

    no_color_flag = bool(
        getattr(args, "no_color", False) or getattr(args, "no_color_global", False)
    )
    style = Style(color_enabled(no_color=no_color_flag), no_color=no_color_flag)
    try:
        if not hasattr(args, "func"):
            parser.print_help()
            return 0
        result = args.func(args, style)
        return int(result or 0)
    except UsageError as exc:
        _print_error(style, exc)
        return 2
    except AnyLLMError as exc:
        _print_error(style, exc)
        return 1
    except KeyboardInterrupt:
        print()
        return 130
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except OSError:
            pass
        return 0


def _stderr_style(style: Style) -> Style:
    """Color errors for the terminal even when stdout is a pipe."""
    if style.no_color:
        return Style(False, no_color=True)
    return Style(color_enabled(stream=sys.stderr))


def _print_error(style: Style, exc: AnyLLMError) -> None:
    for line in format_error(_stderr_style(style), exc):
        print(line, file=sys.stderr)


# -- shared resolution ------------------------------------------------------


def _as_float(value, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"Invalid {name} value {value!r}.", hint="Use a number, e.g. 0.7") from None


def _as_int(value, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"Invalid {name} value {value!r}.", hint="Use a whole number, e.g. 1024") from None


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "on", "yes", "1"):
            return True
        if lowered in ("false", "off", "no", "0", ""):
            return False
    raise ConfigError(f"Invalid {name} value {value!r}.", hint="Use true or false.")


def _temperature(value) -> float | None:
    number = _as_float(value, "temperature")
    if number is not None and not 0.0 <= number <= 2.0:
        raise ConfigError(
            f"temperature must be between 0 and 2 (got {number}).",
            hint="Most models are happiest between 0 and 1.",
        )
    return number


def _max_tokens(value) -> int | None:
    number = _as_int(value, "max_tokens")
    if number is not None and number <= 0:
        raise ConfigError(
            f"max_tokens must be greater than 0 (got {number}).",
            hint="Try 1024 or 4096.",
        )
    return number


def _missing_key(provider: Provider) -> ConfigError:
    variable = provider.env_vars[0] if provider.env_vars else f"ANYLLM_{provider.name.upper()}_API_KEY"
    if os.name == "nt":
        example = f'$env:{variable} = "YOUR_KEY"'
    else:
        example = f'export {variable}="YOUR_KEY"'
    return ConfigError(
        f"No API key found for {provider.name}.",
        hint=f"Run `anyllm keys set {provider.name}`, or set {variable} "
        f"(for example: {example}).",
    )


def _pick_provider(
    name: str | None,
    config: Config,
    providers: dict[str, Provider],
    store: KeyStore,
    *,
    allow_auto: bool = True,
) -> Provider:
    if name:
        return get_provider(name, providers)
    configured = config.get("provider")
    if configured:
        return get_provider(str(configured), providers)
    if allow_auto:
        ready: list[Provider] = []
        for provider in providers.values():
            resolution = resolve_key(provider, store)
            # A generic ANYLLM_API_KEY says nothing about which provider to use.
            if resolution.value and resolution.env_var != "ANYLLM_API_KEY":
                ready.append(provider)
        if len(ready) == 1:
            return ready[0]
        if len(ready) > 1:
            names = ", ".join(provider.name for provider in ready)
            raise ConfigError(
                "No provider selected, and several have keys: " + names,
                hint="Choose one with --provider NAME, or run "
                "`anyllm config set provider NAME`.",
            )
    names = ", ".join(sorted(providers))
    raise ConfigError(
        "No provider selected.",
        hint=f"Pass --provider NAME, or run `anyllm config set provider NAME`. "
        f"Available: {names}. Start with `anyllm keys set openai`.",
    )


def _resolve_target(
    args, config: Config, providers: dict[str, Provider], store: KeyStore
) -> tuple[Provider, str, str]:
    provider = _pick_provider(args.provider, config, providers, store)
    base_url = args.base_url or provider.base_url
    if args.base_url:
        base_url = validate_base_url(args.base_url)
    if not base_url:
        if provider.name == "custom":
            raise ConfigError(
                "The 'custom' provider needs an endpoint.",
                hint="Pass --base-url https://..., or add a named provider with "
                "`anyllm providers add NAME --base-url URL`.",
            )
        raise ConfigError(f"Provider {provider.name!r} has no base URL configured.")
    configured_provider = normalize_name(config.get("provider") or "")
    model = args.model
    if not model and configured_provider == provider.name:
        model = config.get("model")
    if not model:
        model = provider.default_model
    if not model:
        raise ConfigError(
            f"No model selected for {provider.name}.",
            hint=f"Pass --model NAME, or run `anyllm models {provider.name}` "
            "to see available models.",
        )
    return provider, str(model), base_url


def _build_session(
    args, config: Config, providers: dict[str, Provider], store: KeyStore
) -> Session:
    provider, model, base_url = _resolve_target(args, config, providers, store)
    key = resolve_key(provider, store)
    if provider.key_required and not key.value:
        raise _missing_key(provider)
    system = args.system if args.system is not None else config.get("system")
    if system is not None and not str(system).strip():
        system = None
    temperature = (
        args.temperature if args.temperature is not None else config.get("temperature")
    )
    max_tokens = args.max_tokens if args.max_tokens is not None else config.get("max_tokens")
    return Session(
        provider,
        model,
        base_url=base_url,
        api_key=key.value,
        key_source=key.source,
        system=None if system is None else str(system),
        temperature=_temperature(temperature),
        max_tokens=_max_tokens(max_tokens),
        timeout=max(1.0, float(args.timeout)),
        show_reasoning=bool(
            args.reasoning or _as_bool(config.get("show_reasoning"), "show_reasoning")
        ),
    )


def _session_from_saved(
    name: str,
    args,
    providers: dict[str, Provider],
    store: KeyStore,
    style: Style,
) -> Session:
    data = load_session_data(name)
    provider = get_provider(str(data.get("provider")), providers)
    key = resolve_key(provider, store)
    if provider.key_required and not key.value:
        raise _missing_key(provider)
    model = args.model or data.get("model") or provider.default_model
    if not model:
        raise ConfigError(
            f"Session {name!r} has no model.",
            hint="Pass --model NAME.",
        )
    system = args.system if args.system is not None else data.get("system")
    if system is not None and not str(system).strip():
        system = None
    temperature = (
        args.temperature if args.temperature is not None else data.get("temperature")
    )
    max_tokens = args.max_tokens if args.max_tokens is not None else data.get("max_tokens")
    session = Session(
        provider,
        str(model),
        base_url=args.base_url or provider.base_url,
        api_key=key.value,
        key_source=key.source,
        system=None if system is None else str(system),
        temperature=_temperature(temperature),
        max_tokens=_max_tokens(max_tokens),
        timeout=max(1.0, float(args.timeout)),
        show_reasoning=bool(
            args.reasoning or _as_bool(data.get("show_reasoning"), "show_reasoning")
        ),
        messages=data.get("messages") or [],
        usage=Usage.from_dict(data.get("usage")),
    )
    print(
        style.dim(
            f"resumed {name} · {provider.name}/{model} · "
            f"{len(session.messages)} messages"
        )
    )
    return session


# -- chat and ask -----------------------------------------------------------


def _choose_provider(providers: dict[str, Provider], style: Style) -> Provider:
    names = [name for name in sorted(providers) if name != "custom"]
    print()
    print(style.bold("Welcome to AnyLLM."))
    print("No provider is configured yet. Choose one:")
    for name in names:
        provider = providers[name]
        print(f"  {style.accent(name.ljust(12))} {provider.label}")
    print(style.dim("  custom endpoints: add one first with `anyllm providers add`"))
    while True:
        try:
            choice = input("Provider [openai]: ").strip().lower() or "openai"
        except EOFError:
            raise ConfigError(
                "Setup cancelled.",
                hint="Run `anyllm keys set openai` when you are ready.",
            ) from None
        if choice == "custom":
            print(
                style.dim(
                    "  custom endpoints need a base URL; use `anyllm providers add` first"
                )
            )
            continue
        try:
            return get_provider(choice, providers)
        except ConfigError as exc:
            print(style.red("  " + exc.message))


def _interactive_setup(
    args,
    config: Config,
    providers: dict[str, Provider],
    store: KeyStore,
    style: Style,
) -> None:
    """Guide a first interactive run through provider and key setup.

    Only runs on a real terminal; scripts and pipes keep the old, fast errors.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    selected = args.provider or config.get("provider")
    if not selected:
        try:
            # If exactly one environment/stored key exists, use it silently.
            _pick_provider(None, config, providers, store)
            return
        except ConfigError:
            pass

    if selected:
        provider = get_provider(str(selected), providers)
    else:
        provider = _choose_provider(providers, style)
        config.set("provider", provider.name)
        config.set("model", None)
        config.save()
        print(style.dim(f"default provider → {provider.name}"))

    if not provider.base_url and not getattr(args, "base_url", None):
        url = input(
            f"Base URL for {provider.name} (for example https://api.example.com/v1): "
        ).strip()
        if not url:
            raise ConfigError(
                "Setup cancelled.",
                hint="Add the endpoint with "
                "`anyllm providers add NAME --base-url URL`.",
            )
        args.base_url = validate_base_url(url)
        print(style.dim("  using that endpoint for this session"))

    key = resolve_key(provider, store)
    if provider.key_required and not key.value:
        print(style.dim(f"No API key found for {provider.name}."))
        try:
            entered = getpass.getpass(
                f"Paste your {provider.label} API key (hidden, blank to cancel): "
            ).strip()
        except EOFError:
            entered = ""
        if not entered:
            raise ConfigError(
                "Setup cancelled.",
                hint=f"Run `anyllm keys set {provider.name}` when you are ready.",
            )
        scheme = store.set(provider.name, entered)
        if scheme == "dpapi":
            detail = "encrypted with Windows DPAPI"
        elif os.name == "nt":
            detail = "saved as plain text"
        else:
            detail = "saved with owner-only permissions"
        print(style.dim(f"saved {provider.name} key ({detail})"))

    configured_model = (
        config.get("model")
        if normalize_name(config.get("provider") or "") == provider.name
        else None
    )
    if not args.model and not configured_model and not provider.default_model:
        entered_model = input(f"Model name for {provider.name}: ").strip()
        if not entered_model:
            raise ConfigError(
                "No model selected.",
                hint=f"Run `anyllm models {provider.name}` to list models.",
            )
        config.set("model", entered_model)
        config.save()
        print(style.dim(f"default model → {entered_model}"))


def cmd_chat(args, style: Style) -> int:
    config = load_config()
    providers = load_providers(config)
    store = KeyStore()

    if args.load:
        session = _session_from_saved(args.load, args, providers, store, style)
    elif args.continue_last:
        name = latest_session_name()
        if not name:
            raise ConfigError(
                "No saved conversations yet.",
                hint="Use /save NAME inside chat, then --continue will find it.",
            )
        session = _session_from_saved(name, args, providers, store, style)
    else:
        _interactive_setup(args, config, providers, store, style)
        session = _build_session(args, config, providers, store)

    if not sys.stdin.isatty():
        prompt = sys.stdin.read()
        if not prompt.strip():
            raise UsageError("No prompt was received on stdin.")
        return _ask(
            session,
            prompt.strip(),
            style,
            json_output=False,
            stream=True,
        )
    return Repl(session, style, providers, store, config).run()


def cmd_ask(args, style: Style) -> int:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        if sys.stdin.isatty():
            raise UsageError(
                "No prompt given.",
                hint='Try: anyllm ask "explain git rebase in three lines"',
            )
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise UsageError("No prompt given.")

    config = load_config()
    providers = load_providers(config)
    store = KeyStore()
    session = _build_session(args, config, providers, store)
    return _ask(
        session,
        prompt,
        style,
        json_output=args.json_output,
        stream=not args.no_stream,
    )


def _ask(
    session: Session,
    prompt: str,
    style: Style,
    *,
    json_output: bool,
    stream: bool,
) -> int:
    if json_output:
        try:
            turn = session.send(prompt)
        except KeyboardInterrupt:
            return 130
        payload = {
            "provider": session.provider.name,
            "model": session.model,
            "text": turn.text,
            "usage": turn.usage.to_dict(),
            "finish_reason": turn.finish_reason,
            "elapsed_seconds": round(turn.elapsed, 3),
            "base_url": session.base_url,
        }
        if session.show_reasoning and turn.reasoning:
            payload["reasoning"] = turn.reasoning
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        _warn_finish(turn, style)
        return 0

    # Piped stdout is the answer alone: no spinner, color, or status line.
    interactive = is_tty(sys.stdout)
    answer_style = style if interactive else Style(False, no_color=True)
    if stream:
        printer = StreamPrinter(
            answer_style,
            show_reasoning=session.show_reasoning,
            spinner=interactive,
        )
        printer.begin()
        try:
            turn = session.send(
                prompt, on_text=printer.text, on_reasoning=printer.reasoning
            )
        except KeyboardInterrupt:
            printer.end()
            return 130
        except BaseException:
            printer.end()
            raise
        printer.end()
    else:
        try:
            turn = session.send(prompt)
        except KeyboardInterrupt:
            return 130
        sys.stdout.write(turn.text)
        if not turn.text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    if interactive:
        print(format_reply_status(style, turn.elapsed, turn.usage), file=sys.stderr)
    _warn_finish(turn, style)
    return 0


def _warn_finish(turn, style: Style) -> None:
    notice = finish_notice(turn.finish_reason)
    if notice:
        print(style.yellow("warning: " + notice), file=sys.stderr)


# -- models -----------------------------------------------------------------


def cmd_models(args, style: Style) -> int:
    config = load_config()
    providers = load_providers(config)
    store = KeyStore()
    provider = _pick_provider(args.provider, config, providers, store)
    key = resolve_key(provider, store)
    if provider.key_required and not key.value:
        raise _missing_key(provider)
    base_url = validate_base_url(args.base_url) if args.base_url else provider.base_url
    models = list_models(
        provider,
        key=key.value,
        base_url=base_url,
        timeout=max(1.0, float(args.timeout)),
    )
    if args.filter:
        needle = args.filter.lower()
        models = [name for name in models if needle in name.lower()]
    if args.json_output:
        print(json.dumps(models, indent=2))
        return 0
    if not models:
        print(style.yellow("no models returned"), file=sys.stderr)
        return 0
    # Scripts and pipes keep one name per line. A terminal gets a small table.
    if not is_tty(sys.stdout):
        for name in models:
            print(name)
        return 0
    for line in format_table(style, ["MODEL"], [[name] for name in models]):
        print(line)
    return 0


# -- keys -------------------------------------------------------------------


def _key_status(provider: Provider, store: KeyStore) -> str:
    if not provider.key_required:
        return "not needed"
    key = resolve_key(provider, store)
    if key.env_var:
        return f"env:{key.env_var}"
    if key.stored:
        return "saved"
    return "missing"


def cmd_keys_set(args, style: Style) -> int:
    config = load_config()
    providers = load_providers(config)
    store = KeyStore()
    name = args.provider or config.get("provider")
    if not name:
        raise UsageError(
            "Choose a provider for the key.",
            hint="For example: anyllm keys set openai. Available: "
            + ", ".join(sorted(providers)),
        )
    provider = get_provider(str(name), providers)
    if args.stdin:
        key = sys.stdin.readline().strip()
    else:
        try:
            key = getpass.getpass(f"API key for {provider.name}: ").strip()
        except EOFError:
            raise UsageError("No key entered; nothing was saved.") from None
    if not key:
        raise UsageError("No key entered; nothing was saved.")

    scheme = store.set(provider.name, key)
    if scheme == "dpapi":
        detail = "encrypted with Windows DPAPI"
    elif os.name == "nt":
        detail = "stored as plain text (DPAPI was unavailable)"
    else:
        detail = "stored in a file with permissions 600"
    print(f"saved {provider.name} key → {keys_path()} ({detail})")

    resolution = resolve_key(provider, store)
    if resolution.env_var:
        print(
            style.yellow(
                f"note: {resolution.env_var} is set and takes precedence over the saved key"
            )
        )
    print(style.dim("check it with: anyllm doctor --test"))
    return 0


def cmd_keys_list(args, style: Style) -> int:
    config = load_config()
    providers = load_providers(config)
    store = KeyStore()
    infos = store.info()
    rows: list[tuple[str, str, str]] = []
    for name in sorted(providers):
        provider = providers[name]
        info = infos.get(name)
        if info:
            try:
                masked = mask_key(store.get(name) or "")
                saved = f"{masked} [{info.get('scheme', '?')}]"
            except AnyLLMError:
                saved = "(unreadable)"
        else:
            saved = "-"
        env_names = [variable for variable in provider.env_vars if os.environ.get(variable)]
        if not provider.key_required:
            env_text = "not needed"
        elif env_names:
            env_text = ", ".join(env_names)
        else:
            env_text = "-"
        rows.append((name, saved, env_text))

    for line in format_table(style, ["PROVIDER", "SAVED", "ENVIRONMENT"], rows):
        print(line)
    print()
    print(style.dim(f"key file: {keys_path()}"))
    print(
        style.dim(
            "environment variables take precedence over saved keys; "
            "ANYLLM_API_KEY is used as a last resort"
        )
    )
    return 0


def cmd_keys_remove(args, style: Style) -> int:
    config = load_config()
    providers = load_providers(config)
    store = KeyStore()
    name = args.provider or config.get("provider")
    if not name:
        raise UsageError("Choose a provider to remove.", hint="anyllm keys remove openai")
    provider = get_provider(str(name), providers)
    removed = store.remove(provider.name)
    if removed:
        print(f"removed the saved {provider.name} key")
    else:
        print(f"no saved key for {provider.name}")
    resolution = resolve_key(provider, store)
    if resolution.env_var:
        print(
            style.yellow(
                f"note: {resolution.env_var} is still set in the environment "
                "and will keep being used"
            )
        )
    return 0


def cmd_keys_path(args, style: Style) -> int:
    print(keys_path())
    return 0


# -- providers --------------------------------------------------------------


def cmd_providers_list(args, style: Style) -> int:
    config = load_config()
    providers = load_providers(config)
    store = KeyStore()
    custom = set((config.get("custom_providers") or {}).keys())
    rows = []
    for name in sorted(providers):
        provider = providers[name]
        label = f"{name} *" if name in custom else name
        rows.append(
            (
                label,
                provider.kind,
                _key_status(provider, store),
                provider.default_model or "-",
            )
        )
    for line in format_table(
        style, ["PROVIDER", "KIND", "KEY", "DEFAULT MODEL"], rows
    ):
        print(line)
    print()
    print(style.dim("* = custom provider; see `anyllm providers show NAME`"))
    return 0


def cmd_providers_show(args, style: Style) -> int:
    config = load_config()
    providers = load_providers(config)
    store = KeyStore()
    provider = get_provider(args.name, providers)
    resolution = resolve_key(provider, store)
    print(style.bold(provider.name))
    print(f"  label:      {provider.label}")
    print(f"  kind:       {provider.kind}")
    print(f"  base URL:   {redact_endpoint(provider.base_url, resolution.value)}")
    print(f"  key needed: {'yes' if provider.key_required else 'no'}")
    print(f"  key source: {_key_status(provider, store)}")
    print(f"  models:     {', '.join(provider.models) if provider.models else '(none suggested)'}")
    if provider.env_vars:
        print(f"  env vars:   {', '.join(provider.env_vars)}")
    if provider.auth_header != "Authorization" or provider.auth_prefix != "Bearer ":
        print(
            f"  auth:       {provider.auth_header} "
            f"(prefix {provider.auth_prefix!r})"
        )
    print()
    return 0


def cmd_providers_add(args, style: Style) -> int:
    name = validate_custom_name(args.name)
    for variable in args.env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable or ""):
            raise UsageError(
                f"Invalid environment variable name {variable!r}.",
                hint="Use letters, digits, and underscores, starting with a letter or underscore.",
            )
    config = load_config()
    spec = provider_spec_from_args(name, args)
    custom = dict(config.get("custom_providers") or {})
    replacing = name in custom
    custom[name] = spec
    config.set("custom_providers", custom)
    config.save()
    action = "updated" if replacing else "added"
    print(f"{action} provider {style.bold(name)} → {redact_endpoint(spec['base_url'])}")
    print(style.dim(f"  kind: {spec['kind']}  key needed: {'yes' if spec['key_required'] else 'no'}"))
    if spec["models"]:
        print(style.dim(f"  default model: {spec['models'][0]}"))
    print()
    print(style.dim(f"try: anyllm chat -p {name}" + (f" -m {spec['models'][0]}" if spec["models"] else " -m MODEL")))
    if spec["key_required"]:
        print(style.dim(f"key: anyllm keys set {name}"))
    return 0


def cmd_providers_remove(args, style: Style) -> int:
    name = normalize_name(args.name)
    config = load_config()
    custom = dict(config.get("custom_providers") or {})
    if name not in custom:
        raise ConfigError(
            f"{name!r} is not a custom provider.",
            hint="Built-in providers cannot be removed.",
        )
    custom.pop(name)
    config.set("custom_providers", custom)
    if normalize_name(config.get("provider") or "") == name:
        config.set("provider", None)
        config.set("model", None)
    config.save()
    print(f"removed provider {name}")
    print(style.dim(f"  any saved key was left in place; remove it with `anyllm keys remove {name}`"))
    return 0


# -- config -----------------------------------------------------------------


def cmd_config_show(args, style: Style) -> int:
    config = load_config()
    print(json.dumps(config.data, indent=2, ensure_ascii=False))
    print(style.dim(f"# {config_path()}"), file=sys.stderr)
    return 0


def _valid_config_key(key: str) -> None:
    if key not in ALLOWED_KEYS:
        raise UsageError(
            f"Unknown config key {key!r}.",
            hint="Known keys: " + ", ".join(ALLOWED_KEYS),
        )


def cmd_config_get(args, style: Style) -> int:
    _valid_config_key(args.key)
    config = load_config()
    value = config.get(args.key)
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False))
    elif value is None:
        print("null")
    elif isinstance(value, bool):
        print("true" if value else "false")
    else:
        print(value)
    return 0


def cmd_config_set(args, style: Style) -> int:
    _valid_config_key(args.key)
    config = load_config()
    providers = load_providers(config)
    value = parse_config_value(args.value)

    if args.key == "provider":
        if value is None:
            config.set("provider", None)
        else:
            provider = get_provider(str(value), providers)
            config.set("provider", provider.name)
            if config.get("model"):
                config.set("model", None)
                print(style.dim("cleared the saved model because the provider changed"))
    elif args.key == "model":
        config.set("model", None if value is None else str(value))
    elif args.key == "system":
        config.set("system", None if value is None else str(value))
    elif args.key == "temperature":
        config.set("temperature", _temperature(value))
    elif args.key == "max_tokens":
        config.set("max_tokens", _max_tokens(value))
    elif args.key == "show_reasoning":
        config.set("show_reasoning", _as_bool(value, "show_reasoning"))
    else:
        config.set(args.key, value)
    config.save()
    stored = config.get(args.key)
    if isinstance(stored, str):
        print(f"{args.key} = {stored}")
    else:
        print(f"{args.key} = {json.dumps(stored, ensure_ascii=False)}")
    return 0


def cmd_config_unset(args, style: Style) -> int:
    _valid_config_key(args.key)
    config = load_config()
    config.unset(args.key)
    config.save()
    print(f"{args.key} reset to default")
    return 0


def cmd_config_path(args, style: Style) -> int:
    print(config_path())
    return 0


# -- doctor -----------------------------------------------------------------


def cmd_doctor(args, style: Style) -> int:
    config = load_config()
    providers = load_providers(config)
    store = KeyStore()

    print(style.bold(f"AnyLLM {__version__}"))
    print(f"  python     {platform.python_version()} ({platform.system()} {platform.release()})")
    print(f"  config     {config_path()}")
    if os.name == "nt":
        scheme = "Windows DPAPI available" if dpapi_available() else "plain text only"
    else:
        scheme = "file permissions 600"
    print(f"  keys       {keys_path()} ({scheme})")
    default_provider = config.get("provider") or "(not set)"
    default_model = config.get("model") or "(provider default)"
    print(f"  default    {default_provider} / {default_model}")
    print()

    rows = []
    ready = 0
    for name in sorted(providers):
        provider = providers[name]
        status = _key_status(provider, store)
        if resolve_key(provider, store).value:
            ready += 1
        rows.append((name, provider.kind, status, provider.default_model or "-"))
    for line in format_table(style, ["PROVIDER", "KIND", "KEY", "DEFAULT MODEL"], rows):
        print(line)
    print()

    if ready == 0:
        print(style.yellow("No provider has a key yet."))
        print("  next: anyllm keys set openai")
        print()
    else:
        print(style.dim("Run `anyllm doctor --test` to send a tiny request."))
        print()

    if not args.test:
        return 0

    targets: list[Provider] = []
    if args.provider:
        for name in args.provider:
            targets.append(get_provider(name, providers))
    else:
        for name in sorted(providers):
            provider = providers[name]
            resolution = resolve_key(provider, store)
            # Only test providers that actually have a credential. Keyless
            # local servers and the generic ANYLLM_API_KEY are tested only when
            # named with --provider.
            if not resolution.value:
                continue
            if resolution.env_var == "ANYLLM_API_KEY":
                continue
            if not provider.default_model and not (
                normalize_name(config.get("provider") or "") == provider.name
                and config.get("model")
            ):
                continue
            targets.append(provider)

    if not targets:
        print(style.yellow("No configured provider to test."))
        return 1

    failures = 0
    print(style.bold("Test requests"))
    for provider in targets:
        model = provider.default_model
        if (
            not model
            and normalize_name(config.get("provider") or "") == provider.name
            and config.get("model")
        ):
            model = str(config.get("model"))
        if not model:
            print(f"  {provider.name:<12} skipped (no model)")
            continue
        resolution = resolve_key(provider, store)
        session = Session(
            provider,
            model,
            base_url=provider.base_url,
            api_key=resolution.value,
            key_source=resolution.source,
            system="You are a health check. Reply with the single word OK.",
            max_tokens=256,
            timeout=max(1.0, float(args.timeout)),
        )
        print(f"  {provider.name:<12} {model} … ", end="", flush=True)
        try:
            turn = session.send("Reply with OK.")
            print(style.green(f"ok ({turn.elapsed:.1f}s)"))
        except AnyLLMError as exc:
            failures += 1
            print(style.red("failed"))
            print(style.dim(f"    {exc.message}"))
            if exc.hint:
                print(style.dim(f"    {exc.hint}"))
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
