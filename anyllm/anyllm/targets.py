"""Named provider/model/base-URL targets.

A target stores no secrets. Credentials are resolved at request time. A target
whose URL is not the provider's configured host does not inherit that
provider's saved key or usual environment variable. That stops a key for
api.openai.com from being sent to some other host just because the provider
name is ``openai``.

``anyllm ask --target NAME`` conflicts with ``--provider``, ``--model``, and
``--base-url`` instead of picking one of them silently.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from .catalog import get_provider, validate_base_url
from .errors import ConfigError, UsageError
from .keystore import KeyStore, resolve_key
from .seam import feature_path, load_feature, register_repl, save_feature

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class Target:
    """One concrete endpoint. ``alias`` is a local label, not a secret."""

    provider: str
    model: str
    base_url: str
    alias: str | None = None


@dataclass(frozen=True)
class ResolvedTarget:
    target: Target
    provider: object
    credential: str | None
    credential_source: str
    host_is_default: bool
    http_warning: str | None


def targets_path():
    return feature_path("targets.json")


def _blank() -> dict:
    return {"version": 1, "targets": {}}


def load_targets() -> dict:
    path = targets_path()
    data = load_feature(path, what="targets")
    targets = data.get("targets", {})
    if targets is None:
        targets = {}
    if not isinstance(targets, dict):
        raise ConfigError(
            "targets.json is malformed.",
            hint=f"'targets' must be an object. Fix or remove {path}.",
        )
    for name, spec in targets.items():
        _validate_stored(name, spec, path)
    return {"version": 1, "targets": targets}


def _validate_stored(name: str, spec, path) -> None:
    if not isinstance(spec, dict):
        raise ConfigError(
            f"Target {name!r} is malformed.",
            hint=f"Fix or remove {path}.",
        )
    if not isinstance(spec.get("provider"), str) or not spec.get("provider").strip():
        raise ConfigError(f"Target {name!r} has no provider.", hint=f"Fix or remove {path}.")
    if not isinstance(spec.get("model"), str) or not spec.get("model").strip():
        raise ConfigError(f"Target {name!r} has no model.", hint=f"Fix or remove {path}.")
    base = spec.get("base_url")
    if base is not None and not isinstance(base, str):
        raise ConfigError(f"Target {name!r} has an invalid base URL.", hint=f"Fix or remove {path}.")


def save_targets(data: dict) -> None:
    save_feature(targets_path(), {"targets": data.get("targets") or {}})


def validate_alias(name: str) -> str:
    cleaned = (name or "").strip()
    if not _NAME.match(cleaned) or ":" in cleaned:
        raise ConfigError(
            f"Invalid target name {name!r}.",
            hint="Use 1-32 letters, digits, dots, dashes, or underscores. Names cannot contain a colon.",
        )
    return cleaned


def add_target(name: str, *, provider: str, model: str, base_url: str | None, providers: dict) -> Target:
    alias = validate_alias(name)
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("A target needs a model name.", hint="Pass --model NAME.")
    provider_obj = get_provider(provider, providers)
    stored_url = None
    if base_url:
        stored_url = validate_base_url(base_url)
    data = load_targets()
    data["targets"][alias] = {
        "provider": provider_obj.name,
        "model": model.strip(),
        "base_url": stored_url,
    }
    save_targets(data)
    resolved = resolve_named(alias, providers, KeyStore(), require_credential=False)
    return resolved.target


def remove_target(name: str) -> None:
    alias = validate_alias(name)
    data = load_targets()
    if alias not in data["targets"]:
        raise ConfigError(
            f"Unknown target {alias!r}.",
            hint="Run `anyllm targets list` to see saved names.",
        )
    data["targets"].pop(alias)
    save_targets(data)


def list_target_specs() -> dict[str, dict]:
    return dict(load_targets()["targets"])


def split_provider_model(spec: str) -> tuple[str, str]:
    """Split ``provider:model`` at the first colon. Model IDs may contain slashes and colons."""

    text = (spec or "").strip()
    if ":" not in text:
        raise ConfigError(
            f"Could not read a provider:model target from {spec!r}.",
            hint="Use provider:model, for example openscale:deepseek/deepseek-v4.1-flash, or a saved alias.",
        )
    provider, model = text.split(":", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ConfigError(
            f"Could not read a provider:model target from {spec!r}.",
            hint="Put the provider before the first colon and the model after it.",
        )
    return provider, model


def plain_http_warning(url: str) -> str | None:
    """Warn before sending a key to non-local plain HTTP. Localhost is allowed."""

    parts = urlsplit(url or "")
    if parts.scheme != "http":
        return None
    host = (parts.hostname or "").lower().strip("[]")
    if host in _LOCAL_HOSTS:
        return None
    return (
        f"{url} is plain HTTP on a non-local host. "
        "The request, including any API key, would not be encrypted."
    )


def _urls_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    try:
        return validate_base_url(left) == validate_base_url(right)
    except ConfigError:
        return False


def _alias_credential(alias: str | None, store: KeyStore) -> tuple[str | None, str]:
    if not alias:
        return None, ""
    env_name = "ANYLLM_" + re.sub(r"[^A-Za-z0-9]+", "_", alias).strip("_").upper() + "_API_KEY"
    env_value = os.environ.get(env_name)
    if env_value and env_value.strip():
        return env_value.strip(), f"env:{env_name}"
    saved = store.get(alias)
    if saved:
        return saved, "saved-target"
    return None, ""


def resolve_credentials(provider, target: Target, store: KeyStore) -> tuple[str | None, str, bool]:
    """Return ``(credential, source, host_is_default)``.

    A non-default host never receives the builtin or saved provider key.
    """

    provider_url = getattr(provider, "base_url", "") or ""
    host_is_default = bool(provider_url) and _urls_match(target.base_url, provider_url)
    custom_endpoint = not provider_url
    alias_key, alias_source = _alias_credential(target.alias, store)
    if alias_key:
        return alias_key, alias_source, host_is_default or custom_endpoint
    if host_is_default or custom_endpoint:
        resolution = resolve_key(provider, store)
        return resolution.value, resolution.source, host_is_default or custom_endpoint
    # Named builtin (or a user provider) pointed at a different host.
    return None, "withheld", False


def resolve_named(name: str, providers: dict, store: KeyStore, *, require_credential: bool = True) -> ResolvedTarget:
    alias = validate_alias(name)
    spec = list_target_specs().get(alias)
    if spec is None:
        raise ConfigError(
            f"Unknown target {alias!r}.",
            hint="Add it with `anyllm targets add`, or run `anyllm targets list`.",
        )
    return _resolve_spec(
        provider_name=spec["provider"],
        model=spec["model"],
        base_url=spec.get("base_url"),
        alias=alias,
        providers=providers,
        store=store,
        require_credential=require_credential,
    )


def resolve_provider_model(
    spec: str,
    providers: dict,
    store: KeyStore,
    *,
    require_credential: bool = True,
) -> ResolvedTarget:
    provider_name, model = split_provider_model(spec)
    return _resolve_spec(
        provider_name=provider_name,
        model=model,
        base_url=None,
        alias=None,
        providers=providers,
        store=store,
        require_credential=require_credential,
    )


def resolve_member_spec(spec: str, providers: dict, store: KeyStore, *, require_credential: bool = True) -> ResolvedTarget:
    """Resolve an alias, or a ``provider:model`` string. Never guess between them.

    A colon means provider:model. Anything else must already be a saved alias.
    Base URLs come only from saved targets, never from the model text.
    """

    text = (spec or "").strip()
    if ":" in text:
        return resolve_provider_model(text, providers, store, require_credential=require_credential)
    return resolve_named(text, providers, store, require_credential=require_credential)


def _resolve_spec(
    *,
    provider_name: str,
    model: str,
    base_url: str | None,
    alias: str | None,
    providers: dict,
    store: KeyStore,
    require_credential: bool,
) -> ResolvedTarget:
    provider = get_provider(provider_name, providers)
    if base_url:
        concrete = validate_base_url(base_url)
    else:
        concrete = provider.base_url or ""
    if not concrete:
        raise ConfigError(
            f"Provider {provider.name!r} has no base URL.",
            hint="Add the target with --base-url, or choose a provider that has an endpoint.",
        )
    target = Target(provider=provider.name, model=model.strip(), base_url=concrete, alias=alias)
    credential, source, host_is_default = resolve_credentials(provider, target, store)
    if require_credential and provider.key_required and not credential:
        if host_is_default or not (provider.base_url or ""):
            raise ConfigError(
                f"No API key found for {provider.name}.",
                hint=f"Run `anyllm keys set {alias or provider.name}`, or set the provider's environment variable.",
            )
        raise ConfigError(
            f"No credential is configured for target {alias or provider.name!r} at {concrete}.",
            hint="This URL is not the provider's default host, so the provider's usual key "
            "will not be sent there. Save a key for the target name "
            f"(`anyllm keys set {alias}`) or set ANYLLM_{_env_slug(alias or 'TARGET')}_API_KEY.",
        )
    return ResolvedTarget(
        target=target,
        provider=provider,
        credential=credential,
        credential_source=source,
        host_is_default=host_is_default if (provider.base_url or "") else True,
        http_warning=plain_http_warning(concrete),
    )


def _env_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def conflicting_target_flags(args) -> None:
    """Reject ``--target`` combined with an explicit provider, model, or base URL."""

    chosen = getattr(args, "target", None)
    if not chosen:
        return
    clashes = []
    if getattr(args, "provider", None):
        clashes.append("--provider")
    if getattr(args, "model", None):
        clashes.append("--model")
    if getattr(args, "base_url", None):
        clashes.append("--base-url")
    if clashes:
        raise UsageError(
            "Use either --target or " + ", ".join(clashes) + ", not both.",
            hint="A saved target already names the provider, model, and base URL.",
        )


def apply_target_to_session(session, name: str, providers: dict, store: KeyStore):
    """Point a live session at a saved target without copying the key into saved transcripts."""

    resolved = resolve_named(name, providers, store, require_credential=True)
    session.provider = resolved.provider
    session.model = resolved.target.model
    session.base_url = resolved.target.base_url
    session.api_key = resolved.credential
    session.key_source = resolved.credential_source
    session.target_alias = resolved.target.alias
    return resolved


def register(register_command=register_repl) -> None:
    register_command("target", repl_target, ("/target NAME", "use a saved model target"))
    register_command("targets", repl_targets, ("/targets", "list saved targets"))


def repl_target(repl, rest: str) -> None:
    name = (rest or "").strip()
    if not name:
        raise UsageError("Usage: /target NAME", hint="Save one with `anyllm targets add`.")
    resolved = apply_target_to_session(repl.session, name, repl.providers, repl.store)
    repl.say(
        repl.style.dim(
            f"target → {resolved.target.alias} · {resolved.target.provider}/{resolved.target.model}"
        )
    )
    repl.say(repl.style.dim(f"  endpoint {resolved.target.base_url}"))
    if resolved.http_warning:
        repl.say(repl.style.yellow("  " + resolved.http_warning))
    if not resolved.host_is_default and resolved.credential_source == "withheld":
        repl.say(repl.style.dim("  provider key withheld because this is not the default host"))


def repl_targets(repl, _rest: str) -> None:
    specs = list_target_specs()
    if not specs:
        repl.say(repl.style.dim("no saved targets"))
        return
    repl.say(repl.style.bold("Targets"))
    for name in sorted(specs):
        spec = specs[name]
        url = spec.get("base_url") or "(provider default)"
        repl.say(f"  {name}  {spec.get('provider')}/{spec.get('model')}  {url}")
    repl.say("")


def render_target_list(style) -> None:
    specs = list_target_specs()
    if not specs:
        print(style.dim("no saved targets"))
        return
    print(style.bold("NAME          PROVIDER/MODEL"))
    for name in sorted(specs):
        spec = specs[name]
        url = spec.get("base_url") or "(provider default)"
        print(f"{name.ljust(12)}  {spec.get('provider')}/{spec.get('model')}")
        print(style.dim(f"              {url}"))
