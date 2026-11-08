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


def load_targets(providers: dict | None = None) -> dict:
    """Load every target or raise. A bad entry rejects the whole file."""

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
    if providers is None:
        from .catalog import load_providers
        from .config import load_config

        providers = load_providers(load_config())
    # Validate the full map before returning any of it.
    checked: dict[str, dict] = {}
    for name, spec in targets.items():
        alias = validate_target_alias(str(name), providers)
        _validate_stored(alias, spec, path)
        if alias in checked:
            raise ConfigError(
                f"Target names {name!r} and {alias!r} normalize to the same alias.",
                hint=f"Fix or remove {path}.",
            )
        checked[alias] = spec
    return {"version": 1, "targets": checked}


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


def normalize_alias(name: str) -> str:
    """Case-fold a label. This does not check provider collisions."""

    cleaned = (name or "").strip().casefold()
    if not _NAME.match(cleaned) or ":" in cleaned:
        raise ConfigError(
            f"Invalid target name {name!r}.",
            hint="Use 1-32 letters, digits, dots, dashes, or underscores. Names cannot contain a colon.",
        )
    return cleaned


def _env_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def target_env_var(alias: str) -> str:
    """Environment variable for a target. Distinct from ANYLLM_<PROVIDER>_API_KEY."""

    return "ANYLLM_TARGET_" + _env_slug(alias) + "_API_KEY"


def target_store_name(alias: str) -> str:
    """Key-store id for a target. Distinct from a provider's saved key."""

    return "target:" + normalize_alias(alias)


def alias_collides(alias: str, providers: dict) -> str | None:
    """Return a short reason when ``alias`` shares a provider name or env slug."""

    folded = alias.casefold()
    slug = _env_slug(alias)
    for name in providers:
        if name.casefold() == folded:
            return f"provider {name!r}"
        if _env_slug(name) == slug:
            return f"the environment name of provider {name!r} (ANYLLM_{_env_slug(name)}_API_KEY)"
    return None


def validate_target_alias(name: str, providers: dict) -> str:
    cleaned = normalize_alias(name)
    collision = alias_collides(cleaned, providers)
    if collision:
        raise ConfigError(
            f"Target name {cleaned!r} collides with {collision}.",
            hint="Choose a name that is not a provider and does not normalize to a provider's "
            "ANYLLM_<PROVIDER>_API_KEY variable. Target keys use ANYLLM_TARGET_<NAME>_API_KEY "
            "or `anyllm keys set target:<name>`.",
        )
    return cleaned


def add_target(name: str, *, provider: str, model: str, base_url: str | None, providers: dict) -> Target:
    alias = validate_target_alias(name, providers)
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
    alias = normalize_alias(name)
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
    """Read only the target namespace. Never a provider key or ANYLLM_<PROVIDER>_API_KEY."""

    if not alias:
        return None, ""
    try:
        alias = normalize_alias(alias)
    except ConfigError:
        return None, ""
    env_name = target_env_var(alias)
    env_value = os.environ.get(env_name)
    if env_value and env_value.strip():
        return env_value.strip(), f"env:{env_name}"
    # Provider keys live under the bare provider name. Do not read that slot.
    saved = store.get(target_store_name(alias))
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
    alias = validate_target_alias(name, providers)
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
            if alias:
                hint = (
                    f"Set {target_env_var(alias)} or `anyllm keys set {target_store_name(alias)}`. "
                    f"The {provider.name} key is also accepted on that provider's own endpoint."
                )
            else:
                hint = (
                    f"Run `anyllm keys set {provider.name}`, or set the provider's environment variable."
                )
            raise ConfigError(f"No API key found for {provider.name}.", hint=hint)
        raise ConfigError(
            f"No credential is configured for target {alias or provider.name!r} at {concrete}.",
            hint="This URL is not the provider's default host, so the provider's usual key "
            "will not be sent there. Save a target key with "
            f"`anyllm keys set {target_store_name(alias) if alias else 'target:NAME'}` "
            f"or set {target_env_var(alias or 'NAME')}.",
        )
    return ResolvedTarget(
        target=target,
        provider=provider,
        credential=credential,
        credential_source=source,
        host_is_default=host_is_default if (provider.base_url or "") else True,
        http_warning=plain_http_warning(concrete),
    )


def host_uses_provider_key(provider, base_url: str | None) -> bool:
    """True when this URL is the provider's own configured endpoint.

    An empty provider URL (the built-in ``custom`` provider) means the given
    URL is that endpoint. Any other non-matching URL must not inherit the
    provider key.
    """

    provider_url = getattr(provider, "base_url", "") or ""
    if not provider_url:
        return True
    if not base_url:
        return True
    return _urls_match(base_url, provider_url)


def resolve_endpoint_key(
    provider,
    base_url: str | None,
    *,
    target_alias: str | None,
    store: KeyStore,
    providers: dict,
) -> tuple[str | None, str]:
    """Pick the key for one host. This is the only host-aware lookup.

    A target alias uses the target key namespace. A non-default host without
    a target alias gets no provider key. Callers that send a request, list
    models, restore a session, or re-bind after ``/provider`` should use this
    instead of ``resolve_key``.
    """

    alias = (target_alias or "").strip() or None
    if alias:
        resolved = resolve_named(alias, providers, store, require_credential=False)
        if base_url and not _urls_match(base_url, resolved.target.base_url):
            return None, "withheld"
        return resolved.credential, resolved.credential_source
    if host_uses_provider_key(provider, base_url):
        resolution = resolve_key(provider, store)
        return resolution.value, resolution.source
    return None, "withheld"


def missing_endpoint_key(provider, base_url: str | None, target_alias: str | None) -> ConfigError:
    if target_alias:
        return ConfigError(
            f"No target key for {target_alias!r} at {base_url}.",
            hint="The provider key is not used for this host. "
            f"Set {target_env_var(target_alias)} or "
            f"`anyllm keys set {target_store_name(target_alias)}`.",
        )
    if not host_uses_provider_key(provider, base_url):
        return ConfigError(
            f"Refusing to send the {provider.name} key to {base_url}.",
            hint="That URL is not the provider's default host. "
            "Save a target for it and set ANYLLM_TARGET_<NAME>_API_KEY, "
            "or use the provider's normal endpoint.",
        )
    return ConfigError(
        f"No API key found for {provider.name}.",
        hint=f"Run `anyllm keys set {provider.name}`, or set its environment variable.",
    )


def saved_target_alias(data: dict) -> str | None:
    """Read a persisted target name. Reject anything that is not a string."""

    if "target" not in data:
        return None
    raw = data.get("target")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(
            "Saved session has an invalid target name.",
            hint="The target field must be a string alias, or be omitted. It must not contain a key.",
        )
    return raw.strip()


def rebind_session_credentials(session, providers: dict, store: KeyStore) -> None:
    """Refresh ``session.api_key`` from the target or host rule. Raises when required and missing."""

    alias = getattr(session, "target_alias", None)
    credential, source = resolve_endpoint_key(
        session.provider,
        session.base_url,
        target_alias=alias,
        store=store,
        providers=providers,
    )
    if credential:
        session.api_key = credential
        session.key_source = source
        return
    # An in-memory key is kept only for the provider's own endpoint. A
    # non-default host must not continue to use a provider key that was cached earlier.
    if host_uses_provider_key(session.provider, session.base_url) and session.api_key and not alias:
        if not session.key_source:
            session.key_source = "session"
        return
    session.api_key = None
    session.key_source = source or "withheld"
    if session.provider.key_required:
        raise missing_endpoint_key(session.provider, session.base_url, alias)


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
