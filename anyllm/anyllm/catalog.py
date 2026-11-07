"""Known providers and the custom-provider registry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from .errors import ConfigError

OPENAI = "openai"
ANTHROPIC = "anthropic"
GEMINI = "gemini"
KINDS = (OPENAI, ANTHROPIC, GEMINI)

# Names that are part of the built-in catalog. Custom providers cannot reuse
# them, otherwise a typo would silently shadow a known service.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,38}$")


@dataclass(frozen=True)
class Provider:
    """One API endpoint plus the details needed to talk to it."""

    name: str
    label: str
    kind: str
    base_url: str
    env_vars: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    key_required: bool = True
    local: bool = False
    docs: str = ""
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "

    @property
    def default_model(self) -> str:
        return self.models[0] if self.models else ""

    def auth_headers(self, key: str | None) -> dict[str, str]:
        if not key:
            return {}
        return {self.auth_header: f"{self.auth_prefix}{key}"}


BUILTINS: dict[str, Provider] = {
    "openai": Provider(
        name="openai",
        label="OpenAI",
        kind=OPENAI,
        base_url="https://api.openai.com/v1",
        env_vars=("OPENAI_API_KEY",),
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"),
        docs="https://platform.openai.com/docs",
    ),
    "anthropic": Provider(
        name="anthropic",
        label="Anthropic Claude",
        kind=ANTHROPIC,
        base_url="https://api.anthropic.com/v1",
        env_vars=("ANTHROPIC_API_KEY",),
        models=("claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"),
        docs="https://docs.anthropic.com",
        auth_header="x-api-key",
        auth_prefix="",
    ),
    "google": Provider(
        name="google",
        label="Google Gemini",
        kind=GEMINI,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        models=("gemini-2.5-flash", "gemini-2.5-pro"),
        docs="https://ai.google.dev/gemini-api/docs",
        auth_header="x-goog-api-key",
        auth_prefix="",
    ),
    "deepseek": Provider(
        name="deepseek",
        label="DeepSeek",
        kind=OPENAI,
        base_url="https://api.deepseek.com/v1",
        env_vars=("DEEPSEEK_API_KEY",),
        models=("deepseek-chat", "deepseek-reasoner"),
        docs="https://api-docs.deepseek.com",
    ),
    "groq": Provider(
        name="groq",
        label="Groq",
        kind=OPENAI,
        base_url="https://api.groq.com/openai/v1",
        env_vars=("GROQ_API_KEY",),
        models=("llama-3.3-70b-versatile",),
        docs="https://console.groq.com/docs",
    ),
    "mistral": Provider(
        name="mistral",
        label="Mistral",
        kind=OPENAI,
        base_url="https://api.mistral.ai/v1",
        env_vars=("MISTRAL_API_KEY",),
        models=("mistral-large-latest", "mistral-small-latest"),
        docs="https://docs.mistral.ai",
    ),
    "xai": Provider(
        name="xai",
        label="xAI Grok",
        kind=OPENAI,
        base_url="https://api.x.ai/v1",
        env_vars=("XAI_API_KEY",),
        models=("grok-3", "grok-3-mini"),
        docs="https://docs.x.ai",
    ),
    "openrouter": Provider(
        name="openrouter",
        label="OpenRouter",
        kind=OPENAI,
        base_url="https://openrouter.ai/api/v1",
        env_vars=("OPENROUTER_API_KEY",),
        models=("openrouter/auto",),
        docs="https://openrouter.ai/docs",
    ),
    "openscale": Provider(
        name="openscale",
        label="OpenScale",
        kind=OPENAI,
        base_url="https://api.openscale.so/v1",
        env_vars=("OPENSCALE_API_KEY",),
        models=(
            "deepseek/deepseek-v4.1-flash",
            "z-ai/glm-5.3-flash",
            "deepseek/deepseek-v4.1-flash:fast",
        ),
        docs="https://openscale.so/docs",
    ),
    "together": Provider(
        name="together",
        label="Together AI",
        kind=OPENAI,
        base_url="https://api.together.xyz/v1",
        env_vars=("TOGETHER_API_KEY",),
        models=("meta-llama/Llama-3.3-70B-Instruct-Turbo",),
        docs="https://docs.together.ai",
    ),
    "ollama": Provider(
        name="ollama",
        label="Ollama (local)",
        kind=OPENAI,
        base_url="http://127.0.0.1:11434/v1",
        env_vars=(),
        models=("llama3.2",),
        key_required=False,
        local=True,
        docs="https://github.com/ollama/ollama/blob/main/docs/openai.md",
    ),
    "lmstudio": Provider(
        name="lmstudio",
        label="LM Studio (local)",
        kind=OPENAI,
        base_url="http://127.0.0.1:1234/v1",
        env_vars=(),
        models=(),
        key_required=False,
        local=True,
        docs="https://lmstudio.ai/docs",
    ),
    "custom": Provider(
        name="custom",
        label="Custom OpenAI-compatible endpoint",
        kind=OPENAI,
        base_url="",
        env_vars=("ANYLLM_API_KEY",),
        models=(),
        docs="",
    ),
}


def normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def validate_custom_name(name: str) -> str:
    cleaned = normalize_name(name)
    if not _NAME_RE.match(cleaned):
        raise ConfigError(
            f"Invalid provider name {name!r}.",
            hint="Use 1-39 letters, digits, dots, dashes, or underscores, "
            "starting with a letter or digit.",
        )
    if cleaned in BUILTINS:
        raise ConfigError(
            f"{cleaned!r} is a built-in provider and cannot be redefined.",
            hint="Pick another name, or use it directly with --provider.",
        )
    return cleaned


def validate_base_url(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    if not value:
        raise ConfigError("A base URL is required.", hint="Example: --base-url https://api.example.com/v1")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ConfigError(
            f"Invalid base URL {url!r}.",
            hint="Use an http:// or https:// URL, for example https://api.example.com/v1",
        )
    return value


def _provider_from_spec(name: str, spec: object) -> Provider:
    if not isinstance(spec, dict):
        raise ConfigError(
            f"Provider {name!r} in the config is not an object.",
            hint="Fix the custom_providers entry in the config file, or remove it.",
        )
    kind = str(spec.get("kind") or OPENAI).lower()
    if kind not in KINDS:
        raise ConfigError(
            f"Provider {name!r} has unsupported kind {kind!r}.",
            hint="Use one of: " + ", ".join(KINDS),
        )
    try:
        base_url = validate_base_url(str(spec.get("base_url") or ""))
    except ConfigError as exc:
        raise ConfigError(
            f"Provider {name!r} has an invalid base URL.",
            hint=exc.hint,
        ) from None
    env = spec.get("env") or []
    models = spec.get("models") or []
    if not isinstance(env, list) or not all(isinstance(item, str) for item in env):
        raise ConfigError(f"Provider {name!r} has an invalid 'env' list in the config.")
    if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
        raise ConfigError(f"Provider {name!r} has an invalid 'models' list in the config.")
    return Provider(
        name=name,
        label=str(spec.get("label") or name),
        kind=kind,
        base_url=base_url,
        env_vars=tuple(env),
        models=tuple(models),
        key_required=bool(spec.get("key_required", True)),
        auth_header=str(spec.get("auth_header") or "Authorization"),
        auth_prefix=str(spec.get("auth_prefix") if spec.get("auth_prefix") is not None else "Bearer "),
    )


def load_providers(config) -> dict[str, Provider]:
    """Merge built-ins with custom providers stored in the config."""
    providers = dict(BUILTINS)
    custom = config.get("custom_providers") or {}
    if not isinstance(custom, dict):
        raise ConfigError(
            "'custom_providers' in the config must be an object.",
            hint="Run `anyllm config path` and fix the file, or delete it to start over.",
        )
    for raw_name, spec in custom.items():
        name = normalize_name(raw_name)
        if not name or name in BUILTINS:
            raise ConfigError(
                f"Custom provider name {raw_name!r} is reserved or empty.",
                hint="Remove or rename that entry in the config file.",
            )
        providers[name] = _provider_from_spec(name, spec)
    return providers


def get_provider(name: str, providers: dict[str, Provider]) -> Provider:
    key = normalize_name(name)
    if key in providers:
        return providers[key]
    available = ", ".join(sorted(providers))
    raise ConfigError(
        f"Unknown provider {name!r}.",
        hint=f"Available providers: {available}. "
        "Add your own with `anyllm providers add NAME --base-url URL`.",
    )


def provider_spec_from_args(name: str, args) -> dict:
    """Build the JSON spec stored for a custom provider."""
    kind = args.kind
    if args.auth_header:
        auth_header = args.auth_header
    elif kind == ANTHROPIC:
        auth_header = "x-api-key"
    elif kind == GEMINI:
        auth_header = "x-goog-api-key"
    else:
        auth_header = "Authorization"
    if args.auth_prefix is not None:
        auth_prefix = args.auth_prefix
    elif kind in (ANTHROPIC, GEMINI):
        auth_prefix = ""
    else:
        auth_prefix = "Bearer "
    spec = {
        "kind": kind,
        "base_url": validate_base_url(args.base_url),
        "env": list(args.env or []),
        "models": list(args.model or []),
        "key_required": not args.no_key,
        "auth_header": auth_header,
        "auth_prefix": auth_prefix,
        "label": args.label or name,
    }
    return spec
