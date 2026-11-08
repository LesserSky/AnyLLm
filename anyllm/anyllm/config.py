"""Configuration paths, defaults, and JSON persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .errors import ConfigError

APP_NAME = "anyllm"

DEFAULTS: dict = {
    "version": 1,
    "provider": None,
    "model": None,
    "system": None,
    "temperature": None,
    "max_tokens": None,
    "show_reasoning": False,
    "custom_providers": {},
}

# Keys a user may change with `anyllm config set`.
# `custom_providers` is managed by the `providers` commands, not by hand.
ALLOWED_KEYS = (
    "provider",
    "model",
    "system",
    "temperature",
    "max_tokens",
    "show_reasoning",
)

# Written by the app itself (the `providers` commands), not by `config set`.
INTERNAL_KEYS = ("custom_providers",)


def config_dir() -> Path:
    """Directory for config, keys, and saved sessions.

    ``ANYLLM_HOME`` overrides the platform default (useful for tests and for a
    portable copy of the tool).
    """
    override = os.environ.get("ANYLLM_HOME")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    else:
        base = os.environ.get("XDG_CONFIG_HOME")
        root = Path(base) if base else Path.home() / ".config"
    return root / APP_NAME


def ensure_config_dir() -> Path:
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
    return directory


def config_path() -> Path:
    return config_dir() / "config.json"


def keys_path() -> Path:
    return config_dir() / "keys.json"


def sessions_dir() -> Path:
    return ensure_config_dir() / "sessions"


def history_path() -> Path:
    return config_dir() / "input-history"


def write_json(path: Path, data: dict, *, private: bool = False) -> None:
    """Atomically write JSON, optionally with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        mode = 0o600 if private else 0o644
        try:
            os.chmod(temp_path, mode)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_json(path: Path, *, what: str) -> dict:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read the {what} file: {exc}", hint=str(path)) from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"The {what} file is not valid JSON (line {exc.lineno}).",
            hint=f"Fix or delete {path}",
        ) from None
    if not isinstance(data, dict):
        raise ConfigError(
            f"The {what} file must contain a JSON object.",
            hint=f"Fix or delete {path}",
        )
    return data


class Config:
    """Small wrapper around the config dict with defaults applied."""

    def __init__(self, path: Path, data: dict | None = None) -> None:
        self.path = path
        self.data = dict(DEFAULTS)
        if data:
            self.data.update(data)
        if not isinstance(self.data.get("custom_providers"), dict):
            raise ConfigError(
                "'custom_providers' in the config must be an object.",
                hint=f"Fix or delete {path}",
            )

    def get(self, key: str, default=None):
        if key in self.data:
            return self.data[key]
        return DEFAULTS.get(key, default)

    def set(self, key: str, value) -> None:
        if key not in ALLOWED_KEYS and key not in INTERNAL_KEYS:
            raise ConfigError(
                f"Unknown config key {key!r}.",
                hint="Known keys: " + ", ".join(ALLOWED_KEYS),
            )
        self.data[key] = value

    def unset(self, key: str) -> None:
        if key not in ALLOWED_KEYS:
            raise ConfigError(
                f"Unknown config key {key!r}.",
                hint="Known keys: " + ", ".join(ALLOWED_KEYS),
            )
        self.data[key] = DEFAULTS.get(key)

    def save(self) -> None:
        payload = dict(self.data)
        payload["version"] = 1
        write_json(self.path, payload, private=True)


def load_config() -> Config:
    path = config_path()
    data = read_json(path, what="config")
    return Config(path, data)


# -- value parsing for `config set` -----------------------------------------


def parse_config_value(text: str):
    """Parse a command-line value into a JSON-ish Python value."""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        return text


def _as_float(value, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"Invalid {name} value {value!r}.",
            hint="Use a number, e.g. 0.7",
        ) from None


def _as_int(value, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"Invalid {name} value {value!r}.",
            hint="Use a whole number, e.g. 1024",
        ) from None


def validate_temperature(value) -> float | None:
    """Accept a sampling temperature in ``[0, 2]``, or ``None`` for the default."""
    number = _as_float(value, "temperature")
    if number is not None and not 0.0 <= number <= 2.0:
        raise ConfigError(
            f"temperature must be between 0 and 2 (got {number}).",
            hint="Most models are happiest between 0 and 1.",
        )
    return number


def validate_max_tokens(value) -> int | None:
    """Accept a positive response-token cap, or ``None`` for the provider default."""
    number = _as_int(value, "max_tokens")
    if number is not None and number <= 0:
        raise ConfigError(
            f"max_tokens must be greater than 0 (got {number}).",
            hint="Try 1024 or 4096.",
        )
    return number
