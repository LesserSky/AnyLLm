"""API key storage.

Keys are kept in a separate ``keys.json`` file, never in the main config:

* Windows: encrypted with DPAPI (the current Windows user can decrypt it, other
  users and copies on other machines cannot).
* Linux/macOS: stored in a file with permissions 0600.

Environment variables always win over saved keys, which keeps CI and scripts
predictable. Base64 is only an encoding, not encryption, for the plain scheme.
"""

from __future__ import annotations

import base64
import ctypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .catalog import BUILTINS, OPENAI
from .config import keys_path, read_json, write_json
from .errors import ConfigError

SCHEME_DPAPI = "dpapi"
SCHEME_PLAIN = "plain"

_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.c_void_p)]


def _dpapi_functions():
    """Load crypt32/kernel32 and declare the DPAPI signatures."""
    if os.name != "nt":
        raise OSError("Windows DPAPI is only available on Windows.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_wchar_p,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = ctypes.c_int
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(ctypes.c_wchar_p),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def _dpapi_protect(data: bytes) -> bytes:
    crypt32, kernel32 = _dpapi_functions()
    # Keep the input buffer alive for the duration of the native call.
    buffer = ctypes.create_string_buffer(data, len(data))
    blob_in = _DataBlob(len(data), ctypes.cast(buffer, ctypes.c_void_p))
    blob_out = _DataBlob()
    ok = crypt32.CryptProtectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not ok:
        raise OSError(ctypes.get_last_error(), "CryptProtectData failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(blob_out.pbData, ctypes.c_void_p))


def _dpapi_unprotect(data: bytes) -> bytes:
    crypt32, kernel32 = _dpapi_functions()
    buffer = ctypes.create_string_buffer(data, len(data))
    blob_in = _DataBlob(len(data), ctypes.cast(buffer, ctypes.c_void_p))
    blob_out = _DataBlob()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not ok:
        raise OSError(ctypes.get_last_error(), "CryptUnprotectData failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(blob_out.pbData, ctypes.c_void_p))


def dpapi_available() -> bool:
    return os.name == "nt"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def mask_key(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _env_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


@dataclass(frozen=True)
class KeyResolution:
    value: str | None
    source: str
    env_var: str | None = None
    stored: bool = False

    @property
    def present(self) -> bool:
        return bool(self.value)

    @property
    def display(self) -> str:
        if self.env_var:
            return f"env:{self.env_var}"
        if self.stored:
            return "saved"
        if self.value:
            return "configured"
        return "missing"


class KeyStore:
    """Read/write the per-user key file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or keys_path()

    def _load(self) -> dict:
        data = read_json(self.path, what="key store")
        if not data:
            return {"version": 1, "keys": {}}
        keys = data.get("keys")
        if keys is None:
            data["keys"] = {}
        elif not isinstance(keys, dict):
            raise ConfigError(
                "The key store is malformed: 'keys' must be an object.",
                hint=f"Fix or delete {self.path}",
            )
        return data

    def _save(self, data: dict) -> None:
        data["version"] = 1
        try:
            write_json(self.path, data, private=True)
        except OSError as exc:
            raise ConfigError(
                f"Could not save the API key: {exc}",
                hint=f"Check that you can write to {self.path.parent}",
            ) from None

    def get(self, provider: str) -> str | None:
        data = self._load()
        entry = (data.get("keys") or {}).get(provider)
        if entry is None:
            return None
        if not isinstance(entry, dict):
            raise ConfigError(
                f"The saved key for {provider!r} is malformed.",
                hint=f"Run `anyllm keys set {provider}` again.",
            )
        scheme = entry.get("scheme")
        encoded = entry.get("value")
        if not isinstance(encoded, str) or not encoded:
            raise ConfigError(
                f"The saved key for {provider!r} is empty or malformed.",
                hint=f"Run `anyllm keys set {provider}` again.",
            )
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (ValueError, UnicodeError):
            raise ConfigError(
                f"The saved key for {provider!r} is corrupted.",
                hint=f"Run `anyllm keys set {provider}` again.",
            ) from None
        if scheme == SCHEME_PLAIN:
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                raise ConfigError(
                    f"The saved key for {provider!r} is corrupted.",
                    hint=f"Run `anyllm keys set {provider}` again.",
                ) from None
        if scheme == SCHEME_DPAPI:
            if not dpapi_available():
                raise ConfigError(
                    f"The key for {provider!r} was encrypted with Windows DPAPI "
                    "and cannot be read on this system.",
                    hint=f"Set it again on this machine: anyllm keys set {provider}",
                )
            try:
                return _dpapi_unprotect(raw).decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ConfigError(
                    f"The saved key for {provider!r} could not be decrypted: {exc}",
                    hint="It was probably saved by a different Windows user. "
                    f"Run `anyllm keys set {provider}` again.",
                ) from None
        raise ConfigError(
            f"The saved key for {provider!r} uses unknown encryption {scheme!r}.",
            hint=f"Run `anyllm keys set {provider}` again.",
        )

    def set(self, provider: str, key: str) -> str:
        key = (key or "").strip()
        if not key:
            raise ConfigError("The API key is empty; nothing was saved.")
        if any(ch.isspace() for ch in key):
            raise ConfigError(
                "That API key contains whitespace.",
                hint="Paste the key without spaces or quotes.",
            )
        scheme = SCHEME_PLAIN
        payload = key.encode("utf-8")
        if dpapi_available():
            try:
                payload = _dpapi_protect(payload)
                scheme = SCHEME_DPAPI
            except OSError:
                scheme = SCHEME_PLAIN
        data = self._load()
        data.setdefault("keys", {})[provider] = {
            "scheme": scheme,
            "value": base64.b64encode(payload).decode("ascii"),
            "updated": _now(),
        }
        self._save(data)
        return scheme

    def remove(self, provider: str) -> bool:
        data = self._load()
        keys = data.get("keys") or {}
        if provider not in keys:
            return False
        keys.pop(provider, None)
        self._save(data)
        return True

    def info(self) -> dict[str, dict]:
        """Return non-secret metadata for every saved key."""
        data = self._load()
        result: dict[str, dict] = {}
        for name, entry in (data.get("keys") or {}).items():
            if not isinstance(entry, dict):
                continue
            result[name] = {
                "scheme": entry.get("scheme", "?"),
                "updated": entry.get("updated"),
            }
        return result

    def masked(self, provider: str) -> str | None:
        value = self.get(provider)
        return mask_key(value) if value else None


def accepts_generic_api_key(provider) -> bool:
    """Whether the generic ``ANYLLM_API_KEY`` may be sent to ``provider``.

    That variable is for an OpenAI-compatible endpoint the user stood up
    themselves (the built-in ``custom`` provider, or one added with
    ``providers add``). Built-in hosts have their own variables. Using the
    generic key for those would send it to a service it does not name.
    A provider that lists ``ANYLLM_API_KEY`` in its own env vars still honors
    that explicit setting; this only gates the implicit fallback.
    """
    if getattr(provider, "kind", None) != OPENAI:
        return False
    name = getattr(provider, "name", "")
    if name == "custom":
        return True
    return name not in BUILTINS


def resolve_key(provider, store: KeyStore | None = None) -> KeyResolution:
    """Find a key for ``provider``, preferring environment variables."""
    store = store or KeyStore()
    candidates: list[str] = [f"ANYLLM_{_env_slug(provider.name)}_API_KEY"]
    candidates.extend(provider.env_vars)
    if accepts_generic_api_key(provider):
        candidates.append("ANYLLM_API_KEY")
    seen: set[str] = set()
    for variable in candidates:
        if not variable or variable in seen:
            continue
        seen.add(variable)
        value = os.environ.get(variable)
        if value and value.strip():
            return KeyResolution(value.strip(), f"env:{variable}", env_var=variable)
    stored = store.get(provider.name)
    if stored:
        return KeyResolution(stored, "saved", stored=True)
    if not provider.key_required:
        return KeyResolution(None, "not required")
    return KeyResolution(None, "missing")
