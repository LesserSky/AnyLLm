"""Targets: aliases, flag conflicts, and host isolation."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.catalog import OPENAI, Provider, BUILTINS
from anyllm.errors import ConfigError, UsageError
from anyllm.keystore import KeyStore
from anyllm.targets import (
    add_target,
    conflicting_target_flags,
    list_target_specs,
    load_targets,
    plain_http_warning,
    remove_target,
    resolve_member_spec,
    resolve_named,
    split_provider_model,
    targets_path,
)
from anyllm.config import write_json


def _home(case: unittest.TestCase) -> Path:
    path = ROOT / ".test-run" / f"targets-{os.getpid()}-{id(case)}"
    path.mkdir(parents=True, exist_ok=True)
    case.addCleanup(lambda: __import__("shutil").rmtree(path, True))
    return path


class TargetTests(unittest.TestCase):
    def test_roundtrip_and_remove(self):
        home = _home(self)
        with self._env(home):
            add_target(
                "fast",
                provider="openscale",
                model="deepseek/deepseek-v4.1-flash",
                base_url=None,
                providers=BUILTINS,
            )
            add_target(
                "dev",
                provider="custom",
                model="local",
                base_url="http://localhost:1234/v1",
                providers=BUILTINS,
            )
            specs = list_target_specs()
            self.assertEqual(specs["fast"]["provider"], "openscale")
            self.assertIsNone(specs["fast"]["base_url"])
            self.assertEqual(specs["dev"]["base_url"], "http://localhost:1234/v1")
            text = targets_path().read_text(encoding="utf-8")
            self.assertNotIn("sk-", text)
            self.assertNotIn("api_key", text.lower())
            remove_target("fast")
            self.assertNotIn("fast", list_target_specs())

    def test_split_keeps_model_colons_and_slashes(self):
        provider, model = split_provider_model("openscale:deepseek/deepseek-v4.1-flash:fast")
        self.assertEqual(provider, "openscale")
        self.assertEqual(model, "deepseek/deepseek-v4.1-flash:fast")
        with self.assertRaises(ConfigError):
            split_provider_model("no-colon")

    def test_unknown_alias_fails_before_a_request(self):
        home = _home(self)
        called = {"n": 0}

        def boom(*_args, **_kwargs):
            called["n"] += 1
            raise AssertionError("network")

        with self._env(home):
            with self.assertRaises(ConfigError):
                resolve_named("missing", BUILTINS, KeyStore())
        self.assertEqual(called["n"], 0)

    def test_conflicting_flags(self):
        with self.assertRaises(UsageError):
            conflicting_target_flags(SimpleNamespace(target="fast", provider="openai", model=None, base_url=None))
        with self.assertRaises(UsageError):
            conflicting_target_flags(SimpleNamespace(target="fast", provider=None, model="x", base_url=None))
        with self.assertRaises(UsageError):
            conflicting_target_flags(SimpleNamespace(target="fast", provider=None, model=None, base_url="http://x"))
        conflicting_target_flags(SimpleNamespace(target=None, provider="openai", model="m", base_url=None))

    def test_malformed_and_future_version(self):
        home = _home(self)
        with self._env(home):
            path = targets_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"version": 1, "targets": []}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_targets()
            write_json(path, {"version": 2, "targets": {}}, private=True)
            with self.assertRaises(ConfigError) as caught:
                load_targets()
            self.assertIn("version", str(caught.exception).lower())

    def test_default_host_uses_provider_key_and_other_host_does_not(self):
        home = _home(self)
        with self._env(home, OPENAI_API_KEY="sk-openai-secret"):
            add_target(
                "official",
                provider="openai",
                model="gpt-4o-mini",
                base_url=None,
                providers=BUILTINS,
            )
            official = resolve_named("official", BUILTINS, KeyStore())
            self.assertEqual(official.credential, "sk-openai-secret")
            self.assertTrue(official.host_is_default)

            add_target(
                "dev",
                provider="openai",
                model="local",
                base_url="http://127.0.0.1:1234/v1",
                providers=BUILTINS,
            )
            with self.assertRaises(ConfigError) as caught:
                resolve_named("dev", BUILTINS, KeyStore())
            self.assertNotIn("sk-openai-secret", str(caught.exception))
            self.assertIn("will not be sent", caught.exception.hint or "")

    def test_alias_key_is_sent_to_a_non_default_host(self):
        home = _home(self)
        with self._env(home, OPENAI_API_KEY="sk-openai-secret", ANYLLM_DEV_API_KEY="dev-secret"):
            add_target(
                "dev",
                provider="openai",
                model="local",
                base_url="http://127.0.0.1:1234/v1",
                providers=BUILTINS,
            )
            resolved = resolve_named("dev", BUILTINS, KeyStore())
            self.assertEqual(resolved.credential, "dev-secret")
            self.assertNotEqual(resolved.credential, "sk-openai-secret")
            self.assertFalse(resolved.host_is_default)

    def test_custom_endpoint_can_use_generic_key(self):
        home = _home(self)
        with self._env(home, ANYLLM_API_KEY="generic-secret"):
            add_target(
                "box",
                provider="custom",
                model="local",
                base_url="http://localhost:1234/v1",
                providers=BUILTINS,
            )
            resolved = resolve_named("box", BUILTINS, KeyStore())
            self.assertEqual(resolved.credential, "generic-secret")

    def test_user_provider_key_is_not_forwarded_to_a_different_url(self):
        home = _home(self)
        lab = Provider(
            name="lab",
            label="Lab",
            kind=OPENAI,
            base_url="http://lab.internal/v1",
            env_vars=("LAB_API_KEY",),
        )
        providers = {"lab": lab}
        with self._env(home, LAB_API_KEY="lab-secret"):
            same = resolve_member_spec("lab:model", providers, KeyStore())
            self.assertEqual(same.credential, "lab-secret")
            add_target(
                "elsewhere",
                provider="lab",
                model="model",
                base_url="http://other.internal/v1",
                providers=providers,
            )
            with self.assertRaises(ConfigError) as caught:
                resolve_named("elsewhere", providers, KeyStore())
            self.assertNotIn("lab-secret", str(caught.exception))

    def test_plain_http_warning_is_only_for_nonlocal_hosts(self):
        self.assertIsNone(plain_http_warning("http://127.0.0.1:1234/v1"))
        self.assertIsNone(plain_http_warning("http://localhost:1234/v1"))
        self.assertIsNone(plain_http_warning("https://example.com/v1"))
        warning = plain_http_warning("http://example.com/v1")
        self.assertIsNotNone(warning)
        self.assertIn("not be encrypted", warning)

    def _env(self, home: Path, **extra):
        cleaned = {
            key: value
            for key, value in os.environ.items()
            if not key.endswith("_API_KEY") and not key.startswith("ANYLLM_")
        }
        cleaned["ANYLLM_HOME"] = str(home)
        cleaned.update(extra)
        return mock.patch.dict(os.environ, cleaned, clear=True)


if __name__ == "__main__":
    unittest.main()
