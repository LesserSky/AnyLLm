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
    Target,
    add_target,
    conflicting_target_flags,
    list_target_specs,
    load_targets,
    plain_http_warning,
    remove_target,
    resolve_credentials,
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
        with self._env(home, OPENAI_API_KEY="sk-openai-secret", ANYLLM_TARGET_DEV_API_KEY="dev-secret"):
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

    def test_alias_cannot_reuse_a_provider_name_or_env_slug(self):
        home = _home(self)
        lab = Provider(
            name="open.scale",
            label="Open Scale",
            kind=OPENAI,
            base_url="https://open.scale.example/v1",
            env_vars=("OPEN_SCALE_API_KEY",),
        )
        providers = dict(BUILTINS)
        providers["open.scale"] = lab
        with self._env(home, OPENSCALE_API_KEY="provider-sentinel", ANYLLM_OPENSCALE_API_KEY="also-provider"):
            store = KeyStore()
            store.set("openscale", "saved-provider-sentinel")
            for name in ("openscale", "OpenScale", "OPENSCALE"):
                with self.subTest(name=name):
                    with self.assertRaises(ConfigError) as caught:
                        add_target(
                            name,
                            provider="openscale",
                            model="m",
                            base_url="http://127.0.0.1:9/v1",
                            providers=providers,
                        )
                    self.assertIn("collides", str(caught.exception).lower())
            with self.assertRaises(ConfigError):
                add_target(
                    "open-scale",
                    provider="openai",
                    model="m",
                    base_url="http://127.0.0.1:9/v1",
                    providers=providers,
                )
            path = targets_path()
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "targets": {
                            "ok": {
                                "provider": "openai",
                                "model": "gpt-4o-mini",
                                "base_url": None,
                            },
                            "openscale": {
                                "provider": "openscale",
                                "model": "m",
                                "base_url": "http://127.0.0.1:9/v1",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_targets(providers)
            target = Target("openscale", "m", "http://127.0.0.1:9/v1", alias="openscale")
            credential, _source, _default = resolve_credentials(BUILTINS["openscale"], target, store)
            self.assertNotEqual(credential, "provider-sentinel")
            self.assertNotEqual(credential, "also-provider")
            self.assertNotEqual(credential, "saved-provider-sentinel")

    def _env(self, home: Path, **extra):
        cleaned = {
            key: value
            for key, value in os.environ.items()
            if not key.endswith("_API_KEY") and not key.startswith("ANYLLM_")
        }
        cleaned["ANYLLM_HOME"] = str(home)
        cleaned.update(extra)
        return mock.patch.dict(os.environ, cleaned, clear=True)


class EndpointCredentialTests(unittest.TestCase):
    """Local HTTP only. Sentinel keys must not cross from a provider onto another host."""

    def test_target_identity_survives_models_provider_save_and_resume(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        seen = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_args):
                return

            def _record(self):
                seen.append(self.headers.get("Authorization"))

            def do_GET(self):
                self._record()
                body = json.dumps({"data": [{"id": "mock-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self._record()
                payload = (
                    b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        home = _home(self)
        from anyllm.repl import Repl
        from anyllm.session import Session
        from anyllm.term import Style
        from anyllm.config import load_config
        from anyllm.catalog import load_providers

        with mock.patch.dict(
            os.environ,
            {
                "ANYLLM_HOME": str(home),
                "NO_COLOR": "1",
                "OPENSCALE_API_KEY": "provider-sentinel",
                "ANYLLM_TARGET_LAB_API_KEY": "target-sentinel",
            },
            clear=False,
        ):
            add_target(
                "lab",
                provider="openscale",
                model="mock-model",
                base_url=base,
                providers=BUILTINS,
            )
            config = load_config()
            providers = load_providers(config)
            store = KeyStore()
            session = Session(BUILTINS["openscale"], "placeholder", api_key="provider-sentinel")
            repl = Repl(session, Style(False), providers, store, config)
            from anyllm.targets import repl_target

            repl_target(repl, "lab")
            self.assertEqual(repl.session.base_url, base)
            self.assertEqual(repl.session.api_key, "target-sentinel")
            repl._cmd_models("")
            repl._cmd_provider("openscale")
            self.assertEqual(repl.session.base_url, base)
            self.assertEqual(repl.session.target_alias, "lab")
            self.assertEqual(repl.session.api_key, "target-sentinel")
            repl._send("hi")
            repl._cmd_save("demo")
            fresh = Session(BUILTINS["openai"], "gpt-4o-mini", api_key="provider-sentinel")
            other = Repl(fresh, Style(False), providers, store, config)
            other._cmd_load("demo")
            self.assertEqual(other.session.base_url, base)
            self.assertEqual(other.session.target_alias, "lab")
            self.assertEqual(other.session.api_key, "target-sentinel")
            other._cmd_models("")
            # Old file: custom host, no target identity, provider key must not be sent.
            raw = json.loads((home / "sessions" / "demo.json").read_text(encoding="utf-8"))
            raw.pop("target")
            (home / "sessions" / "old.json").write_text(json.dumps(raw), encoding="utf-8")
            before = len(seen)
            with self.assertRaises(ConfigError) as caught:
                other._cmd_load("old")
            self.assertIn("Refusing to send", str(caught.exception))
            self.assertEqual(other.session.api_key, "target-sentinel")
            self.assertEqual(seen[before:], [])
        self.assertIn("Bearer target-sentinel", seen)
        self.assertNotIn("Bearer provider-sentinel", seen)


if __name__ == "__main__":
    unittest.main()
