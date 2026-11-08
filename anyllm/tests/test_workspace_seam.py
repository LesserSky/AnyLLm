"""Seam-shaped consent and proposal calls, without Package A installed."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import unittest
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.catalog import OPENAI, Provider
from anyllm.config import load_config
from anyllm.errors import UsageError
from anyllm.keystore import KeyStore
from anyllm.repl import Repl
from anyllm.session import Session
from anyllm.term import Style
from anyllm.workspace import WorkspaceService
from anyllm.workspace_seam import (
    dispatch_repl,
    SeamPort,
    TargetView,
    load_orchestration_seam,
    register_workspace,
    wiring_notes,
)


def new_temp_dir(case: unittest.TestCase) -> Path:
    base = ROOT / ".test-run"
    base.mkdir(exist_ok=True)
    path = base / f"seam-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    case.addCleanup(shutil.rmtree, path, True)
    return path


@dataclass(frozen=True)
class Approved:
    path: str
    sha256: str
    content: str

    def ref(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": len(self.content.encode("utf-8")),
        }


class Store:
    def __init__(self) -> None:
        self.items: dict[str, Approved] = {}

    def approve(self, attachment: Approved) -> None:
        if ".." in attachment.path.split("/"):
            raise ValueError("path")
        self.items[attachment.path] = attachment

    def revoke(self, path: str) -> None:
        self.items.pop(path, None)

    def clear(self) -> None:
        self.items.clear()

    def list(self):
        return tuple(self.items[key] for key in sorted(self.items))


@dataclass
class Snapshot:
    history: tuple
    system: str | None
    temperature: float | None
    max_tokens: int | None
    attachments: tuple
    record_refs: bool = False
    snapshot_id: str = "snap"


def make_snapshot(history, *, system=None, temperature=None, max_tokens=None, attachments=(), record_refs=False, snapshot_id=None):
    frozen = []
    for item in history or []:
        if isinstance(item, dict):
            frozen.append((item.get("role"), item.get("content")))
        else:
            frozen.append((item[0], item[1]))
    return Snapshot(
        history=tuple(frozen),
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        attachments=tuple(attachments or ()),
        record_refs=record_refs,
        snapshot_id=snapshot_id or "snap",
    )


@dataclass
class Options:
    timeout: float = 30.0
    max_tokens: int | None = None


@dataclass
class Result:
    status: str
    text: str = ""
    error: str | None = None
    usage: object = None


DIFF = """--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-print(1)
+print(2)
"""


class SeamContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = new_temp_dir(self)
        self.home = new_temp_dir(self)
        (self.root / "app.py").write_text("print(1)\n", encoding="utf-8")
        self.store = Store()
        self.calls = []
        self.events = queue.Queue()

        def run_request(provider, target, snapshot, options, *, credential, prompt="", messages=None, cancel_event=None, event_queue=None):
            self.calls.append(
                {
                    "provider": provider,
                    "target": target,
                    "snapshot": snapshot,
                    "options": options,
                    "credential": credential,
                    "prompt": prompt,
                    "messages": messages,
                    "cancel_event": cancel_event,
                    "event_queue": event_queue,
                }
            )
            if event_queue is not None:
                event_queue.put(type("Event", (), {"target": target.alias or target.provider, "status": "ok"})())
            return Result(status="ok", text=DIFF)

        self.run_request = run_request
        self.port = SeamPort(
            store=self.store,
            make_snapshot=make_snapshot,
            run_request=run_request,
            factory=Approved,
            provider_obj={"name": "openai"},
            target=TargetView("openai", "mock", "https://example.test/v1", None),
            credential="secret-key",
            options=Options(),
            history=lambda: [{"role": "user", "content": "earlier"}],
            system=lambda: None,
            event_queue=self.events,
        )
        self.service = WorkspaceService(
            self.root,
            home=self.home,
            destination="openai",
            seam=self.port,
        )
        self.service.handle("file", "add app.py")

    def test_live_seam_is_absent_on_this_branch(self):
        self.assertIsNone(load_orchestration_seam())
        notes = wiring_notes()
        self.assertIn("register_workspace(register_repl)", notes)
        self.assertIn("attachment_store().approve", notes)
        self.assertIn("run_request", notes)
        self.assertIn("append_exchange", notes)

    def test_consent_approves_then_run_request_once(self):
        answers = iter(["y", "n"])
        lines = self.service.handle(
            "edit",
            '--format diff "change"',
            interactive=True,
            confirm=lambda _prompt: next(answers),
        )
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["credential"], "secret-key")
        self.assertIsNone(call["messages"])
        self.assertNotIn("secret-key", call["prompt"])
        self.assertIn("untrusted data", call["prompt"])
        self.assertIn("print(1)", call["prompt"])
        snapshot = call["snapshot"]
        self.assertFalse(hasattr(snapshot, "api_key") and snapshot.api_key)
        self.assertFalse(getattr(snapshot, "credential", None))
        self.assertFalse(snapshot.record_refs)
        self.assertEqual(snapshot.history, (("user", "earlier"),))
        self.assertEqual(len(snapshot.attachments), 1)
        attachment = snapshot.attachments[0]
        self.assertEqual(attachment.path, "app.py")
        self.assertEqual(attachment.content, "print(1)\n")
        self.assertEqual(attachment.sha256, self.service.workspace.attachments["app.py"].sha256)
        self.assertEqual(self.store.list()[0].path, "app.py")
        self.assertIn("Not applied", "\n".join(lines))
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "print(1)\n")
        self.assertFalse(self.events.empty())

    def test_declined_consent_does_not_approve_or_call(self):
        with self.assertRaises(UsageError):
            self.service.handle(
                "edit",
                '--format diff "change"',
                interactive=True,
                confirm=lambda _prompt: "n",
            )
        self.assertEqual(self.calls, [])
        self.assertEqual(self.store.list(), ())
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "print(1)\n")

    def test_noninteractive_does_not_touch_the_store(self):
        with self.assertRaises(UsageError):
            self.service.handle("edit", '--format diff "change"', interactive=False, confirm=lambda _prompt: "y")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.store.list(), ())

    def test_failed_status_writes_nothing_and_file_change_revokes(self):
        def fail(*_args, **_kwargs):
            return Result(status="error", error="provider down", text="")

        self.port.run_request = fail
        with self.assertRaises(UsageError) as caught:
            self.service.handle(
                "edit",
                '--format diff "change"',
                interactive=True,
                confirm=lambda _prompt: "y",
            )
        self.assertIn("provider down", str(caught.exception))
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "print(1)\n")
        self.assertEqual(len(self.store.list()), 1)
        self.service.handle("file", "remove app.py")
        self.assertEqual(self.store.list(), ())

    def test_register_handlers_match_repl_shape(self):
        found = {}

        def register(name, handler, help_row=None):
            found[name] = (handler, help_row)

        register_workspace(register)
        self.assertEqual(set(found), {"file", "edit", "undo"})
        for handler, row in found.values():
            self.assertEqual(handler.__code__.co_argcount, 2)
            self.assertIsInstance(row, tuple)
            self.assertEqual(len(row), 2)


class WiredCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = new_temp_dir(self)
        self.home = new_temp_dir(self)
        (self.root / "app.py").write_text("print(1)\n", encoding="utf-8")
        self.provider = Provider(
            name="mock",
            label="Mock",
            kind=OPENAI,
            base_url="https://example.test/v1",
        )

    def test_repl_file_commands_do_not_send_or_write(self):
        session = Session(self.provider, "mock", api_key="secret-key")
        with mock.patch.dict(os.environ, {"ANYLLM_HOME": str(self.home)}):
            repl = Repl(
                session,
                Style(False),
                {"mock": self.provider},
                KeyStore(),
                load_config(),
            )
            repl.workspace_root = str(self.root)
            lines = iter(
                [
                    "/file add app.py",
                    "/file list",
                    "/edit --format diff change the print",
                    "/help",
                    "/exit",
                ]
            )
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with (
                mock.patch("builtins.input", side_effect=lambda prompt="": next(lines)),
                redirect_stdout(buf),
            ):
                self.assertEqual(repl.run(), 0)
        text = buf.getvalue()
        self.assertIn("staged app.py", text)
        self.assertIn("app.py", text)
        self.assertIn("Refusing to send attachments in noninteractive mode", text)
        self.assertIn("/file ...", text)
        self.assertNotIn("secret-key", text)
        self.assertEqual(session.messages, [])
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "print(1)\n")
        self.assertIsNone(load_orchestration_seam())

    def test_interactive_edit_without_seam_does_not_send(self):
        session = Session(self.provider, "mock", api_key="secret-key")

        class Stub:
            def __init__(self) -> None:
                self.session = session
                self.style = Style(False)
                self.workspace_root = str(self_root)
                self.lines: list[str] = []

            def say(self, text: str = "") -> None:
                self.lines.append(text)

        self_root = self.root
        stub = Stub()
        with mock.patch("anyllm.workspace_seam._stdio_is_interactive", return_value=True):
            with self.assertRaises(UsageError) as caught:
                dispatch_repl(stub, "edit", "--format diff change the print")
        self.assertIn("No proposal runner", str(caught.exception))
        self.assertNotIn("secret-key", str(caught.exception))
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "print(1)\n")
        self.assertEqual(session.messages, [])

    def test_cli_preview_never_writes_or_calls_out(self):
        proposal = self.root / "proposal.diff"
        proposal.write_text(
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print(1)\n+print(2)\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["ANYLLM_HOME"] = str(self.home)
        env["NO_COLOR"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        for name in list(env):
            if name.endswith("_API_KEY") or name.startswith("ANYLLM_"):
                if name != "ANYLLM_HOME":
                    env.pop(name, None)
        env["ANYLLM_HOME"] = str(self.home)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "anyllm",
                "edit",
                "--root",
                str(self.root),
                "--file",
                "app.py",
                "--format",
                "diff",
                "--proposal-file",
                str(proposal),
                "change the print",
            ],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Not applied", result.stdout)
        self.assertIn("print(2)", result.stdout)
        self.assertNotIn("\x1b", result.stdout)
        self.assertNotIn("secret", result.stdout)
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "print(1)\n")
        listed = subprocess.run(
            [sys.executable, "-m", "anyllm", "undo", "--root", str(self.root), "last"],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertNotEqual(listed.returncode, 0)
        self.assertIn("No applied edit", listed.stderr + listed.stdout)
        self.assertEqual((self.root / "app.py").read_text(encoding="utf-8"), "print(1)\n")


if __name__ == "__main__":
    unittest.main()
