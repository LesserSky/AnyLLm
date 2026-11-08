"""Exercise Package A's real seam without merging that branch.

The module is loaded from ``origin/feat/orchestration`` when that ref is
present. It is not copied into this pull request.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import types
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.workspace import WorkspaceService
from anyllm.workspace_seam import bind_service


def new_temp_dir(case: unittest.TestCase) -> Path:
    base = ROOT / ".test-run"
    base.mkdir(exist_ok=True)
    path = base / f"live-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    case.addCleanup(shutil.rmtree, path, True)
    return path


def _load_remote_seam():
    try:
        source = subprocess.check_output(
            ["git", "show", "origin/feat/orchestration:anyllm/anyllm/seam.py"],
            cwd=str(REPO),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    module = types.ModuleType("anyllm.seam")
    module.__package__ = "anyllm"
    module.__file__ = "seam.py"
    sys.modules["anyllm.seam"] = module
    exec(compile(source, "seam.py", "exec"), module.__dict__)
    return module


class LiveSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seam = _load_remote_seam()
        if self.seam is None:
            self.skipTest("origin/feat/orchestration is not available for a read-only seam load")
        self.root = new_temp_dir(self)
        self.home = new_temp_dir(self)
        (self.root / "a.py").write_text("old\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self.seam is not None:
            self.seam.attachment_store().clear()
        sys.modules.pop("anyllm.seam", None)

    def test_bind_follows_the_current_session(self):
        service = WorkspaceService(self.root, home=self.home)
        service.handle("file", "add a.py")

        class Session:
            def __init__(self, name, url):
                self.provider = type("P", (), {"name": name})()
                self.model = "model"
                self.base_url = url
                self.api_key = "not-a-real-key"
                self.messages = []
                self.system = None
                self.max_tokens = 20
                self.temperature = None
                self.timeout = 10

        class Repl:
            def __init__(self):
                self.session = Session("A", "https://a.example/v1")
                self.style = None

        repl = Repl()
        self.assertTrue(bind_service(service, repl))
        calls = []

        def run_request(provider, target, snapshot, options, *, credential, prompt="", messages=None, cancel_event=None, event_queue=None):
            calls.append((provider.name, target.model, target.base_url, credential, snapshot.record_refs))
            self.assertIsNone(getattr(snapshot, "api_key", None))
            self.assertFalse(getattr(snapshot, "record_refs", True))
            return self.seam.RunResult(
                status="ok",
                text='{"edits":[{"path":"a.py","search":"old","replace":"new"}]}',
            )

        self.seam.run_request = run_request
        # bind captured the function object already. Replace the port's callable.
        service.seam.run_request = run_request

        prompts = []

        def confirm(prompt):
            prompts.append(prompt)
            return "n" if prompt.startswith("Apply") else "y"

        repl.session.provider = type("P", (), {"name": "B"})()
        repl.session.base_url = "https://b.example/v1"
        first = io.StringIO()
        with redirect_stdout(first):
            service.handle("edit", '--format blocks "change"', interactive=True, confirm=confirm)
        shown = first.getvalue()
        self.assertIn("Provider: B", shown)
        self.assertIn("https://b.example/v1", shown)
        self.assertIn("Model: model", shown)
        self.assertEqual(calls[-1][0], "B")
        self.assertEqual(calls[-1][2], "https://b.example/v1")
        self.assertEqual(calls[-1][3], "not-a-real-key")
        self.assertFalse(calls[-1][4])

        repl.session = Session("C", "https://c.example/v1")
        second = io.StringIO()
        with redirect_stdout(second):
            service.handle("edit", '--format blocks "change"', interactive=True, confirm=confirm)
        self.assertIn("Provider: C", second.getvalue())
        self.assertEqual(calls[-1][0], "C")
        self.assertEqual(calls[-1][2], "https://c.example/v1")
        self.assertEqual((self.root / "a.py").read_text(encoding="utf-8"), "old\n")


if __name__ == "__main__":
    unittest.main()
