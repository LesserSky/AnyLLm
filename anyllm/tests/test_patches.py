"""Proposal parsing, preview, apply, rollback, and undo tests."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.errors import UsageError
from anyllm.patches import Faults, preview_lines
from anyllm.workspace import WorkspaceService


def new_temp_dir(case: unittest.TestCase) -> Path:
    base = ROOT / ".test-run"
    base.mkdir(exist_ok=True)
    path = base / f"patch-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    case.addCleanup(shutil.rmtree, path, True)
    return path


DIFF = """--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def run(value):
-    return value
+    return value or 0
"""

BLOCKS = """{"edits":[{"path":"src/app.py","search":"return value\\n","replace":"return value or 0\\n"}]}"""


class ParseAndPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = new_temp_dir(self)
        self.home = new_temp_dir(self)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_bytes(b"def run(value):\n    return value\n")
        self.service = WorkspaceService(self.root, home=self.home, destination="openai")
        self.service.handle("file", "add src/app.py")

    def test_diff_preview_is_regenerated_and_escaped(self):
        hostile = DIFF.replace(
            "return value or 0",
            "return value or 0 \x1b]8;;http://evil\x07x",
        )
        lines = self.service.handle("edit", '--format diff "handle empty"', model_text=hostile)
        shown = "\n".join(lines)
        self.assertIn("+", shown)
        self.assertIn("return value or 0", shown)
        self.assertNotIn("\x1b", shown)
        self.assertIn("\\u001b", shown)
        self.assertIn("Not applied", shown)
        self.assertEqual((self.root / "src" / "app.py").read_bytes(), b"def run(value):\n    return value\n")

    def test_rejects_prose_bad_counts_duplicate_and_unattached(self):
        with self.assertRaises(UsageError):
            self.service.handle("edit", '--format diff "x"', model_text="Here is the diff:\n" + DIFF)
        bad_count = DIFF.replace("@@ -1,2 +1,2 @@", "@@ -1,9 +1,9 @@")
        with self.assertRaises(UsageError) as caught:
            self.service.handle("edit", '--format diff "x"', model_text=bad_count)
        self.assertIn("counts", str(caught.exception).lower())
        doubled = DIFF + DIFF
        with self.assertRaises(UsageError):
            self.service.handle("edit", '--format diff "x"', model_text=doubled)
        other = DIFF.replace("src/app.py", "src/other.py")
        with self.assertRaises(UsageError) as caught:
            self.service.handle("edit", '--format diff "x"', model_text=other)
        self.assertIn("not an attached", str(caught.exception).lower())

    def test_rejects_creation_and_mismatched_context(self):
        created = """--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,1 @@
+print(1)
"""
        with self.assertRaises(UsageError):
            self.service.handle("edit", '--format diff "x"', model_text=created)
        mismatch = """--- a/src/app.py
+++ b/src/app.py
@@ -1,1 +1,1 @@
-missing
+added
"""
        with self.assertRaises(UsageError):
            self.service.handle("edit", '--format diff "x"', model_text=mismatch)

    def test_blocks_require_one_exact_match_and_no_extra_keys(self):
        lines = self.service.handle("edit", '--format blocks "rename"', model_text=BLOCKS)
        self.assertIn("Not applied", "\n".join(lines))
        ambiguous = BLOCKS.replace("return value\\n", "value")
        with self.assertRaises(UsageError) as caught:
            self.service.handle("edit", '--format blocks "x"', model_text=ambiguous)
        self.assertIn("exactly once", str(caught.exception).lower())
        extra = '{"edits":[{"path":"src/app.py","search":"return value\\n","replace":"x\\n","mode":"755"}]}'
        with self.assertRaises(UsageError):
            self.service.handle("edit", '--format blocks "x"', model_text=extra)
        prose = "sure\n" + BLOCKS
        with self.assertRaises(UsageError):
            self.service.handle("edit", '--format blocks "x"', model_text=prose)
        overlap = (
            '{"edits":['
            '{"path":"src/app.py","search":"def run(value):\\n","replace":"def run(value):\\n"},'
            '{"path":"src/app.py","search":"def run","replace":"def RUN"}'
            ']}'
        )
        with self.assertRaises(UsageError) as caught:
            self.service.handle("edit", '--format blocks "x"', model_text=overlap)
        self.assertIn("overlap", str(caught.exception).lower())

    def test_show_and_discard_do_not_write(self):
        original = (self.root / "src" / "app.py").read_bytes()
        self.service.handle("edit", '--format diff "x"', model_text=DIFF)
        shown = "\n".join(self.service.handle("edit", "show"))
        self.assertIn("does not write", shown)
        self.assertIn("+", shown)
        self.service.handle("edit", "discard")
        with self.assertRaises(UsageError):
            self.service.handle("edit", "show")
        self.assertEqual((self.root / "src" / "app.py").read_bytes(), original)

    def test_overlapping_hunks_are_rejected(self):
        text = """--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
-def run(value):
-    return value
+def run(value):
+    return 1
@@ -2,1 +2,1 @@
-    return value
+    return 2
"""
        with self.assertRaises(UsageError):
            self.service.handle("edit", '--format diff "x"', model_text=text)


class ApplyUndoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = new_temp_dir(self)
        self.home = new_temp_dir(self)
        (self.root / "src").mkdir()
        self.target = self.root / "src" / "app.py"
        self.original = b"def run(value):\n    return value\n"
        self.target.write_bytes(self.original)
        os.chmod(self.target, 0o640)
        self.calls = []

        def runner(request):
            self.calls.append(request)
            return DIFF

        self.runner = runner
        self.service = WorkspaceService(
            self.root, home=self.home, runner=runner, destination="openai"
        )
        self.service.handle("file", "add src/app.py")

    def _apply(self, answers):
        stream = iter(answers)

        def confirm(prompt):
            return next(stream)

        return self.service.handle(
            "edit",
            '--format diff "handle empty input"',
            interactive=True,
            confirm=confirm,
        )

    def test_interactive_apply_preserves_newline_and_mode(self):
        lines = self._apply(["y", "y"])
        self.assertEqual(len(self.calls), 1)
        request = self.calls[0]
        self.assertEqual(request.destination, "openai")
        self.assertEqual(request.files[0].relative, "src/app.py")
        self.assertIn("untrusted data", request.prompt)
        self.assertIn("def run(value):", request.prompt)
        self.assertIn("Applied", "\n".join(lines))
        updated = self.target.read_bytes()
        self.assertEqual(updated, b"def run(value):\n    return value or 0\n")
        self.assertTrue(updated.endswith(b"\n"))
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o640)

    def test_apply_question_defaults_to_no(self):
        lines = self._apply(["y", "n"])
        self.assertIn("Not applied", "\n".join(lines))
        self.assertEqual(self.target.read_bytes(), self.original)
        lines = self._apply(["y", ""])
        self.assertEqual(self.target.read_bytes(), self.original)

    def test_second_apply_reasks_and_undo_restores(self):
        self._apply(["y", "n"])
        again = self.service.handle("edit", "apply", interactive=True, confirm=lambda _prompt: "y")
        self.assertIn("Applied", "\n".join(again))
        self.assertEqual(len(self.calls), 1)
        refused = self.service.handle("undo", "last", interactive=True, confirm=lambda _prompt: "n")
        self.assertIn("Not restored", "\n".join(refused))
        self.assertIn(b"return value or 0", self.target.read_bytes())
        undone = self.service.handle("undo", "last", interactive=True, confirm=lambda _prompt: "y")
        self.assertIn("Restored", "\n".join(undone))
        self.assertEqual(self.target.read_bytes(), self.original)
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o640)

    def test_noninteractive_undo_does_not_write(self):
        self._apply(["y", "y"])
        lines = self.service.handle("undo", "last", interactive=False, confirm=lambda _prompt: "y")
        self.assertIn("Not restored", "\n".join(lines))
        self.assertIn("return value or 0", self.target.read_text(encoding="utf-8"))

    def test_stale_hash_refuses_apply(self):
        self._apply(["y", "n"])
        self.target.write_bytes(b"def run(value):\n    return value\nchanged\n")
        with self.assertRaises(UsageError) as caught:
            self.service.handle("edit", "apply", interactive=True, confirm=lambda _prompt: "y")
        self.assertIn("changed", str(caught.exception).lower())
        self.assertIn("changed", self.target.read_text(encoding="utf-8"))

    def test_crlf_and_missing_final_newline_round_trip(self):
        crlf = self.root / "src" / "win.py"
        crlf.write_bytes(b"alpha\r\nbeta\r\n")
        bare = self.root / "src" / "bare.py"
        bare.write_bytes(b"alpha\nbeta")
        self.service.handle("file", "add src/win.py src/bare.py")
        win_diff = """--- a/src/win.py
+++ b/src/win.py
@@ -1,2 +1,2 @@
 alpha
-beta
+BETA
"""
        self.service.handle("edit", '--format diff "win"', model_text=win_diff)
        self.service.handle("edit", "apply", interactive=True, confirm=lambda _prompt: "y")
        self.assertEqual(crlf.read_bytes(), b"alpha\r\nBETA\r\n")
        bare_diff = """--- a/src/bare.py
+++ b/src/bare.py
@@ -1,2 +1,2 @@
 alpha
-beta
+BETA
\\ No newline at end of file
"""
        self.service.handle("edit", '--format diff "bare"', model_text=bare_diff)
        self.service.handle("edit", "apply", interactive=True, confirm=lambda _prompt: "y")
        self.assertEqual(bare.read_bytes(), b"alpha\nBETA")

    def test_mixed_newlines_are_not_rewritten(self):
        mixed = self.root / "src" / "mixed.py"
        mixed.write_bytes(b"alpha\nbeta\r\n")
        self.service.handle("file", "add src/mixed.py")
        diff = """--- a/src/mixed.py
+++ b/src/mixed.py
@@ -1,1 +1,1 @@
-alpha
+ALPHA
"""
        with self.assertRaises(UsageError) as caught:
            self.service.handle("edit", '--format diff "x"', model_text=diff)
        self.assertIn("mixed", str(caught.exception).lower())
        self.assertEqual(mixed.read_bytes(), b"alpha\nbeta\r\n")

    def test_unicode_diff(self):
        text = self.root / "src" / "utf.py"
        text.write_text("café\n", encoding="utf-8")
        self.service.handle("file", "add src/utf.py")
        diff = """--- a/src/utf.py
+++ b/src/utf.py
@@ -1,1 +1,1 @@
-café
+コーヒー
"""
        self.service.handle("edit", '--format diff "x"', model_text=diff)
        self.service.handle("edit", "apply", interactive=True, confirm=lambda _prompt: "y")
        self.assertEqual(text.read_text(encoding="utf-8"), "コーヒー\n")

    def test_multi_file_diff(self):
        other = self.root / "src" / "util.py"
        other.write_bytes(b"VALUE = 1\n")
        self.service.handle("file", "add src/util.py")
        diff = DIFF + """--- a/src/util.py
+++ b/src/util.py
@@ -1,1 +1,1 @@
-VALUE = 1
+VALUE = 2
"""
        self.service.handle("edit", '--format diff "both"', model_text=diff)
        self.service.handle("edit", "apply", interactive=True, confirm=lambda _prompt: "y")
        self.assertIn(b"return value or 0", self.target.read_bytes())
        self.assertEqual(other.read_bytes(), b"VALUE = 2\n")

    def test_backup_failure_writes_nothing(self):
        self._apply(["y", "n"])
        from anyllm.patches import apply_proposal

        proposal = self.service.pending
        with self.assertRaises(UsageError) as caught:
            apply_proposal(proposal, self.root, self.home, faults=Faults(fail_backup=0))
        self.assertIn("No source file was modified", str(caught.exception))
        self.assertEqual(self.target.read_bytes(), self.original)

    def test_replace_failure_rolls_back(self):
        other = self.root / "src" / "util.py"
        other.write_bytes(b"VALUE = 1\n")
        os.chmod(other, 0o640)
        self.service.handle("file", "add src/util.py")
        diff = DIFF + """--- a/src/util.py
+++ b/src/util.py
@@ -1,1 +1,1 @@
-VALUE = 1
+VALUE = 2
"""
        self.service.handle("edit", '--format diff "both"', model_text=diff)
        from anyllm.patches import apply_proposal

        with self.assertRaises(UsageError) as caught:
            apply_proposal(
                self.service.pending,
                self.root,
                self.home,
                faults=Faults(fail_replace=1),
            )
        self.assertIn("restored", str(caught.exception).lower())
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertEqual(other.read_bytes(), b"VALUE = 1\n")

    def test_partial_restore_failure_is_journaled(self):
        other = self.root / "src" / "util.py"
        other.write_bytes(b"VALUE = 1\n")
        self.service.handle("file", "add src/util.py")
        diff = DIFF + """--- a/src/util.py
+++ b/src/util.py
@@ -1,1 +1,1 @@
-VALUE = 1
+VALUE = 2
"""
        self.service.handle("edit", '--format diff "both"', model_text=diff)
        from anyllm.patches import apply_proposal, list_journals

        with self.assertRaises(UsageError) as caught:
            apply_proposal(
                self.service.pending,
                self.root,
                self.home,
                faults=Faults(fail_replace=1, fail_restore=0),
            )
        self.assertIn("partial", str(caught.exception).lower())
        journals = list_journals(self.home)
        self.assertTrue(any(item["state"] == "partial" for item in journals))
        # First file stayed replaced because restore failed; second was not replaced.
        self.assertIn(b"return value or 0", self.target.read_bytes())
        self.assertEqual(other.read_bytes(), b"VALUE = 1\n")

    def test_undo_conflict_and_recovery_do_not_silently_roll_back(self):
        self._apply(["y", "y"])
        self.target.write_text("# user edit\n", encoding="utf-8")
        with self.assertRaises(UsageError) as caught:
            self.service.handle("undo", "last", interactive=True, confirm=lambda _prompt: "y")
        self.assertIn("no longer matches", str(caught.exception).lower())
        self.assertEqual(self.target.read_text(encoding="utf-8"), "# user edit\n")
        report = self.service.recovery_lines()
        self.assertEqual(report, [])
        self.assertEqual(self.target.read_text(encoding="utf-8"), "# user edit\n")

    def test_applying_journal_recovery_restores_after_confirmation(self):
        self._apply(["y", "y"])
        from anyllm.patches import list_journals

        applied = [item for item in list_journals(self.home) if item["state"] == "applied"]
        self.assertEqual(len(applied), 1)
        run_id = applied[0]["id"]
        journal = self.home / "edits" / run_id / "journal.json"
        payload = journal.read_text(encoding="utf-8").replace('"state": "applied"', '"state": "applying"', 1)
        journal.write_text(payload, encoding="utf-8")
        report = "\n".join(self.service.recovery_lines())
        self.assertIn(run_id, report)
        self.assertIn("return value or 0", self.target.read_text(encoding="utf-8"))
        lines = self.service.handle("undo", "recover", interactive=True, confirm=lambda _prompt: "n")
        self.assertIn("Not recovered", "\n".join(lines))
        self.assertIn("return value or 0", self.target.read_text(encoding="utf-8"))
        lines = self.service.handle("undo", "recover", interactive=True, confirm=lambda _prompt: "y")
        self.assertIn("Restored", "\n".join(lines))
        self.assertEqual(self.target.read_bytes(), self.original)

    def test_prepared_journal_recovery_does_not_touch_source(self):
        self._apply(["y", "n"])
        from anyllm.patches import _journal_payload, _write_journal, edits_dir

        proposal = self.service.pending
        run_id = "preparedcase"
        run_dir = edits_dir(self.home) / run_id
        run_dir.mkdir()
        payload = _journal_payload(run_id, proposal, "prepared", [])
        _write_journal(run_dir, payload)
        lines = self.service.handle("undo", "recover", interactive=False)
        self.assertIn("Not recovered", "\n".join(lines))
        self.assertEqual(self.target.read_bytes(), self.original)
        lines = self.service.handle("undo", "recover", interactive=True, confirm=lambda _prompt: "y")
        self.assertIn("not replaced", "\n".join(lines).lower())
        self.assertEqual(self.target.read_bytes(), self.original)

    def test_preview_colors_when_style_enabled(self):
        self.service.handle("edit", '--format diff "x"', model_text=DIFF)
        from anyllm.term import Style

        shown = "\n".join(preview_lines(self.service.pending, Style(True)))
        self.assertIn("\x1b[", shown)
        plain = "\n".join(preview_lines(self.service.pending, Style(False)))
        self.assertNotIn("\x1b", plain)
        self.assertIn("+    return value or 0", plain)

    def test_runner_sees_only_approved_files(self):
        secret = self.root / ".env"
        secret.write_text("API_KEY=hidden-value\n", encoding="utf-8")
        with self.assertRaises(UsageError):
            self.service.handle("file", "add .env")
        self._apply(["y", "n"])
        self.assertEqual(len(self.calls), 1)
        paths = [item.relative for item in self.calls[0].files]
        self.assertEqual(paths, ["src/app.py"])
        self.assertNotIn("hidden-value", self.calls[0].prompt)


if __name__ == "__main__":
    unittest.main()
