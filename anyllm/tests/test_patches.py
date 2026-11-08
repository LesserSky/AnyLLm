"""Proposal parsing, preview, apply, rollback, and undo tests."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

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
        text.write_bytes("café\n".encode("utf-8"))
        self.service.handle("file", "add src/utf.py")
        diff = """--- a/src/utf.py
+++ b/src/utf.py
@@ -1,1 +1,1 @@
-café
+コーヒー
"""
        self.service.handle("edit", '--format diff "x"', model_text=diff)
        self.service.handle("edit", "apply", interactive=True, confirm=lambda _prompt: "y")
        self.assertEqual(text.read_bytes(), "コーヒー\n".encode("utf-8"))

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
        payload = _journal_payload(run_id, proposal, "prepared", [], root=self.root)
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


class ReviewRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = new_temp_dir(self)
        self.home = new_temp_dir(self)
        (self.root / "a.txt").write_bytes(b"old\n")
        (self.root / "b.txt").write_bytes(b"old\n")
        self.service = WorkspaceService(self.root, home=self.home, destination="openai")
        self.service.handle("file", "add a.txt b.txt")

    def _proposal(self, paths):
        from anyllm.patches import parse_proposal

        payload = {
            "edits": [{"path": path, "search": "old", "replace": "new"} for path in paths]
        }
        return parse_proposal(__import__("json").dumps(payload), self.service.workspace.attachments, "blocks")

    def test_review_text_is_visible_before_confirmation(self):
        import io
        from contextlib import redirect_stdout

        seen = []

        def confirm(prompt):
            seen.append(prompt)
            return "y" if prompt.startswith("Approve") else "n"

        self.service.runner = lambda _request: __import__("json").dumps(
            {"edits": [{"path": "a.txt", "search": "old", "replace": "new"}]}
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.service.handle("edit", '--format blocks "change"', interactive=True, confirm=confirm)
        self.assertGreaterEqual(len(seen), 2)
        before_send = buf.getvalue().split("Approve sending")[0]
        self.assertIn("a.txt", before_send)
        self.assertIn("Destination:", before_send)
        self.assertIn("old", (self.root / "a.txt").read_text(encoding="utf-8"))
        self.assertIn("-old", buf.getvalue().split("Apply these changes")[0])
        self.assertEqual((self.root / "a.txt").read_bytes(), b"old\n")

    def test_undo_does_not_cross_workspaces(self):
        from anyllm.patches import apply_proposal, undo_last_applied

        apply_proposal(self._proposal(["a.txt"]), self.root, self.home)
        other = self.root.parent / "other-root"
        other.mkdir()
        (other / "a.txt").write_bytes(b"new\n")
        with self.assertRaises(UsageError) as caught:
            undo_last_applied(other, self.home)
        self.assertIn("workspace", str(caught.exception).lower())
        self.assertEqual((other / "a.txt").read_bytes(), b"new\n")
        self.assertEqual((self.root / "a.txt").read_bytes(), b"new\n")

    def test_undo_last_follows_creation_time_not_uuid_sort(self):
        from anyllm import patches as patchmod

        class Ident:
            def __init__(self, value):
                self.hex = value

        first = self._proposal(["a.txt"])
        second = self._proposal(["b.txt"])
        with mock.patch.object(patchmod.uuid, "uuid4", return_value=Ident("f" * 32)):
            patchmod.apply_proposal(first, self.root, self.home)
        with mock.patch.object(patchmod.uuid, "uuid4", return_value=Ident("0" * 32)):
            patchmod.apply_proposal(second, self.root, self.home)
        result = patchmod.undo_last_applied(self.root, self.home)
        self.assertEqual(result.run_id, "0" * 32)
        self.assertEqual((self.root / "a.txt").read_bytes(), b"new\n")
        self.assertEqual((self.root / "b.txt").read_bytes(), b"old\n")

    def test_stale_later_file_rolls_back_without_touching_it(self):
        from anyllm import patches as patchmod

        proposal = self._proposal(["a.txt", "b.txt"])
        original = patchmod._replace_file

        def swap(root, run_dir, entry):
            if entry["path"] == "b.txt":
                (root / "b.txt").write_bytes(b"user kept\n")
            return original(root, run_dir, entry)

        with self.assertRaises(UsageError):
            with mock.patch.object(patchmod, "_replace_file", swap):
                patchmod.apply_proposal(proposal, self.root, self.home)
        self.assertEqual((self.root / "a.txt").read_bytes(), b"old\n")
        self.assertEqual((self.root / "b.txt").read_bytes(), b"user kept\n")

    def test_rollback_does_not_overwrite_a_newer_edit(self):
        from anyllm import patches as patchmod

        proposal = self._proposal(["a.txt", "b.txt"])
        original = patchmod._replace_file

        def crash(root, run_dir, entry):
            if entry["path"] == "b.txt":
                (root / "a.txt").write_bytes(b"newer user\n")
                raise OSError(28, "disk full")
            return original(root, run_dir, entry)

        with self.assertRaises(UsageError) as caught:
            with mock.patch.object(patchmod, "_replace_file", crash):
                patchmod.apply_proposal(proposal, self.root, self.home)
        self.assertIn("partial", str(caught.exception).lower())
        self.assertEqual((self.root / "a.txt").read_bytes(), b"newer user\n")
        self.assertEqual((self.root / "b.txt").read_bytes(), b"old\n")

    def test_undo_rejects_a_backup_that_does_not_match_the_hash(self):
        from anyllm.patches import apply_proposal, list_journals, undo_last_applied

        result = apply_proposal(self._proposal(["a.txt"]), self.root, self.home)
        journal = list_journals(self.home, self.root)[-1]
        backup = self.home / "edits" / result.run_id / journal["files"][0]["backup"]
        backup.write_bytes(b"corrupted backup\n")
        with self.assertRaises(UsageError) as caught:
            undo_last_applied(self.root, self.home)
        self.assertIn("hash", str(caught.exception).lower())
        self.assertEqual((self.root / "a.txt").read_bytes(), b"new\n")

    def test_prepared_journal_with_replaced_bytes_is_not_called_untouched(self):
        from anyllm import patches as patchmod

        proposal = self._proposal(["a.txt"])
        original = patchmod._write_journal

        def stop(directory, data):
            if data.get("state") == "applying":
                raise KeyboardInterrupt("power loss")
            return original(directory, data)

        with self.assertRaises(KeyboardInterrupt):
            with mock.patch.object(patchmod, "_write_journal", stop):
                patchmod.apply_proposal(proposal, self.root, self.home)
        self.assertEqual((self.root / "a.txt").read_bytes(), b"new\n")
        found = patchmod.unfinished_journals(self.home, self.root)
        self.assertTrue(found)
        recovered = patchmod.recover_journal(self.root, self.home, found[-1]["id"])
        self.assertNotIn("not replaced", " ".join(recovered.messages).lower())
        self.assertEqual((self.root / "a.txt").read_bytes(), b"old\n")

    def test_restoring_state_is_recoverable(self):
        from anyllm.patches import apply_proposal, list_journals, unfinished_journals, _write_journal

        result = apply_proposal(self._proposal(["a.txt"]), self.root, self.home)
        journal = list_journals(self.home, self.root)[-1]
        journal["state"] = "restoring"
        _write_journal(self.home / "edits" / result.run_id, journal)
        self.assertTrue(unfinished_journals(self.home, self.root))

    def test_nul_in_a_proposal_is_rejected(self):
        from anyllm.patches import parse_proposal

        payload = __import__("json").dumps(
            {"edits": [{"path": "a.txt", "search": "old", "replace": "new\x00"}]}
        )
        with self.assertRaises(UsageError):
            parse_proposal(payload, self.service.workspace.attachments, "blocks")
        self.assertEqual((self.root / "a.txt").read_bytes(), b"old\n")

    def test_standard_two_marker_diff_preserves_missing_newline(self):
        bare = self.root / "bare.txt"
        bare.write_bytes(b"beta")
        self.service.handle("file", "add bare.txt")
        diff = (
            "--- a/bare.txt\n+++ b/bare.txt\n@@ -1 +1 @@\n-beta\n"
            "\\ No newline at end of file\n+BETA\n\\ No newline at end of file\n"
        )
        self.service.handle("edit", '--format diff "x"', model_text=diff)
        shown = "\n".join(self.service.handle("edit", "show"))
        self.assertIn("\\ No newline at end of file", shown)
        self.service.handle("edit", "apply", interactive=True, confirm=lambda _prompt: "y")
        self.assertEqual(bare.read_bytes(), b"BETA")


if __name__ == "__main__":
    unittest.main()
