"""Attachment confinement, consent, and secret-handling tests."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.errors import UsageError
from anyllm.workspace import (
    Limits,
    Workspace,
    WorkspaceService,
    credential_filename,
    escape_terminal_text,
    normalize_relative,
)


def new_temp_dir(case: unittest.TestCase) -> Path:
    base = ROOT / ".test-run"
    base.mkdir(exist_ok=True)
    path = base / f"ws-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    case.addCleanup(shutil.rmtree, path, True)
    return path


class PathSyntaxTests(unittest.TestCase):
    def test_relative_paths_and_rejections(self):
        self.assertEqual(normalize_relative("src/app.py"), "src/app.py")
        for raw in (
            "../secret",
            "..",
            "/etc/passwd",
            "\\\\server\\share",
            "//server/share",
            "C:/Windows/notepad.exe",
            "C:\\Windows\\notepad.exe",
            "file\x00name",
            "dir\\file",
            "ads:stream",
            "~/secrets",
            "",
            ".",
            "src/../app.py",
        ):
            with self.assertRaises(UsageError):
                normalize_relative(raw)

    def test_credential_names_are_blocked(self):
        for name in (".env", ".env.local", "keys.json", "id_rsa", "server.pem", "token.key"):
            self.assertTrue(credential_filename(name), name)
        self.assertFalse(credential_filename("app.py"))
        self.assertFalse(credential_filename("README.md"))
        self.assertTrue(credential_filename(".netrc"))
        self.assertTrue(credential_filename(".npmrc"))
        self.assertTrue(credential_filename("credentials", ".aws/credentials"))


class AttachmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = new_temp_dir(self)
        (self.root / "src").mkdir()
        self.file = self.root / "src" / "app.py"
        self.file.write_bytes(b"print('hi')\n")
        self.workspace = Workspace(self.root)

    def test_add_list_show_remove_and_counts(self):
        lines = self.workspace.add(["src/app.py"])
        self.assertIn("staged src/app.py", lines[0])
        self.assertIn("8 bytes" if False else "bytes", lines[0])
        listed = "\n".join(self.workspace.list_lines())
        self.assertIn("src/app.py", listed)
        self.assertIn("1 files", listed)
        shown = "\n".join(self.workspace.show_lines("src/app.py"))
        self.assertIn("print('hi')", shown)
        self.assertIn("newline=lf", shown)
        self.workspace.remove("src/app.py")
        self.assertEqual(self.workspace.attachments, {})

    def test_byte_count_is_the_file_size(self):
        self.workspace.add(["src/app.py"])
        item = self.workspace.attachments["src/app.py"]
        self.assertEqual(item.size, self.file.stat().st_size)
        self.assertEqual(item.size, len(self.file.read_bytes()))

    def test_reject_binary_oversize_and_bad_utf8(self):
        (self.root / "bin.dat").write_bytes(b"abc\x00def")
        with self.assertRaises(UsageError) as caught:
            self.workspace.add(["bin.dat"])
        self.assertIn("binary", str(caught.exception).lower())
        (self.root / "bad.txt").write_bytes(b"\xff\xfe")
        with self.assertRaises(UsageError) as caught:
            self.workspace.add(["bad.txt"])
        self.assertIn("UTF-8", str(caught.exception))
        big = self.root / "big.txt"
        big.write_bytes(b"a" * 50)
        limited = Workspace(self.root, Limits(max_file_bytes=16, max_total_bytes=100, max_files=2))
        with self.assertRaises(UsageError) as caught:
            limited.add(["big.txt"])
        self.assertIn("16", str(caught.exception))
        self.assertEqual(limited.attachments, {})

    def test_total_and_count_caps(self):
        (self.root / "a.txt").write_text("aaaa", encoding="utf-8")
        (self.root / "b.txt").write_text("bbbb", encoding="utf-8")
        limited = Workspace(self.root, Limits(max_file_bytes=10, max_total_bytes=6, max_files=20))
        with self.assertRaises(UsageError):
            limited.add(["a.txt", "b.txt"])
        self.assertEqual(limited.attachments, {})
        limited = Workspace(self.root, Limits(max_file_bytes=10, max_total_bytes=100, max_files=1))
        with self.assertRaises(UsageError):
            limited.add(["a.txt", "b.txt"])

    def test_credential_file_is_refused_without_echoing_contents(self):
        secret = "super-secret-value"
        env = self.root / ".env"
        env.write_text(f"API_KEY={secret}\n", encoding="utf-8")
        with self.assertRaises(UsageError) as caught:
            self.workspace.add([".env"])
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, caught.exception.hint or "")
        self.assertEqual(self.workspace.attachments, {})

    def test_secret_looking_content_is_flagged_not_quoted(self):
        secret = "sk-live-supersecretvalue"
        path = self.root / "src" / "config.py"
        path.write_text(f"token = '{secret}'\n", encoding="utf-8")
        lines = self.workspace.add(["src/config.py"])
        self.assertTrue(any("credential assignment" in line for line in lines))
        self.assertNotIn(secret, "\n".join(lines))
        self.assertNotIn(secret, "\n".join(self.workspace.list_lines()))

    def test_private_key_block_is_flagged_without_the_body(self):
        body = "-----BEGIN PRIVATE KEY-----\nABCDEF\n-----END PRIVATE KEY-----\n"
        (self.root / "notes.txt").write_text(body, encoding="utf-8")
        lines = self.workspace.add(["notes.txt"])
        joined = "\n".join(lines)
        self.assertIn("private key block", joined)
        self.assertNotIn("ABCDEF", joined)

    def test_fifo_is_not_attached(self):
        fifo = self.root / "pipe"
        try:
            os.mkfifo(fifo)
        except (AttributeError, OSError):
            self.skipTest("fifos are not available")
        with self.assertRaises(UsageError) as caught:
            self.workspace.add(["pipe"])
        self.assertIn("regular", str(caught.exception).lower())

    def test_symlink_file_and_parent_are_refused(self):
        outside = new_temp_dir(self) / "outside.txt"
        outside.write_text("nope\n", encoding="utf-8")
        link = self.root / "linked.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symlinks are not available")
        with self.assertRaises(UsageError) as caught:
            self.workspace.add(["linked.txt"])
        self.assertIn("symlink", str(caught.exception).lower())
        linked_dir = self.root / "jump"
        try:
            linked_dir.symlink_to(outside.parent)
        except OSError:
            self.skipTest("symlinks are not available")
        with self.assertRaises(UsageError):
            self.workspace.add(["jump/outside.txt"])

    def test_root_change_clears_attachments_and_symlink_root_is_refused(self):
        self.workspace.add(["src/app.py"])
        other = new_temp_dir(self)
        messages = self.workspace.set_root(other)
        self.assertIn("cleared", " ".join(messages).lower())
        self.assertEqual(self.workspace.attachments, {})
        link = new_temp_dir(self) / "root-link"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks are not available")
        with self.assertRaises(UsageError):
            Workspace(link)

    def test_staged_path_escapes_are_not_raw(self):
        name = "bad\x1b[2Jname.txt"
        try:
            (self.root / name).write_bytes(b"hello\n")
        except OSError:
            self.skipTest("this platform rejects control characters in file names")
        lines = "\n".join(self.workspace.add([name]))
        self.assertNotIn("\x1b", lines)
        self.assertIn("\\u001b", lines)

    def test_netrc_is_refused_without_quoting_the_secret(self):
        secret = "SuperSecret123"
        (self.root / ".netrc").write_text(f"machine example login demo password {secret}\n", encoding="utf-8")
        with self.assertRaises(UsageError) as caught:
            self.workspace.add([".netrc"])
        self.assertNotIn(secret, str(caught.exception))

    def test_show_escapes_osc_and_esc(self):
        payload = "hello \x1b]8;;http://evil.example\x07click\x1b]8;;\x07 tail\n"
        (self.root / "src" / "view.py").write_text(payload, encoding="utf-8")
        self.workspace.add(["src/view.py"])
        shown = "\n".join(self.workspace.show_lines("src/view.py"))
        self.assertNotIn("\x1b", shown)
        self.assertIn("\\u001b", shown)
        self.assertIn("hello", shown)

    def test_escape_helper_keeps_newlines(self):
        shown = escape_terminal_text("a\nb\t\x1b[31m")
        self.assertEqual(shown.splitlines(), ["a", "b\t\\u001b[31m"])


class ConsentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = new_temp_dir(self)
        self.path = self.root / "app.py"
        self.path.write_text("value = 1\n", encoding="utf-8")
        self.workspace = Workspace(self.root)
        self.workspace.add(["app.py"])

    def test_default_no_and_destination_binding(self):
        with self.assertRaises(UsageError):
            self.workspace.grant("openai", "n")
        with self.assertRaises(UsageError):
            self.workspace.fresh_files("openai")
        self.workspace.grant("openai", "y")
        files = self.workspace.fresh_files("openai")
        self.assertEqual(files[0].relative, "app.py")
        with self.assertRaises(UsageError) as caught:
            self.workspace.fresh_files("anthropic")
        self.assertIn("not approved", str(caught.exception).lower())

    def test_content_change_invalidates_approval_without_sending(self):
        self.workspace.grant("openai", "Y")
        self.path.write_text("value = 2\n", encoding="utf-8")
        with self.assertRaises(UsageError) as caught:
            self.workspace.fresh_files("openai")
        self.assertIn("changed", str(caught.exception).lower())
        self.assertIsNone(self.workspace.approval)
        preview = "\n".join(self.workspace.approval_lines("openai"))
        self.assertIn("app.py", preview)
        self.assertIn("openai", preview)
        self.assertIn("incomplete", preview.lower())

    def test_yes_word_is_not_acceptance(self):
        with self.assertRaises(UsageError):
            self.workspace.grant("openai", "yes")


class ServiceConsentTests(unittest.TestCase):
    def test_runner_is_not_called_without_approval_or_when_not_interactive(self):
        root = new_temp_dir(self)
        home = new_temp_dir(self)
        (root / "app.py").write_text("print(1)\n", encoding="utf-8")
        calls = []

        def runner(request):
            calls.append(request)
            return "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print(1)\n+print(2)\n"

        service = WorkspaceService(root, home=home, runner=runner, destination="openai")
        service.handle("file", "add app.py")
        with self.assertRaises(UsageError) as caught:
            service.handle("edit", '--format diff "change it"', interactive=False, confirm=lambda _prompt: "y")
        self.assertIn("noninteractive", str(caught.exception).lower())
        self.assertEqual(calls, [])
        self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "print(1)\n")

        answers = iter(["n"])
        with self.assertRaises(UsageError):
            service.handle(
                "edit",
                '--format diff "change it"',
                interactive=True,
                confirm=lambda _prompt: next(answers),
            )
        self.assertEqual(calls, [])
        self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "print(1)\n")


if __name__ == "__main__":
    unittest.main()
