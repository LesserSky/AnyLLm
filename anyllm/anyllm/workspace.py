"""Stage local text files and record consent before they can be sent.

Attachments are explicit, regular, UTF-8 files under one workspace root.
Nothing in this module contacts a provider. A proposal runner, if the caller
supplies one, is invoked only after an interactive send approval.

The integration seam that would share a session runner does not exist yet.
``WorkspaceService`` is the handler interface tests and a thin CLI/REPL
adapter call directly.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from .errors import UsageError
from .patches import (
    EditRequest,
    Proposal,
    apply_proposal,
    build_prompt,
    edits_dir,
    list_journals,
    parse_proposal,
    preview_lines,
    recover_journal,
    undo_last_applied,
    unfinished_journals,
)

DEFAULT_MAX_FILE_BYTES = 256 * 1024
DEFAULT_MAX_TOTAL_BYTES = 1024 * 1024
DEFAULT_MAX_FILES = 20

_CREDENTIAL_NAMES = {
    ".env",
    "keys.json",
    "credentials.json",
    "credentials.yml",
    "credentials.yaml",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
_CREDENTIAL_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".kdbx")

_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("private key block", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    ("aws access key id", r"\bAKIA[0-9A-Z]{16}\b"),
    (
        "credential assignment",
        r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*\S{8,}",
    ),
)


@dataclass(frozen=True)
class Limits:
    """Local caps. They are adjustable by constructing a different Limits."""

    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_files: int = DEFAULT_MAX_FILES


@dataclass(frozen=True)
class Attachment:
    """One staged file. ``raw`` is the on-disk bytes, never a re-encoding."""

    relative: str
    size: int
    sha256: str
    newline: str
    logical_text: str
    raw: bytes
    mode: int
    secret_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Approval:
    """Consent for one exact set of bytes and one exact destination set."""

    destinations: tuple[str, ...]
    files: tuple[tuple[str, str, int], ...]

    def matches(self, destinations: tuple[str, ...], attachments: list[Attachment]) -> bool:
        current = tuple((item.relative, item.sha256, item.size) for item in attachments)
        return self.destinations == destinations and self.files == current


@dataclass
class Workspace:
    """In-process attachment set. Not written into the chat session."""

    root: Path
    limits: Limits = field(default_factory=Limits)
    attachments: dict[str, Attachment] = field(default_factory=dict)
    approval: Approval | None = None

    def __post_init__(self) -> None:
        self.root = establish_root(self.root)

    def set_root(self, path: str | Path) -> list[str]:
        new_root = establish_root(path)
        messages: list[str] = []
        if new_root == self.root:
            messages.append(f"Workspace root is {self.root}")
            return messages
        if self.attachments or self.approval is not None:
            messages.append(
                "Workspace root changed. Attachments and send approval were cleared."
            )
        self.root = new_root
        self.attachments = {}
        self.approval = None
        messages.append(f"Workspace root is {self.root}")
        return messages

    def add(self, relatives: list[str]) -> list[str]:
        if not relatives:
            raise UsageError("Name at least one file to attach.", hint="Example: /file add src/app.py")
        messages: list[str] = []
        staged: dict[str, Attachment] = dict(self.attachments)
        for raw in relatives:
            relative = normalize_relative(raw)
            item = read_attachment(self.root, relative, self.limits)
            staged[item.relative] = item
            if len(staged) > self.limits.max_files:
                raise UsageError(
                    f"At most {self.limits.max_files} attached files are allowed.",
                    hint="Remove a file with /file remove, or raise the local limit.",
                )
            total = sum(entry.size for entry in staged.values())
            if total > self.limits.max_total_bytes:
                raise UsageError(
                    f"Attached files would exceed {self.limits.max_total_bytes} bytes.",
                    hint="Attach fewer or smaller files.",
                )
            flag = ""
            if item.secret_flags:
                flag = "  warning: " + ", ".join(item.secret_flags)
            messages.append(f"staged {item.relative} ({item.size} bytes){flag}")
        self.attachments = staged
        self.approval = None
        messages.append(
            "Files are staged locally only. Nothing is sent until you approve a destination."
        )
        messages.append("Automatic secret detection is incomplete.")
        return messages

    def remove(self, raw: str) -> list[str]:
        relative = normalize_relative(raw)
        if relative not in self.attachments:
            raise UsageError(f"{relative} is not attached.")
        self.attachments.pop(relative)
        self.approval = None
        return [f"removed {relative}", "Send approval was cleared."]

    def clear(self) -> list[str]:
        self.attachments = {}
        self.approval = None
        return ["Attachments cleared.", "Send approval was cleared."]

    def list_lines(self) -> list[str]:
        if not self.attachments:
            return ["No files attached.", f"Workspace root is {self.root}"]
        lines = [f"Workspace root is {self.root}"]
        total = 0
        for item in self._ordered():
            total += item.size
            extra = ""
            if item.secret_flags:
                extra = "  warning: " + ", ".join(item.secret_flags)
            lines.append(f"  {item.relative}  {item.size} bytes  {item.sha256[:12]}{extra}")
        lines.append(f"{len(self.attachments)} files, {total} bytes")
        lines.append("Automatic secret detection is incomplete.")
        return lines

    def show_lines(self, raw: str) -> list[str]:
        relative = normalize_relative(raw)
        item = self.attachments.get(relative)
        if item is None:
            raise UsageError(f"{relative} is not attached.", hint="Stage it with /file add.")
        lines = [
            f"{item.relative}  {item.size} bytes  newline={item.newline}",
        ]
        if item.secret_flags:
            lines.append("warning: " + ", ".join(item.secret_flags))
            lines.append("Automatic secret detection is incomplete.")
        lines.append(escape_terminal_text(item.logical_text))
        return lines

    def approval_lines(self, destination: str) -> list[str]:
        destinations = _destinations(destination)
        items = self._ordered()
        if not items:
            raise UsageError("No files are attached.", hint="Stage files with /file add first.")
        total = sum(item.size for item in items)
        lines = [
            "These attached files would be sent as untrusted data.",
            "Destination: " + ", ".join(destinations),
        ]
        for item in items:
            extra = ""
            if item.secret_flags:
                extra = "  warning: " + ", ".join(item.secret_flags)
            lines.append(f"  {escape_terminal_text(item.relative)}  {item.size} bytes{extra}")
        lines.append(f"Total: {total} bytes in {len(items)} files")
        lines.append("Automatic secret detection is incomplete.")
        return lines

    def grant(self, destination: str, answer: str) -> None:
        if not confirmed(answer):
            self.approval = None
            raise UsageError("Not approved. Nothing was sent.")
        items = self._ordered()
        if not items:
            raise UsageError("No files are attached.")
        self.approval = Approval(
            destinations=_destinations(destination),
            files=tuple((item.relative, item.sha256, item.size) for item in items),
        )

    def fresh_files(self, destination: str) -> list[Attachment]:
        """Re-read approved files. Any byte change drops consent."""
        destinations = _destinations(destination)
        items = self._ordered()
        if self.approval is None or not self.approval.matches(destinations, items):
            raise UsageError(
                "Attachments are not approved for this destination.",
                hint="Approve the listed paths and destination before sending.",
            )
        refreshed: list[Attachment] = []
        changed: list[str] = []
        for item in items:
            current = read_attachment(self.root, item.relative, self.limits)
            if current.sha256 != item.sha256 or current.size != item.size:
                changed.append(item.relative)
            refreshed.append(current)
        if changed:
            self.approval = None
            shown = ", ".join(changed)
            raise UsageError(
                f"Attachment changed since approval: {shown}",
                hint="Review the files and approve the destination again.",
            )
        self.attachments = {item.relative: item for item in refreshed}
        return refreshed

    def _ordered(self) -> list[Attachment]:
        return [self.attachments[key] for key in sorted(self.attachments)]


def confirmed(answer: str | None) -> bool:
    """True only for an unambiguous y. Everything else, including empty, is no."""
    if answer is None:
        return False
    return answer.strip() in {"y", "Y"}


def establish_root(path: str | Path) -> Path:
    """Return a real directory. The root path itself must not be a symlink."""
    raw = str(path).strip()
    if not raw:
        raise UsageError("A workspace root is required.", hint="Example: /file root .")
    _reject_root_syntax(raw)
    candidate = Path(raw).expanduser()
    if candidate.is_symlink():
        raise UsageError(
            "Refusing a symlinked workspace root.",
            hint="Choose the real directory, not a link.",
        )
    if not candidate.exists():
        raise UsageError(f"Workspace root does not exist: {candidate}")
    if not candidate.is_dir():
        raise UsageError("Workspace root must be a directory.")
    real = candidate.resolve(strict=True)
    if not real.is_dir() or real.is_symlink():
        raise UsageError("Workspace root must be a real directory.")
    return real


def normalize_relative(raw: str) -> str:
    """Accept one explicit relative path. Reject traversal and foreign paths."""
    text = raw.strip()
    if not text:
        raise UsageError("A file path is required.")
    _reject_relative_syntax(text)
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise UsageError(
            f"Refusing path {text!r}.",
            hint="Pass a relative file under the workspace root, without . or ..",
        )
    return "/".join(parts)


def credential_filename(name: str) -> bool:
    lower = name.lower()
    if lower in _CREDENTIAL_NAMES:
        return True
    if lower.startswith(".env.") or lower.endswith(".env"):
        return True
    return lower.endswith(_CREDENTIAL_SUFFIXES)


def escape_terminal_text(text: str) -> str:
    """Make text safe to print. Does not change bytes that are written to disk.

    ESC, C1 controls, and other non-text controls are replaced with visible
    ``\\uXXXX`` forms so OSC hyperlinks and cursor sequences cannot run.
    Newlines and tabs are kept so source remains readable.
    """
    pieces: list[str] = []
    for char in text:
        code = ord(char)
        if char in {"\n", "\t"}:
            pieces.append(char)
        elif char == "\r":
            pieces.append("\\u000d")
        elif code < 32 or code == 127 or 0x80 <= code <= 0x9F:
            pieces.append(f"\\u{code:04x}")
        else:
            pieces.append(char)
    return "".join(pieces)


def read_attachment(root: Path, relative: str, limits: Limits) -> Attachment:
    name = relative.rsplit("/", 1)[-1]
    if credential_filename(name):
        raise UsageError(
            f"Refusing {relative}: the name looks like a credential file.",
            hint="Attach a source file. This name is blocked. Automatic detection is incomplete.",
        )
    path = confine_existing_file(root, relative)
    fd = _open_nofollow(path)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise UsageError(f"{relative} is not a regular file.")
        if info.st_size > limits.max_file_bytes:
            raise UsageError(
                f"{relative} is {info.st_size} bytes, above the {limits.max_file_bytes} byte limit."
            )
        raw = os.read(fd, limits.max_file_bytes + 1)
    finally:
        os.close(fd)
    if len(raw) > limits.max_file_bytes:
        raise UsageError(
            f"{relative} is larger than {limits.max_file_bytes} bytes."
        )
    if b"\x00" in raw:
        raise UsageError(f"{relative} looks binary and was not attached.")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeError as exc:
        raise UsageError(f"{relative} is not valid UTF-8 and was not attached.") from exc
    newline, logical = logical_form(decoded)
    flags = secret_flags(decoded)
    return Attachment(
        relative=relative,
        size=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        newline=newline,
        logical_text=logical,
        raw=raw,
        mode=stat.S_IMODE(info.st_mode),
        secret_flags=flags,
    )


def confine_existing_file(root: Path, relative: str) -> Path:
    """Walk ``relative`` under ``root`` without following symlinks."""
    current = root
    if current.is_symlink() or not current.is_dir():
        raise UsageError("Workspace root is not a real directory.")
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise UsageError(
                f"Refusing symlink in {relative}.",
                hint="Attach the real file. Linked parents and linked files are not sent.",
            )
    if not current.exists():
        raise UsageError(f"{relative} does not exist under the workspace root.")
    if current.is_dir():
        raise UsageError(f"{relative} is a directory. Attach a file.")
    if not current.is_file():
        raise UsageError(f"{relative} is not a regular file.")
    real_root = root.resolve()
    real_file = current.resolve()
    try:
        real_file.relative_to(real_root)
    except ValueError as exc:
        raise UsageError(f"{relative} is outside the workspace root.") from exc
    if Path(os.path.abspath(current)) != real_file:
        raise UsageError(f"Refusing symlink in {relative}.")
    return current


def logical_form(text: str) -> tuple[str, str]:
    """Return ``(style, logical_text)``.

    ``logical_text`` uses ``\\n`` and keeps a trailing newline when the file
    had one. Mixed CR, LF, and CRLF is reported as ``mixed`` and left intact
    enough for display, but edits of mixed files are rejected later.
    """
    if text == "":
        return "none", ""
    crlf = text.count("\r\n")
    without = text.replace("\r\n", "")
    lf = without.count("\n")
    cr = without.count("\r")
    kinds = [name for name, count in (("crlf", crlf), ("lf", lf), ("cr", cr)) if count]
    if len(kinds) > 1 or kinds == ["cr"]:
        return "mixed", text.replace("\r\n", "\n").replace("\r", "\n")
    if not kinds:
        return "none", text
    if kinds == ["crlf"]:
        logical = text.replace("\r\n", "\n")
        return "crlf", logical
    return "lf", text


def secret_flags(text: str) -> tuple[str, ...]:
    """Labels only. Matched secret text is never included."""
    import re

    found: list[str] = []
    for label, pattern in _SECRET_PATTERNS:
        if re.search(pattern, text):
            found.append(label)
    return tuple(found)


def _destinations(destination: str) -> tuple[str, ...]:
    name = destination.strip()
    if not name:
        raise UsageError("A destination provider is required before sending files.")
    if any(char.isspace() for char in name) or "/" in name or "\\" in name:
        raise UsageError(f"Destination {destination!r} is not a single provider name.")
    return (name,)


def _reject_root_syntax(raw: str) -> None:
    if "\x00" in raw:
        raise UsageError("Refusing a workspace root that contains a NUL byte.")
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise UsageError("Refusing a UNC or device workspace root.")


def _reject_relative_syntax(text: str) -> None:
    if "\x00" in text:
        raise UsageError("Refusing a path that contains a NUL byte.")
    if text.startswith("/") or text.startswith("\\"):
        raise UsageError(
            f"Refusing absolute path {text!r}.",
            hint="Pass a path relative to the workspace root.",
        )
    if text.startswith("\\\\") or text.startswith("//"):
        raise UsageError(f"Refusing UNC path {text!r}.")
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        raise UsageError(f"Refusing drive path {text!r}.")
    if "\\" in text:
        raise UsageError(
            f"Refusing path {text!r}.",
            hint="Use forward slashes and a relative path under the workspace root.",
        )
    if ":" in text:
        raise UsageError(f"Refusing path {text!r}.")
    if text.startswith("~"):
        raise UsageError(f"Refusing home path {text!r}.")


def _open_nofollow(path: Path) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | directory)
    except (OSError, NotImplementedError, AttributeError):
        dir_fd = None
    try:
        if dir_fd is None:
            return os.open(path, os.O_RDONLY | nofollow)
        return os.open(path.name, os.O_RDONLY | nofollow, dir_fd=dir_fd)
    except OSError as exc:
        raise UsageError(f"Could not read {path.name} without following a link.") from exc
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


class WorkspaceService:
    """Explicit /file, /edit, and /undo handlers.

    ``runner`` receives an ``EditRequest`` and returns model text. It is not
    called unless the user has just approved that destination and file set.
    Noninteractive calls never write source files and never call ``runner``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        home: Path | None = None,
        limits: Limits | None = None,
        runner=None,
        destination: str = "",
        style=None,
        seam=None,
    ) -> None:
        self.workspace = Workspace(root, limits or Limits())
        self.home = home
        self.runner = runner
        self.destination = destination
        self.style = style
        self.seam = seam
        self.pending: Proposal | None = None

    def handle(
        self,
        command: str,
        rest: str = "",
        *,
        interactive: bool = False,
        confirm=None,
        model_text: str | None = None,
    ) -> list[str]:
        name = command.strip().lower()
        if name == "file":
            return self._file(rest)
        if name == "edit":
            return self._edit(rest, interactive=interactive, confirm=confirm, model_text=model_text)
        if name == "undo":
            return self._undo(rest, interactive=interactive, confirm=confirm)
        raise UsageError(f"Unknown workspace command {command!r}.")

    def recovery_lines(self) -> list[str]:
        home = self._home()
        found = unfinished_journals(home)
        if not found:
            return []
        lines = [
            "Unfinished edit journals were found. Source files were not changed automatically.",
        ]
        for item in found:
            paths = ", ".join(entry["path"] for entry in item["files"])
            lines.append(
                f"  {item['id']} state={item['state']} files: {paths}"
            )
        lines.append("Review them with /undo list. Recovery does not run unless you confirm it.")
        return lines

    def _file(self, rest: str) -> list[str]:
        parts = rest.split()
        if not parts:
            raise UsageError("Usage: /file root|add|list|show|remove|clear")
        action = parts[0].lower()
        args = parts[1:]
        if action == "root":
            if len(args) != 1:
                raise UsageError("Usage: /file root PATH")
            self.pending = None
            self._drop_published()
            return self.workspace.set_root(args[0])
        if action == "add":
            self.pending = None
            self._drop_published()
            return self.workspace.add(args)
        if action == "list":
            if args:
                raise UsageError("Usage: /file list")
            return self.workspace.list_lines()
        if action == "show":
            if len(args) != 1:
                raise UsageError("Usage: /file show PATH")
            return self.workspace.show_lines(args[0])
        if action == "remove":
            if len(args) != 1:
                raise UsageError("Usage: /file remove PATH")
            self.pending = None
            self._drop_published()
            return self.workspace.remove(args[0])
        if action == "clear":
            if args:
                raise UsageError("Usage: /file clear")
            self.pending = None
            self._drop_published()
            return self.workspace.clear()
        raise UsageError(
            f"Unknown /file action {action!r}.",
            hint="Use root, add, list, show, remove, or clear.",
        )

    def _edit(
        self,
        rest: str,
        *,
        interactive: bool,
        confirm,
        model_text: str | None,
    ) -> list[str]:
        parts = rest.split(None, 1) if rest.strip() else []
        if parts and parts[0].lower() == "show":
            return self._edit_show()
        if parts and parts[0].lower() == "discard":
            return self._edit_discard()
        if parts and parts[0].lower() == "apply":
            return self._edit_apply(interactive=interactive, confirm=confirm)
        fmt, instruction = _parse_edit_request(rest)
        if model_text is None:
            model_text = self._generate(fmt, instruction, interactive=interactive, confirm=confirm)
        proposal = parse_proposal(model_text, self.workspace.attachments, fmt)
        self.pending = proposal
        lines = preview_lines(proposal, style=self.style)
        lines.append(f"Proposal {proposal.request_id} touches {len(proposal.files)} file(s).")
        if not interactive:
            lines.append("Not applied. Noninteractive mode does not write source files.")
            return lines
        count = len(proposal.files)
        prompt = f"Apply these changes to {count} file{'s' if count != 1 else ''}? [y/N] "
        answer = "" if confirm is None else confirm(prompt)
        lines.append(prompt + answer)
        if not confirmed(answer):
            lines.append("Not applied. Source files were not changed.")
            return lines
        lines.extend(self._apply_pending())
        return lines

    def _generate(self, fmt: str, instruction: str, *, interactive: bool, confirm) -> str:
        if not interactive:
            raise UsageError(
                "Refusing to send attachments in noninteractive mode.",
                hint="Pass proposal text from a file, or run /edit in the interactive REPL.",
            )
        if self.runner is None and self.seam is None:
            raise UsageError(
                "No proposal runner is connected.",
                hint="Nothing was sent and no source file was written.",
            )
        if not self.destination:
            raise UsageError("Set a destination provider before /edit can send files.")
        lines = self.workspace.approval_lines(self.destination)
        prompt = "Approve sending these files to " + self.destination + "? [y/N] "
        answer = "" if confirm is None else confirm(prompt)
        if not confirmed(answer):
            raise UsageError("Not approved. Nothing was sent.")
        self.workspace.grant(self.destination, answer)
        files = self.workspace.fresh_files(self.destination)
        request = EditRequest(
            instruction=instruction,
            format=fmt,
            destination=self.destination,
            files=tuple(files),
            prompt=build_prompt(instruction, fmt, files, self.destination),
        )
        if self.seam is not None:
            text = self.seam.request(files, request)
        else:
            text = self.runner(request)
        if not isinstance(text, str):
            raise UsageError("The proposal runner did not return text. Nothing was written.")
        return text

    def _drop_published(self) -> None:
        if self.seam is not None:
            self.seam.revoke_published()

    def _edit_show(self) -> list[str]:
        if self.pending is None:
            raise UsageError("No edit proposal is pending.")
        lines = preview_lines(self.pending, style=self.style)
        lines.append("Show does not write source files.")
        return lines

    def _edit_discard(self) -> list[str]:
        if self.pending is None:
            return ["No edit proposal was pending."]
        self.pending = None
        return ["Proposal discarded. Source files were not changed."]

    def _edit_apply(self, *, interactive: bool, confirm) -> list[str]:
        if self.pending is None:
            raise UsageError("No edit proposal is pending.")
        lines = preview_lines(self.pending, style=self.style)
        if not interactive:
            lines.append("Not applied. Noninteractive mode does not write source files.")
            return lines
        count = len(self.pending.files)
        prompt = f"Apply these changes to {count} file{'s' if count != 1 else ''}? [y/N] "
        answer = "" if confirm is None else confirm(prompt)
        lines.append(prompt + answer)
        if not confirmed(answer):
            lines.append("Not applied. Source files were not changed.")
            return lines
        lines.extend(self._apply_pending())
        return lines

    def _apply_pending(self) -> list[str]:
        proposal = self.pending
        if proposal is None:
            raise UsageError("No edit proposal is pending.")
        result = apply_proposal(proposal, self.workspace.root, self._home())
        self.pending = None
        self.workspace.approval = None
        return result.messages

    def _undo(self, rest: str, *, interactive: bool, confirm) -> list[str]:
        action = (rest.strip().split() or ["list"])[0].lower()
        home = self._home()
        if action == "list":
            rows = list_journals(home)
            if not rows:
                return ["No edit journals."]
            lines = ["Edit journals (backups stay on disk; they are not deleted):"]
            for item in rows:
                paths = ", ".join(entry["path"] for entry in item["files"])
                lines.append(f"  {item['id']}  {item['state']}  {paths}")
            lines.append(f"Backup directory: {edits_dir(home)}")
            return lines
        if action == "recover":
            return self._recover(interactive=interactive, confirm=confirm)
        if action != "last":
            raise UsageError("Usage: /undo list | /undo last")
        if not interactive:
            found = list_journals(home)
            applied = [item for item in found if item["state"] == "applied"]
            if not applied:
                raise UsageError("No applied edit to undo.")
            lines = [
                f"Undo would restore {len(applied[-1]['files'])} file(s) from {applied[-1]['id']}."
            ]
            lines.append("Not restored. Noninteractive mode does not write source files.")
            return lines
        latest = [item for item in list_journals(home) if item["state"] == "applied"]
        if not latest:
            raise UsageError("No applied edit to undo.")
        count = len(latest[-1]["files"])
        prompt = f"Restore {count} file{'s' if count != 1 else ''} from backup? [y/N] "
        answer = "" if confirm is None else confirm(prompt)
        if not confirmed(answer):
            return [prompt + (answer or ""), "Not restored. Source files were not changed."]
        result = undo_last_applied(self.workspace.root, home)
        self.workspace.approval = None
        return [prompt + answer, *result.messages]

    def _recover(self, *, interactive: bool, confirm) -> list[str]:
        """Reviewed recovery. Never called implicitly, and never without y."""
        home = self._home()
        found = unfinished_journals(home)
        if not found:
            return ["No unfinished edit journals."]
        lines = ["Unfinished journals:"]
        for item in found:
            paths = ", ".join(entry["path"] for entry in item["files"])
            lines.append(f"  {item['id']} state={item['state']} {paths}")
        if not interactive:
            lines.append("Not recovered. Noninteractive mode does not write source files.")
            return lines
        prompt = f"Recover {len(found)} unfinished journal(s)? [y/N] "
        answer = "" if confirm is None else confirm(prompt)
        lines.append(prompt + (answer or ""))
        if not confirmed(answer):
            lines.append("Not recovered. Source files were not changed.")
            return lines
        for item in found:
            lines.extend(
                recover_journal(self.workspace.root, home, item["id"]).messages
            )
        return lines

    def _home(self) -> Path:
        if self.home is not None:
            return self.home
        from .config import ensure_config_dir

        return ensure_config_dir()


def _parse_edit_request(rest: str) -> tuple[str, str]:
    fmt = "diff"
    text = rest.strip()
    if text.startswith("--format"):
        parts = text.split(None, 2)
        if len(parts) < 3 or parts[0] != "--format":
            raise UsageError('Usage: /edit --format diff|blocks "instruction"')
        fmt = parts[1].lower()
        text = parts[2].strip()
    if fmt not in {"diff", "blocks"}:
        raise UsageError("Edit format must be diff or blocks.")
    if not text:
        raise UsageError("An edit instruction is required.")
    return fmt, text
