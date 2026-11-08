"""Adapters from workspace consent to the Package A seam.

The live module is ``anyllm.seam`` on ``feat/orchestration`` (pull request 3).
This branch does not import it at load time and does not merge that branch.
``load_orchestration_seam`` returns the module only when it is already
installed. Tests pass a fake with the same method and attribute names.

Verified signatures, read from that branch and not copied into this tree:

* ``ApprovedAttachment(path, sha256, content)`` and ``ref()``.
* ``AttachmentStore.approve(attachment)``, ``revoke(path)``, ``clear()``,
  ``list()``. ``attachment_store()`` is the process-wide store.
* ``make_snapshot(history, *, system, temperature, max_tokens, attachments,
  record_refs=False, snapshot_id=None)`` and ``snapshot_from_session``.
* ``run_request(provider, target, snapshot, options, *, credential, prompt="",
  messages=None, cancel_event=None, event_queue=None)`` returns ``RunResult``
  with ``text``, ``status``, ``error``, ``usage``. It does not print.
* ``RunOptions(timeout, deadline, temperature, max_tokens, show_reasoning,
  tools_payload)``.
* ``RunEvent(kind, target, status="", detail="")``.
* ``Presentation.line / status_line / warn / block(title, body, *, meta="")``.
* ``register_repl(name, handler, help_row)`` where ``handler`` is
  ``(repl, rest) -> None``.
* ``Session.append_exchange(user_text, assistant_text, *, usage=None)`` is not
  called for proposals. Adopting an edit into the chat is a later explicit step.
* ``Target(provider, model, base_url, alias=None)`` lives in ``targets.py``.
  ``run_request`` only reads those attributes.
* ``load_feature`` / ``save_feature`` are version ``1`` and are not used for
  attachment bodies.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field

from .errors import UsageError

# Help rows match register_repl's (command, action) pairs.
FILE_HELP = ("/file ...", "stage files under the workspace root")
EDIT_HELP = ("/edit ...", "preview a reviewed edit; apply only after y")
UNDO_HELP = ("/undo ...", "list or restore an applied edit from backup")


@dataclass(frozen=True)
class TargetView:
    """Attribute-compatible with ``targets.Target``. Holds no secret."""

    provider: str
    model: str
    base_url: str
    alias: str | None = None


def load_orchestration_seam():
    """Return ``anyllm.seam`` when Package A is installed, else None."""

    try:
        from . import seam as module
    except ImportError:
        return None
    required = (
        "ApprovedAttachment",
        "attachment_store",
        "make_snapshot",
        "run_request",
        "RunOptions",
        "RunResult",
        "Presentation",
        "register_repl",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise UsageError(
            "The installed seam is missing " + ", ".join(missing) + ".",
            hint="Do not send files until the orchestration seam matches Package A.",
        )
    return module


def publish_consent(store, files, factory) -> tuple:
    """Approve exact file bytes after the user has consented.

    ``factory`` is ``ApprovedAttachment`` when the live seam is present.
    Previously published paths are the caller's responsibility to revoke
    first. This function does not call ``store.clear()``.
    """

    published = []
    try:
        for item in files:
            content = item.raw.decode("utf-8")
            approved = factory(path=item.relative, sha256=item.sha256, content=content)
            store.approve(approved)
            published.append(approved)
    except Exception:
        for item in published:
            store.revoke(item.path)
        raise
    return tuple(published)


def revoke_paths(store, paths) -> None:
    for path in paths:
        store.revoke(path)


@dataclass
class SeamPort:
    """One edit proposal through ``run_request``. No second call, no writes."""

    store: object
    make_snapshot: object
    run_request: object
    factory: object
    provider_obj: object
    target: object
    credential: object
    options: object
    history: object = field(default_factory=tuple)
    system: str | None = None
    event_queue: object | None = None
    cancel_event: object | None = None
    destination_now: object | None = None
    published: list = field(default_factory=list)
    last_snapshot: object | None = None
    last_result: object | None = None

    def revoke_published(self) -> None:
        revoke_paths(self.store, list(self.published))
        self.published.clear()

    def request(self, files, edit_request) -> str:
        self.revoke_published()
        approved = publish_consent(self.store, files, self.factory)
        self.published.extend(item.relative for item in files)
        history = self.history() if callable(self.history) else self.history
        system = self.system() if callable(self.system) else self.system
        snapshot = self.make_snapshot(
            history or (),
            system=system,
            temperature=None,
            max_tokens=None,
            attachments=approved,
            record_refs=False,
        )
        if getattr(snapshot, "credential", None) or getattr(snapshot, "api_key", None):
            raise UsageError(
                "Refusing a snapshot that carries a credential.",
                hint="Nothing was written.",
            )
        if self.destination_now is not None and self.destination_now() != edit_request.destination:
            self.revoke_published()
            raise UsageError(
                "Destination changed before the request was sent. Nothing further was sent.",
                hint="Approve the new provider, model, and endpoint.",
            )
        provider = self.provider_obj() if callable(self.provider_obj) else self.provider_obj
        target = self.target() if callable(self.target) else self.target
        options = self.options() if callable(self.options) else self.options
        credential = self.credential() if callable(self.credential) else self.credential
        result = self.run_request(
            provider,
            target,
            snapshot,
            options,
            credential=credential,
            prompt=edit_request.prompt,
            messages=None,
            cancel_event=self.cancel_event,
            event_queue=self.event_queue,
        )
        self.last_snapshot = snapshot
        self.last_result = result
        status = getattr(result, "status", "")
        if status != "ok":
            detail = getattr(result, "error", None) or status or "error"
            raise UsageError(
                f"Edit proposal was not produced: {detail}",
                hint="No source file was written.",
            )
        text = getattr(result, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise UsageError("The proposal response was empty. Nothing was written.")
        return text


def drain_events(event_queue, presentation) -> None:
    """Render queued status on the main thread. Workers must not print."""

    if event_queue is None or presentation is None:
        return
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            return
        label = getattr(event, "target", "")
        status = getattr(event, "status", "")
        presentation.status_line(f"{label}: {status}".strip(": "))


def present_lines(lines, repl) -> None:
    """Prefer ``Presentation.block`` when the REPL has one."""

    presentation = getattr(repl, "presentation", None)
    if presentation is not None:
        presentation.block("workspace", "\n".join(lines))
        return
    say = getattr(repl, "say", None)
    for line in lines:
        if say is not None:
            say(line)
        else:
            print(line)


def register_workspace(register_repl) -> None:
    """Register /file, /edit, and /undo. ``register_repl`` matches Package A."""

    register_repl("file", _repl_command("file"), FILE_HELP)
    register_repl("edit", _repl_command("edit"), EDIT_HELP)
    register_repl("undo", _repl_command("undo"), UNDO_HELP)


def bind_service(service, repl) -> bool:
    """Attach the live seam when it is installed. Return whether that happened.

    The API key is read only as ``run_request(..., credential=...)``. It is
    not copied onto the snapshot. This does not modify ``repl.py``.
    """

    module = load_orchestration_seam()
    if module is None or repl is None or not hasattr(repl, "session"):
        return False
    from .workspace import destination_key

    def session():
        return repl.session

    def history():
        return list(session().messages)

    def system():
        return session().system

    def provider():
        return session().provider

    def target():
        current = session()
        return TargetView(
            provider=current.provider.name,
            model=current.model,
            base_url=current.base_url or "",
            alias=None,
        )

    def credential():
        return session().api_key

    def options():
        current = session()
        return module.RunOptions(
            timeout=getattr(current, "timeout", 120.0),
            max_tokens=current.max_tokens,
            temperature=current.temperature,
            show_reasoning=False,
        )

    def destination():
        current = session()
        return destination_key(current.provider.name, current.model, current.base_url or "")

    service.destination_resolver = destination
    service.seam = SeamPort(
        store=module.attachment_store(),
        make_snapshot=module.make_snapshot,
        run_request=module.run_request,
        factory=module.ApprovedAttachment,
        provider_obj=provider,
        target=target,
        credential=credential,
        options=options,
        history=history,
        system=system,
        event_queue=queue.Queue(),
        destination_now=destination,
    )
    service.style = getattr(repl, "style", service.style)
    return True


def dispatch_repl(repl, command: str, rest: str) -> None:
    """Handle one workspace slash command on the main thread.

    Proposal generation uses the seam only when ``bind_service`` finds it.
    Otherwise ``/edit`` refuses before any network call. Apply and undo write
    only when both stdin and stdout are terminals and the user answers ``y``.
    """

    service = getattr(repl, "_workspace_service", None)
    if service is None:
        root = getattr(repl, "workspace_root", None) or "."
        service = __import__(
            "anyllm.workspace", fromlist=["WorkspaceService"]
        ).WorkspaceService(
            root,
            style=getattr(repl, "style", None),
            destination=_destination(repl),
        )
        repl._workspace_service = service
        bind_service(service, repl)
        for line in service.recovery_lines():
            present_lines([line], repl)
    interactive = _stdio_is_interactive()

    def confirm(prompt: str) -> str:
        return input(prompt)

    def emit(line: str) -> None:
        present_lines([line], repl)

    service.handle(command, rest, interactive=interactive, confirm=confirm, emit=emit)
    if service.seam is not None:
        drain_events(service.seam.event_queue, getattr(repl, "presentation", None))


def _destination(repl) -> str:
    session = getattr(repl, "session", None)
    provider = getattr(session, "provider", None)
    name = getattr(provider, "name", "")
    return name or ""


def _stdio_is_interactive() -> bool:
    import sys

    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _repl_command(command: str):
    def handler(repl, rest: str) -> None:
        dispatch_repl(repl, command, rest)

    return handler


def wiring_notes() -> str:
    """Later coordinator steps. Not executed on this branch."""

    return """\
This branch already dispatches /file, /edit, and /undo from repl.py through
dispatch_repl. The CLI commands file, edit, and undo never set interactive
mode, so they cannot write source files. /edit does not call a provider
until anyllm.seam is installed; bind_service returns false without it.

After feat/orchestration is merged:

1. Do not copy seam.py. Import anyllm.seam.
2. Keep the repl.py _cmd_file/_cmd_edit/_cmd_undo dispatch. Do not also
   register_workspace(register_repl) for those same names, or a REPL that
   checks both tables could run the command twice.
3. Give the REPL a Presentation as repl.presentation so previews use
   Presentation.block and RunEvent status uses status_line on the main thread.
4. bind_service already publishes consent with attachment_store().approve
   and sends the proposal with run_request when the seam imports. The
   credential stays a keyword argument.
5. Do not call Session.append_exchange for a proposal. File bodies stay out of
   Session.messages unless a later command explicitly adopts text.
6. load_feature/save_feature stay for Package A JSON. Edit journals remain under
   ANYLLM_HOME/edits and are not feature files.
"""
