"""Integration seam for model orchestration and later workspace features.

Package B (file edits, undo, exports, markdown, CI) should call this module
instead of duplicating provider calls or storing feature payloads inside
``Session.messages``.

Contracts:

* ``RequestSnapshot`` is immutable and contains no credentials.
* ``run_request`` performs one completion. It never prints. Streaming events
  are queued for the main thread; worker threads must not touch the terminal.
* ``Presentation`` is the main-thread rendering interface. A later markdown
  renderer can wrap ``block`` without changing team scheduling.
* Feature JSON under the config directory is versioned and separate from
  conversation transcripts. Attachments approved for a process live only in
  ``AttachmentStore`` (memory). Package A does not read the filesystem to
  discover files; Package B fills the store after explicit user consent.
* REPL handlers register through ``register_repl`` so new commands do not
  need a new ``if`` chain in the REPL.

Cancellation is cooperative. A blocking HTTP read cannot be killed; an
in-flight provider request may still finish and incur charges. Callers bound
that window with ``RunOptions.timeout`` and an overall deadline.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from .adapters import Usage, stream_chat
from .config import config_dir, write_json, read_json
from .errors import AnyLLMError, NetworkError, ProviderError

SNAPSHOT_VERSION = 1
FEATURE_STATE_VERSION = 1

CANCELLATION_NOTE = (
    "Cancellation stops scheduling and printing. An HTTP read already in "
    "flight cannot be killed, so that provider request may still finish and "
    "incur charges. Timeouts bound the wait; this is not a hard cancel."
)


@dataclass(frozen=True)
class ApprovedAttachment:
    """File text the user already approved for this process.

    ``content`` stays in memory. Persisted artifacts store a path and hash
    only when a caller sets ``record_refs`` on the snapshot, and never the body.
    """

    path: str
    sha256: str
    content: str

    def ref(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": len(self.content.encode("utf-8")),
        }


@dataclass(frozen=True)
class RequestSnapshot:
    """Frozen view of a conversation at the moment a fan-out starts."""

    version: int
    snapshot_id: str
    history: tuple[tuple[str, str], ...]
    system: str | None
    temperature: float | None
    max_tokens: int | None
    attachments: tuple[ApprovedAttachment, ...] = ()
    record_refs: bool = False

    def attachment_map(self) -> dict[str, ApprovedAttachment]:
        return {item.path: item for item in self.attachments}


@dataclass
class RunOptions:
    """Per-call limits. ``deadline`` is an absolute ``time.perf_counter`` value."""

    timeout: float = 120.0
    deadline: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    show_reasoning: bool = False
    tools_payload: list | None = None


@dataclass(frozen=True)
class RunEvent:
    """Status for the main thread. Workers enqueue these and do not print."""

    kind: str
    target: str
    status: str = ""
    detail: str = ""


@dataclass
class RunResult:
    """One completion. ``usage`` is None when the provider reported nothing."""

    status: str
    text: str = ""
    reasoning: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None
    error: str | None = None
    elapsed: float = 0.0
    incomplete: bool = False
    tool_deltas: tuple = ()
    tool_trace: tuple = ()
    request_count: int = 0
    provider: str = ""
    model: str = ""
    base_url: str = ""
    alias: str = ""

    def public_dict(self) -> dict:
        usage = None if self.usage is None else self.usage.to_dict()
        return {
            "status": self.status,
            "text": self.text,
            "reasoning": self.reasoning,
            "usage": usage,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "elapsed": round(self.elapsed, 3),
            "incomplete": self.incomplete,
            "request_count": self.request_count,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "alias": self.alias,
        }


class Presentation:
    """Main-thread output. Package B can subclass ``block`` for Markdown."""

    def __init__(self, style, output=None) -> None:
        import sys

        self.style = style
        self.output = output or sys.stdout

    def line(self, text: str) -> None:
        print(text, file=self.output)

    def status_line(self, text: str) -> None:
        self.line(text)

    def warn(self, text: str) -> None:
        paint = getattr(self.style, "yellow", lambda value: value)
        self.line(paint(text))

    def block(self, title: str, body: str, *, meta: str = "") -> None:
        bold = getattr(self.style, "bold", lambda value: value)
        dim = getattr(self.style, "dim", lambda value: value)
        self.line(bold(title))
        if meta:
            self.line(dim(meta))
        if body:
            self.line(body)
        self.line("")


class AttachmentStore:
    """Process-local approved attachments. Not a filesystem walker."""

    def __init__(self) -> None:
        self._items: dict[str, ApprovedAttachment] = {}

    def approve(self, attachment: ApprovedAttachment) -> None:
        if not attachment.path or attachment.path.strip() != attachment.path:
            raise ValueError("Approved attachment paths must be exact and non-empty.")
        if ".." in attachment.path.replace("\\", "/").split("/"):
            raise ValueError("Approved attachment paths cannot contain '..'.")
        self._items[attachment.path] = attachment

    def revoke(self, path: str) -> None:
        self._items.pop(path, None)

    def clear(self) -> None:
        self._items.clear()

    def list(self) -> tuple[ApprovedAttachment, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


_ATTACHMENTS = AttachmentStore()


def attachment_store() -> AttachmentStore:
    """The process-wide store Package B should fill after consent."""

    return _ATTACHMENTS


def make_snapshot(
    history,
    *,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    attachments: tuple[ApprovedAttachment, ...] | None = None,
    record_refs: bool = False,
    snapshot_id: str | None = None,
) -> RequestSnapshot:
    """Copy history into an immutable snapshot. Later edits to the source are ignored."""

    frozen: list[tuple[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            role, content = item[0], item[1]
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        frozen.append((role, content))
    return RequestSnapshot(
        version=SNAPSHOT_VERSION,
        snapshot_id=snapshot_id or uuid.uuid4().hex,
        history=tuple(frozen),
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        attachments=tuple(attachments or ()),
        record_refs=record_refs,
    )


def snapshot_from_session(session, *, attachments=None, record_refs: bool = False) -> RequestSnapshot:
    """Copy a Session without reading ``api_key`` or ``key_source``."""

    return make_snapshot(
        session.messages,
        system=session.system,
        temperature=session.temperature,
        max_tokens=session.max_tokens,
        attachments=tuple(attachments) if attachments is not None else attachment_store().list(),
        record_refs=record_refs,
    )


def messages_for(snapshot: RequestSnapshot, prompt: str, extra: list | None = None) -> list[dict]:
    """Private message list for one request. Mutating it does not change the snapshot."""

    copied = [{"role": role, "content": content} for role, content in snapshot.history]
    if prompt:
        copied.append({"role": "user", "content": prompt})
    if extra:
        copied.extend(deepcopy(extra))
    return copied


def _emit(event_queue: queue.Queue | None, event: RunEvent) -> None:
    if event_queue is not None:
        event_queue.put(event)


def _blank(status: str, target, message: str, *, incomplete: bool = True) -> RunResult:
    return RunResult(
        status=status,
        error=message,
        incomplete=incomplete,
        provider=getattr(target, "provider", ""),
        model=getattr(target, "model", ""),
        base_url=getattr(target, "base_url", ""),
        alias=getattr(target, "alias", "") or "",
    )


def run_request(
    provider,
    target,
    snapshot: RequestSnapshot,
    options: RunOptions,
    *,
    credential: str | None,
    prompt: str = "",
    messages: list | None = None,
    cancel_event: threading.Event | None = None,
    event_queue: queue.Queue | None = None,
) -> RunResult:
    """Run one completion.

    ``messages``, when passed, is copied and sent as-is (tool rounds). Otherwise
    the snapshot history plus ``prompt`` is used. Credentials stay in the
    ``credential`` argument and are not written onto the result.
    """

    label = target.alias or f"{target.provider}:{target.model}"
    if cancel_event is not None and cancel_event.is_set():
        _emit(event_queue, RunEvent("status", label, "cancelled"))
        return _blank("cancelled", target, "Cancelled before the request was sent.")

    now = time.perf_counter()
    timeout = max(0.1, float(options.timeout))
    if options.deadline is not None:
        remaining = options.deadline - now
        if remaining <= 0:
            _emit(event_queue, RunEvent("status", label, "timeout"))
            return _blank("timeout", target, "The team deadline elapsed before this request started.")
        timeout = max(0.1, min(timeout, remaining))

    outbound = deepcopy(messages) if messages is not None else messages_for(snapshot, prompt)
    temperature = options.temperature if options.temperature is not None else snapshot.temperature
    max_tokens = options.max_tokens if options.max_tokens is not None else snapshot.max_tokens
    _emit(event_queue, RunEvent("status", label, "running"))
    started = time.perf_counter()
    text_parts: list[str] = []
    thought_parts: list[str] = []
    deltas: list = []
    usage: Usage | None = None
    finish_reason: str | None = None
    try:
        for chunk in stream_chat(
            provider,
            model=target.model,
            messages=outbound,
            system=snapshot.system,
            key=credential,
            base_url=target.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            tools=options.tools_payload,
        ):
            if cancel_event is not None and cancel_event.is_set():
                elapsed = time.perf_counter() - started
                partial = "".join(text_parts)
                _emit(event_queue, RunEvent("status", label, "cancelled"))
                return RunResult(
                    status="cancelled",
                    text=partial,
                    reasoning="".join(thought_parts),
                    usage=usage,
                    finish_reason=finish_reason,
                    error=CANCELLATION_NOTE,
                    elapsed=elapsed,
                    incomplete=True,
                    tool_deltas=tuple(deltas),
                    request_count=1,
                    provider=target.provider,
                    model=target.model,
                    base_url=target.base_url,
                    alias=target.alias or "",
                )
            if chunk.usage is not None:
                if usage is None:
                    usage = Usage()
                usage.update(chunk.usage)
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.reasoning:
                thought_parts.append(chunk.reasoning)
            if getattr(chunk, "tool_call", None) is not None:
                deltas.append(chunk.tool_call)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
    except ProviderError as exc:
        return _finished_error("error", target, exc, started, text_parts, thought_parts, usage, finish_reason, deltas)
    except NetworkError as exc:
        status = "timeout" if "timed out" in str(exc).lower() else "error"
        return _finished_error(status, target, exc, started, text_parts, thought_parts, usage, finish_reason, deltas)
    except AnyLLMError as exc:
        return _finished_error("error", target, exc, started, text_parts, thought_parts, usage, finish_reason, deltas)
    except Exception as exc:
        # Isolate a member failure. Do not include the credential in the message.
        cleaned = str(exc)
        if credential and credential in cleaned:
            cleaned = cleaned.replace(credential, "[redacted]")
        return _finished_error(
            "error",
            target,
            cleaned,
            started,
            text_parts,
            thought_parts,
            usage,
            finish_reason,
            deltas,
        )

    text = "".join(text_parts)
    incomplete = finish_reason in {"length", "max_tokens", "max_completion_tokens"}
    status = "incomplete" if incomplete and not deltas else "ok"
    if not text.strip() and not deltas:
        status = "error"
        error = "The provider returned an empty response."
        incomplete = True
    else:
        error = None
    elapsed = time.perf_counter() - started
    _emit(event_queue, RunEvent("status", label, status))
    return RunResult(
        status=status,
        text=text,
        reasoning="".join(thought_parts),
        usage=usage,
        finish_reason=finish_reason,
        error=error,
        elapsed=elapsed,
        incomplete=incomplete or status != "ok",
        tool_deltas=tuple(deltas),
        request_count=1,
        provider=target.provider,
        model=target.model,
        base_url=target.base_url,
        alias=target.alias or "",
    )


def _finished_error(status, target, exc, started, text_parts, thought_parts, usage, finish_reason, deltas):
    message = exc if isinstance(exc, str) else getattr(exc, "message", None) or str(exc)
    elapsed = time.perf_counter() - started
    return RunResult(
        status=status,
        text="".join(text_parts),
        reasoning="".join(thought_parts),
        usage=usage,
        finish_reason=finish_reason,
        error=message,
        elapsed=elapsed,
        incomplete=True,
        tool_deltas=tuple(deltas),
        request_count=1,
        provider=target.provider,
        model=target.model,
        base_url=target.base_url,
        alias=target.alias or "",
    )


def load_feature(path, *, what: str) -> dict:
    """Load a versioned feature file. Missing files are empty version 1 objects."""

    data = read_json(path, what=what)
    if not data:
        return {"version": FEATURE_STATE_VERSION}
    version = data.get("version", 1)
    if version != FEATURE_STATE_VERSION:
        from .errors import ConfigError

        raise ConfigError(
            f"The {what} file uses version {version}, which this build cannot read.",
            hint=f"Expected version {FEATURE_STATE_VERSION}. Fix or remove {path}.",
        )
    if not isinstance(data, dict):
        from .errors import ConfigError

        raise ConfigError(f"The {what} file is malformed.", hint=f"Fix or remove {path}.")
    return data


def save_feature(path, data: dict) -> None:
    payload = dict(data)
    payload["version"] = FEATURE_STATE_VERSION
    write_json(path, payload, private=True)


def feature_path(name: str):
    return config_dir() / name


HELP_ROWS: list[tuple[str, str]] = []
REPL_HANDLERS: dict[str, Callable] = {}


def register_repl(name: str, handler: Callable, help_row: tuple[str, str] | None = None) -> None:
    """Register a REPL command. ``handler`` is ``(repl, rest) -> None``."""

    REPL_HANDLERS[name] = handler
    if help_row is not None and help_row not in HELP_ROWS:
        HELP_ROWS.append(help_row)


_INSTALLED = False


def install_repl_features() -> None:
    """Wire Package A commands. Safe to call more than once."""

    global _INSTALLED
    if _INSTALLED:
        return
    # Imported lazily so seam can be used without loading every command.
    from . import budgets, targets, team, tool_protocol

    targets.register(register_repl)
    budgets.register(register_repl)
    team.register(register_repl)
    tool_protocol.register(register_repl)
    _INSTALLED = True
