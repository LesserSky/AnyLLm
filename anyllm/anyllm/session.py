"""Conversation state and on-disk sessions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .adapters import Usage, stream_chat
from .catalog import Provider, validate_base_url
from .config import read_json, sessions_dir, write_json
from .errors import ConfigError, ProviderError, UsageError

# Stored on a partial assistant turn when the user hits Ctrl-C mid-stream.
INTERRUPTED_MARK = "[interrupted]"


@dataclass
class Turn:
    text: str
    reasoning: str
    usage: Usage
    elapsed: float
    finish_reason: str | None = None


def _interrupted_content(text: str) -> str:
    """Append a stable marker so history shows the answer was cut off."""
    body = text.rstrip()
    if body.endswith(INTERRUPTED_MARK):
        return body
    return f"{body}\n{INTERRUPTED_MARK}"


def session_base_url(data: dict, provider: Provider, override: str | None = None) -> str:
    """Choose the endpoint when a saved conversation is restored.

    An explicit override wins. Otherwise a base URL stored in the session is
    used, including when it differs from the provider default. Files written
    before base URLs were saved omit the field and fall back to that default.
    """
    if isinstance(override, str) and override.strip():
        return validate_base_url(override)
    saved = data.get("base_url") if isinstance(data, dict) else None
    if saved is None:
        return provider.base_url or ""
    if not isinstance(saved, str):
        raise ConfigError(
            "Saved session has an invalid base URL.",
            hint="Expected a string. Fix the session file, or delete it.",
        )
    if not saved.strip():
        return provider.base_url or ""
    try:
        return validate_base_url(saved)
    except ConfigError as exc:
        raise ConfigError(
            f"Saved session has an invalid base URL {saved!r}.",
            hint=exc.hint,
        ) from None


def finish_notice(finish_reason: str | None) -> str | None:
    """Explain a finish reason that means the answer is not whole."""
    if not finish_reason:
        return None
    reason = finish_reason.lower()
    if reason in ("length", "max_tokens", "max_completion_tokens"):
        return "the response hit the token limit and was cut off (try a larger --max-tokens)"
    if reason in (
        "content_filter",
        "safety",
        "blocklist",
        "prohibited_content",
        "recitation",
    ):
        return "the response was stopped by the provider's content filter"
    return None


class Session:
    """One conversation: provider, model, history, and token totals."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        key_source: str = "",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        show_reasoning: bool = False,
        messages: list[dict] | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url or provider.base_url
        self.api_key = api_key
        self.key_source = key_source
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.show_reasoning = show_reasoning
        self.messages: list[dict] = list(messages or [])
        self.usage = usage or Usage()
        self.turns = sum(1 for item in self.messages if item.get("role") == "assistant")

    def send(self, text: str, on_text=None, on_reasoning=None) -> Turn:
        """Send a user message and stream the answer back."""
        if not isinstance(text, str) or not text.strip():
            raise UsageError("The message is empty.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise UsageError(
                "No model selected.",
                hint="Set one with /model NAME.",
            )

        self.messages.append({"role": "user", "content": text})
        start = time.perf_counter()
        text_parts: list[str] = []
        thought_parts: list[str] = []
        usage = Usage()
        finish_reason: str | None = None

        try:
            for chunk in stream_chat(
                self.provider,
                model=self.model,
                messages=self.messages,
                system=self.system,
                key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            ):
                if chunk.usage is not None:
                    usage.update(chunk.usage)
                if chunk.text:
                    text_parts.append(chunk.text)
                    if on_text is not None:
                        on_text(chunk.text)
                if chunk.reasoning:
                    thought_parts.append(chunk.reasoning)
                    if on_reasoning is not None:
                        on_reasoning(chunk.reasoning)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
        except KeyboardInterrupt:
            partial = "".join(text_parts)
            if partial.strip():
                # Keep the fragment so the next turn has context, and mark it
                # so the transcript does not look like a finished answer.
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": _interrupted_content(partial),
                        "interrupted": True,
                    }
                )
            else:
                self.messages.pop()
            raise
        except BaseException:
            self.messages.pop()
            raise

        answer = "".join(text_parts)
        reasoning = "".join(thought_parts)
        if not answer.strip():
            self.messages.pop()
            notice = finish_notice(finish_reason)
            if notice:
                raise ProviderError(notice.capitalize(), hint="The finish reason is the evidence for that limit.")
            if reasoning.strip():
                raise ProviderError(
                    "The model returned reasoning but no answer text.",
                    hint="That does not by itself show that the output cap was hit.",
                )
            raise ProviderError(
                "The provider returned an empty response.",
                hint="Try another model with /model, or rephrase the prompt.",
            )

        self.messages.append({"role": "assistant", "content": answer})
        self.usage = self.usage.combined(usage)
        self.turns += 1
        elapsed = time.perf_counter() - start
        return Turn(
            text=answer,
            reasoning="".join(thought_parts),
            usage=usage,
            elapsed=elapsed,
            finish_reason=finish_reason,
        )

    def append_exchange(self, user_text: str, assistant_text: str, *, usage: Usage | None = None) -> None:
        """Append one finished exchange without calling a provider.

        Package A uses this for ``/team adopt``. Package B can use it when the
        user explicitly keeps a reviewed result. It does not read credentials.
        """

        if not isinstance(user_text, str) or not user_text.strip():
            raise UsageError("The message is empty.")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            raise UsageError("The answer is empty.")
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": assistant_text})
        if usage is not None:
            self.usage = self.usage.combined(usage)
        self.turns += 1

    def to_dict(self) -> dict:
        payload = {
            "version": 1,
            "provider": self.provider.name,
            "model": self.model,
            "system": self.system,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "show_reasoning": self.show_reasoning,
            "messages": [
                item
                for item in self.messages
                if isinstance(item, dict)
                and item.get("role") in ("user", "assistant")
                and isinstance(item.get("content"), str)
            ],
            "usage": self.usage.to_dict(),
            "updated": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        # Always record a resolved endpoint. Old files simply omit this key.
        if isinstance(self.base_url, str) and self.base_url.strip():
            payload["base_url"] = self.base_url
        # Identity only. The key stays in the key store or the environment.
        alias = getattr(self, "target_alias", None)
        if isinstance(alias, str) and alias.strip():
            payload["target"] = alias.strip()
        return payload


# -- saved sessions ---------------------------------------------------------

_BAD_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def sanitize_session_name(name: str) -> str:
    cleaned = _BAD_NAME.sub("-", (name or "").strip())
    cleaned = cleaned.strip(" .-")
    if not cleaned:
        raise ConfigError(
            "Invalid session name.",
            hint="Use letters, digits, spaces, dots, dashes, or underscores.",
        )
    return cleaned[:64]


def session_path(name: str) -> Path:
    return sessions_dir() / f"{sanitize_session_name(name)}.json"


def default_session_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_session(session: Session, name: str | None = None) -> Path:
    target = session_path(name or default_session_name())
    try:
        write_json(target, session.to_dict(), private=True)
    except OSError as exc:
        raise ConfigError(
            f"Could not save the session: {exc}",
            hint=f"Check that you can write to {target.parent}",
        ) from None
    return target


def load_session_data(name: str) -> dict:
    path = session_path(name)
    if not path.exists():
        raise ConfigError(
            f"Session {name!r} was not found.",
            hint="Run `/sessions` in chat to list saved sessions.",
        )
    data = read_json(path, what="session")
    messages = data.get("messages")
    if messages is not None and not isinstance(messages, list):
        raise ConfigError(
            f"Session {name!r} has an invalid message list.",
            hint=f"Fix or delete {path}",
        )
    for item in messages or []:
        if (
            not isinstance(item, dict)
            or item.get("role") not in ("user", "assistant")
            or not isinstance(item.get("content"), str)
        ):
            raise ConfigError(
                f"Session {name!r} contains an invalid message.",
                hint=f"Fix or delete {path}",
            )
    if not data.get("provider"):
        raise ConfigError(
            f"Session {name!r} does not name a provider.",
            hint=f"Fix or delete {path}",
        )
    return data


def list_sessions() -> list[dict]:
    directory = sessions_dir()
    entries: list[dict] = []
    for path in directory.glob("*.json"):
        try:
            data = read_json(path, what="session")
        except ConfigError:
            continue
        message_count = len(data.get("messages") or [])
        updated = data.get("updated")
        if not isinstance(updated, str):
            try:
                stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                updated = stamp.replace(microsecond=0).isoformat()
            except OSError:
                updated = ""
        entries.append(
            {
                "name": path.stem,
                "provider": data.get("provider") or "?",
                "model": data.get("model") or "?",
                "messages": message_count,
                "updated": updated,
                "path": path,
            }
        )
    entries.sort(key=lambda entry: entry["updated"] or "", reverse=True)
    return entries


def latest_session_name() -> str | None:
    entries = list_sessions()
    return entries[0]["name"] if entries else None
