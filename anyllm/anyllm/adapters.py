"""Provider adapters.

Every adapter turns one provider's streaming protocol into the same small
``Chunk`` stream. The rest of the app never sees provider-specific JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import quote

from .catalog import ANTHROPIC, GEMINI, OPENAI, Provider
from .errors import APIError, AuthError, ProviderError
from .transport import iter_sse, join_url, open_stream, request_json


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None

    def update(self, other: "Usage") -> None:
        """Keep the latest (usually cumulative) values from a stream."""
        for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
            value = getattr(other, field)
            if value is None:
                continue
            current = getattr(self, field)
            setattr(self, field, value if current is None else max(current, value))

    def combined(self, other: "Usage") -> "Usage":
        """Add two usage reports, for session totals."""
        result = Usage()
        for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
            left = getattr(self, field)
            right = getattr(other, field)
            if left is None:
                setattr(result, field, right)
            elif right is None:
                setattr(result, field, left)
            else:
                setattr(result, field, left + right)
        return result

    @property
    def total(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    def to_dict(self) -> dict:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "reasoning": self.reasoning_tokens,
            "cached": self.cached_tokens,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, raw) -> "Usage":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            input_tokens=_int(raw.get("input")),
            output_tokens=_int(raw.get("output")),
            reasoning_tokens=_int(raw.get("reasoning")),
            cached_tokens=_int(raw.get("cached")),
        )


@dataclass
class Chunk:
    text: str = ""
    reasoning: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None


def _int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _normalize_messages(messages) -> list[dict]:
    result: list[dict] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if not content.strip():
            continue
        result.append({"role": role, "content": content})
    return result


def _anthropic_messages(messages) -> list[dict]:
    """Anthropic wants user/assistant turns that alternate."""
    normalized = _normalize_messages(messages)
    result: list[dict] = []
    for message in normalized:
        if result and result[-1]["role"] == message["role"]:
            result[-1]["content"] += "\n\n" + message["content"]
        else:
            result.append(dict(message))
    if result and result[0]["role"] == "assistant":
        result.insert(0, {"role": "user", "content": "(continuing the conversation)"})
    return result


def _gemini_contents(messages) -> list[dict]:
    contents: list[dict] = []
    for message in _normalize_messages(messages):
        role = "user" if message["role"] == "user" else "model"
        if not contents and role == "model":
            contents.append({"role": "user", "parts": [{"text": "(continuing the conversation)"}]})
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    return contents


def _openai_usage(raw) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    usage = Usage(
        input_tokens=_int(raw.get("prompt_tokens", raw.get("input_tokens"))),
        output_tokens=_int(raw.get("completion_tokens", raw.get("output_tokens"))),
    )
    details = raw.get("completion_tokens_details")
    if isinstance(details, dict):
        usage.reasoning_tokens = _int(details.get("reasoning_tokens"))
    prompt_details = raw.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        usage.cached_tokens = _int(prompt_details.get("cached_tokens"))
    if usage.input_tokens is None and usage.output_tokens is None:
        return None
    return usage


def _anthropic_usage(raw) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    usage = Usage(
        input_tokens=_int(raw.get("input_tokens")),
        output_tokens=_int(raw.get("output_tokens")),
        cached_tokens=_int(raw.get("cache_read_input_tokens")),
    )
    if (
        usage.input_tokens is None
        and usage.output_tokens is None
        and usage.cached_tokens is None
    ):
        return None
    return usage


def _gemini_usage(raw) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    usage = Usage(
        input_tokens=_int(raw.get("promptTokenCount")),
        output_tokens=_int(raw.get("candidatesTokenCount")),
        reasoning_tokens=_int(raw.get("thoughtsTokenCount")),
        cached_tokens=_int(raw.get("cachedContentTokenCount")),
    )
    if (
        usage.input_tokens is None
        and usage.output_tokens is None
        and usage.reasoning_tokens is None
    ):
        return None
    return usage


def _error_text(detail) -> str:
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("type") or detail)
    return str(detail)


def _token_field(model: str) -> str:
    if model.lower().startswith(("o1", "o3", "o4", "gpt-5")):
        return "max_completion_tokens"
    return "max_tokens"


def _content_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _check_base_url(base_url: str, provider: Provider) -> str:
    if not base_url:
        raise APIError(
            f"Provider {provider.name!r} has no base URL.",
            hint="Pass --base-url, or add the provider with `anyllm providers add`.",
        )
    return base_url


def stream_chat(
    provider: Provider,
    *,
    model: str,
    messages,
    system: str | None = None,
    key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 120.0,
    include_usage: bool = True,
):
    """Yield response chunks for one completion request."""
    base = _check_base_url(base_url or provider.base_url, provider)
    if provider.kind == OPENAI:
        yield from _openai_stream(
            provider,
            model=model,
            messages=messages,
            system=system,
            key=key,
            base_url=base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            include_usage=include_usage,
        )
    elif provider.kind == ANTHROPIC:
        yield from _anthropic_stream(
            provider,
            model=model,
            messages=messages,
            system=system,
            key=key,
            base_url=base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider.kind == GEMINI:
        yield from _gemini_stream(
            provider,
            model=model,
            messages=messages,
            system=system,
            key=key,
            base_url=base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        raise APIError(
            f"Provider {provider.name!r} uses unsupported kind {provider.kind!r}.",
            hint="Supported kinds: openai, anthropic, gemini.",
        )


def _openai_stream(
    provider: Provider,
    *,
    model: str,
    messages,
    system: str | None,
    key: str | None,
    base_url: str,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
    include_usage: bool,
):
    url = join_url(base_url, "chat/completions")
    conversation: list[dict] = []
    if system and system.strip():
        conversation.append({"role": "system", "content": system})
    conversation.extend(_normalize_messages(messages))

    payload: dict = {"model": model, "messages": conversation, "stream": True}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens:
        payload[_token_field(model)] = max_tokens
    if include_usage:
        payload["stream_options"] = {"include_usage": True}

    headers = provider.auth_headers(key)
    headers["Accept"] = "text/event-stream"

    try:
        response = open_stream(url, headers, payload, timeout=timeout)
    except APIError as exc:
        # A few OpenAI-compatible servers reject stream_options. Retry once
        # without it rather than failing a request that would otherwise work.
        if include_usage and "stream_option" in str(exc).lower():
            payload.pop("stream_options", None)
            response = open_stream(url, headers, payload, timeout=timeout)
        else:
            raise

    try:
        for data in iter_sse(response):
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(event, list):
                event = event[-1] if event else {}
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                raise ProviderError(
                    f"{provider.label} error: {_error_text(event['error'])}"
                )

            usage = _openai_usage(event.get("usage"))
            if usage is not None:
                yield Chunk(usage=usage)

            for choice in event.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    text = _content_text(delta.get("content"))
                    if text:
                        yield Chunk(text=text)
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        yield Chunk(reasoning=reasoning)
                finish = choice.get("finish_reason")
                if finish:
                    yield Chunk(finish_reason=str(finish))
    finally:
        try:
            response.close()
        except OSError:
            pass


def _anthropic_stream(
    provider: Provider,
    *,
    model: str,
    messages,
    system: str | None,
    key: str | None,
    base_url: str,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
):
    if not key:
        raise AuthError(
            "Anthropic requires an API key.",
            hint="Run `anyllm keys set anthropic`, or export ANTHROPIC_API_KEY.",
        )
    url = join_url(base_url, "messages")
    payload: dict = {
        "model": model,
        "max_tokens": max_tokens or 4096,
        "messages": _anthropic_messages(messages),
        "stream": True,
    }
    if system and system.strip():
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = temperature

    headers = provider.auth_headers(key)
    headers["anthropic-version"] = "2023-06-01"
    headers["Accept"] = "text/event-stream"

    response = open_stream(url, headers, payload, timeout=timeout)
    try:
        for data in iter_sse(response):
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "error":
                raise ProviderError(
                    f"{provider.label} error: {_error_text(event.get('error'))}"
                )
            if event_type == "message_start":
                usage = _anthropic_usage((event.get("message") or {}).get("usage"))
                if usage is not None:
                    yield Chunk(usage=usage)
            elif event_type == "content_block_delta":
                delta = event.get("delta") or {}
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str) and text:
                        yield Chunk(text=text)
                elif delta_type == "thinking_delta":
                    thought = delta.get("thinking")
                    if isinstance(thought, str) and thought:
                        yield Chunk(reasoning=thought)
            elif event_type == "message_delta":
                usage = _anthropic_usage(event.get("usage"))
                finish = (event.get("delta") or {}).get("stop_reason")
                if usage is not None or finish:
                    yield Chunk(usage=usage, finish_reason=finish)
            elif event_type == "message_stop":
                return
    finally:
        try:
            response.close()
        except OSError:
            pass


_GEMINI_FINISH = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "OTHER": "stop",
}


def _gemini_stream(
    provider: Provider,
    *,
    model: str,
    messages,
    system: str | None,
    key: str | None,
    base_url: str,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
):
    if not key:
        raise AuthError(
            "Google Gemini requires an API key.",
            hint="Run `anyllm keys set google`, or export GEMINI_API_KEY.",
        )
    if model.startswith("models/"):
        model = model[len("models/") :]
    url = join_url(
        base_url,
        f"models/{quote(model, safe='')}:streamGenerateContent?alt=sse",
    )
    payload: dict = {"contents": _gemini_contents(messages)}
    if system and system.strip():
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    generation: dict = {}
    if temperature is not None:
        generation["temperature"] = temperature
    if max_tokens:
        generation["maxOutputTokens"] = max_tokens
    if generation:
        payload["generationConfig"] = generation

    headers = provider.auth_headers(key)
    headers["Accept"] = "text/event-stream"

    response = open_stream(url, headers, payload, timeout=timeout)
    try:
        for data in iter_sse(response):
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                raise ProviderError(
                    f"{provider.label} error: {_error_text(event['error'])}"
                )
            feedback = event.get("promptFeedback") or {}
            blocked = feedback.get("blockReason")
            if blocked:
                raise ProviderError(
                    f"{provider.label} blocked the prompt ({blocked}).",
                    hint="Rephrase the prompt without restricted content.",
                )

            usage = _gemini_usage(event.get("usageMetadata"))
            if usage is not None:
                yield Chunk(usage=usage)

            for candidate in event.get("candidates") or []:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if not isinstance(text, str) or not text:
                        continue
                    if part.get("thought"):
                        yield Chunk(reasoning=text)
                    else:
                        yield Chunk(text=text)
                finish = candidate.get("finishReason")
                if finish:
                    yield Chunk(finish_reason=_GEMINI_FINISH.get(str(finish), str(finish).lower()))
    finally:
        try:
            response.close()
        except OSError:
            pass


def list_models(
    provider: Provider,
    *,
    key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
) -> list[str]:
    """Ask the provider which models this key can use."""
    base = _check_base_url(base_url or provider.base_url, provider)
    headers = provider.auth_headers(key)
    headers["Accept"] = "application/json"

    if provider.kind == OPENAI:
        data = request_json(join_url(base, "models"), headers, timeout=timeout)
        items = data.get("data") if isinstance(data, dict) else data
    elif provider.kind == ANTHROPIC:
        if not key:
            raise AuthError(
                "Anthropic requires an API key to list models.",
                hint="Run `anyllm keys set anthropic`.",
            )
        headers["anthropic-version"] = "2023-06-01"
        data = request_json(join_url(base, "models?limit=1000"), headers, timeout=timeout)
        items = data.get("data") if isinstance(data, dict) else data
    elif provider.kind == GEMINI:
        if not key:
            raise AuthError(
                "Google Gemini requires an API key to list models.",
                hint="Run `anyllm keys set google`.",
            )
        data = request_json(join_url(base, "models?pageSize=1000"), headers, timeout=timeout)
        items = data.get("models") if isinstance(data, dict) else data
        filtered = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods")
            if methods and "generateContent" not in methods:
                continue
            filtered.append(item)
        items = filtered
    else:
        raise APIError(f"Unsupported provider kind {provider.kind!r}.")

    names: list[str] = []
    for item in items or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            value = item.get("id") or item.get("name")
            if isinstance(value, str) and value:
                if value.startswith("models/"):
                    value = value[len("models/") :]
                names.append(value)
    unique = sorted({name for name in names if name}, key=str.lower)
    return unique
