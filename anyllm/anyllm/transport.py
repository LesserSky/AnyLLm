"""HTTP, retries, error mapping, and server-sent-event parsing."""

from __future__ import annotations

import http.client
import json
import platform
import re
import socket
import ssl
import time
from urllib import error, request
from urllib.parse import urlsplit, urlunsplit

from . import __version__
from .errors import APIError, AuthError, NetworkError, NotFoundError, RateLimitError

USER_AGENT = (
    f"anyllm/{__version__} "
    f"({platform.system()} {platform.release()}; Python {platform.python_version()})"
)

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_RETRIES = 2
MAX_RETRY_WAIT = 8.0

_EVENT_BOUNDARY = re.compile(rb"\r?\n\r?\n")


def join_url(base: str, path: str) -> str:
    """Join a base URL and path, preserving any query string on the base."""
    parts = urlsplit(base)
    new_path = f"{parts.path.rstrip('/')}/{path.lstrip('/')}"
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))


def _safe_read(response) -> bytes:
    try:
        return response.read()
    except Exception:
        return b""


def _extract_message(body: bytes) -> str:
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:400]
    if isinstance(parsed, dict):
        detail = parsed.get("error")
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("code") or detail.get("type")
            if message:
                return str(message)[:400]
        if isinstance(detail, str) and detail:
            return detail[:400]
        for key in ("message", "detail", "error_description", "msg"):
            value = parsed.get(key)
            if isinstance(value, str) and value:
                return value[:400]
    return text[:400]


def raise_for_status(status: int, body: bytes, headers, url: str) -> None:
    """Turn an HTTP error into a specific, actionable exception."""
    host = urlsplit(url).netloc or url
    message = _extract_message(body)
    detail = f"{message} (HTTP {status})" if message else f"HTTP {status} from {host}"
    lowered = message.lower()

    if status in (401, 403):
        if status == 401:
            hint = (
                "The API key was rejected. Run `anyllm keys set <provider>` again, "
                "or check the environment variable in `anyllm doctor`."
            )
        else:
            hint = (
                "The key was accepted but access was denied. Check that the key "
                "may use this model, region, or account."
            )
        raise AuthError(f"authentication failed: {detail}", hint=hint)

    if status == 404:
        raise NotFoundError(
            f"endpoint or model not found: {detail}",
            hint="Check the model name and --base-url. "
            "Run `anyllm models <provider>` to see what the key can access.",
        )

    if status == 429:
        retry_after = headers.get("Retry-After") if headers else None
        hint = "You hit a rate limit or quota. Wait a moment and retry."
        if retry_after:
            hint += f" The provider asked to wait {retry_after} seconds."
        raise RateLimitError(f"rate limited: {detail}", hint=hint)

    hint = None
    if status == 400:
        if any(
            word in lowered
            for word in ("context length", "maximum context", "too many tokens", "too long")
        ):
            hint = "The conversation may be too long. Use /clear to start fresh, or shorten the prompt."
        elif "max_tokens" in lowered or "max_completion_tokens" in lowered:
            hint = "The token limit was rejected. Try without --max-tokens, or use a smaller value."
        elif "temperature" in lowered:
            hint = "This model may not support temperature. Try without --temperature."
        elif "model" in lowered and ("not exist" in lowered or "not found" in lowered):
            hint = "Run `anyllm models <provider>` to list available model names."
    elif status >= 500:
        hint = "The provider had a server error. Retry shortly."

    raise APIError(f"request failed: {detail}", hint=hint, status=status)


def _retry_delay(headers, attempt: int) -> float | None:
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after:
        try:
            seconds = float(retry_after)
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None:
            if seconds > MAX_RETRY_WAIT:
                return None
            return max(0.0, seconds)
    return 0.8 * (2**attempt)


def _network_error(exc: BaseException, url: str) -> NetworkError:
    host = urlsplit(url).netloc or url
    reason = getattr(exc, "reason", None) or exc
    text = str(reason) or exc.__class__.__name__
    hint = "Check your internet connection, proxy settings, and the base URL."
    if isinstance(reason, ssl.SSLError) or isinstance(exc, ssl.SSLError):
        hint = "TLS failed. Check your system certificates, proxy, or VPN."
    return NetworkError(f"could not reach {host}: {text}", hint=hint)


def _open(
    method: str,
    url: str,
    data: bytes | None,
    headers: dict[str, str],
    timeout: float,
    retries: int,
):
    """Open a URL, retrying transient failures before any data is consumed."""
    request_headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    }
    request_headers.update(headers)

    for attempt in range(retries + 1):
        req = request.Request(url, data=data, headers=request_headers, method=method)
        try:
            return request.urlopen(req, timeout=timeout)
        except error.HTTPError as exc:
            body = _safe_read(exc)
            try:
                exc.close()
            except OSError:
                pass
            if attempt < retries and exc.code in RETRY_STATUS:
                delay = _retry_delay(exc.headers, attempt)
                if delay is not None:
                    time.sleep(delay)
                    continue
            raise_for_status(exc.code, body, exc.headers, url)
        except (error.URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
            if attempt < retries and not isinstance(exc, ssl.SSLError):
                time.sleep(0.6 * (attempt + 1))
                continue
            raise _network_error(exc, url) from None

    # Unreachable in practice; keeps type checkers happy.
    raise NetworkError(f"could not reach {url}", hint="Check your connection and base URL.")


def open_stream(
    url: str,
    headers: dict[str, str],
    payload: dict,
    *,
    timeout: float = 120.0,
    retries: int = MAX_RETRIES,
):
    """POST JSON and return an open streaming response."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return _open("POST", url, data, headers, timeout, retries)


def request_json(
    url: str,
    headers: dict[str, str],
    payload: dict | None = None,
    *,
    method: str = "GET",
    timeout: float = 60.0,
    retries: int = MAX_RETRIES,
):
    """Perform a request and decode a JSON object or array."""
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    response = _open(method, url, data, headers, timeout, retries)
    try:
        raw = response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise NetworkError(
            f"connection broke while reading from {urlsplit(url).netloc}: {exc}",
            hint="Retry the request.",
        ) from None
    finally:
        try:
            response.close()
        except OSError:
            pass
    if not raw.strip():
        return {}
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        snippet = raw.decode("utf-8", "replace").strip()[:300]
        raise APIError(
            f"{urlsplit(url).netloc} returned a non-JSON response.",
            hint=f"Response started with: {snippet!r}",
        ) from None


def _event_payload(block: bytes) -> str | None:
    lines = block.decode("utf-8", "replace").splitlines()
    parts: list[str] = []
    for line in lines:
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            parts.append(value)
        elif line.startswith(":"):
            continue
    if not parts:
        return None
    return "\n".join(parts)


def _iter_sse(response):
    buffer = b""
    while True:
        chunk = response.read(8192)
        if not chunk:
            break
        buffer += chunk
        while True:
            match = _EVENT_BOUNDARY.search(buffer)
            if match is None:
                break
            block = buffer[: match.start()]
            buffer = buffer[match.end() :]
            payload = _event_payload(block)
            if payload is None:
                continue
            if payload.strip() == "[DONE]":
                return
            yield payload
    payload = _event_payload(buffer)
    if payload is not None and payload.strip() != "[DONE]":
        yield payload


def iter_sse(response):
    """Yield decoded SSE data payloads, translating read errors."""
    try:
        yield from _iter_sse(response)
    except (OSError, TimeoutError, ssl.SSLError, http.client.HTTPException) as exc:
        raise NetworkError(
            f"connection broke while streaming: {exc}",
            hint="The provider may have timed out. Retry, or use --timeout with a larger value.",
        ) from None
