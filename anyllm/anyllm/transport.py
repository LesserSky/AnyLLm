"""HTTP, retries, error mapping, and server-sent-event parsing."""

from __future__ import annotations

import errno
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

# The server asked the client to try again and did not apply the request.
# Safe for a completion POST, which is not idempotent.
_SAFE_RETRY_STATUS = frozenset({408, 425, 429, 503})

# These mean the server, or a gateway in front of it, already had the request.
# Retry only when repeating it cannot duplicate a generation (GET, HEAD).
_IDEMPOTENT_RETRY_STATUS = frozenset({500, 502, 504})

MAX_RETRIES = 2
MAX_RETRY_WAIT = 8.0

# Connect never completed, so no HTTP request bytes were sent.
_BEFORE_SEND_ERRNOS = {
    number
    for number in (
        errno.ECONNREFUSED,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        getattr(errno, "WSAECONNREFUSED", None),
    )
    if number is not None
}

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


def _text_field(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        parts = [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int)) and str(item).strip()
        ]
        if parts:
            return ".".join(parts)
    return None


def _format_detail_list(items) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        loc = item.get("loc")
        loc_text = ""
        if isinstance(loc, list):
            loc_text = ".".join(str(part) for part in loc if part not in ("body", "root"))
        msg = item.get("msg") or item.get("message") or item.get("type") or ""
        msg = str(msg).strip() if msg else ""
        if loc_text and msg:
            parts.append(f"{loc_text}: {msg}")
        elif msg:
            parts.append(msg)
        elif loc_text:
            parts.append(loc_text)
    return "; ".join(parts)


def _parse_error_body(body: bytes) -> tuple[str, str | None, str | None]:
    """Return ``(message, param, code)`` from an error response body."""
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return "", None, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:400], None, None
    if not isinstance(parsed, dict):
        return text[:400], None, None

    param = _text_field(parsed.get("param"))
    code = _text_field(parsed.get("code"))
    message = ""

    detail = parsed.get("error")
    if isinstance(detail, dict):
        param = _text_field(detail.get("param")) or param
        code = _text_field(detail.get("code")) or code
        raw = detail.get("message") or detail.get("code") or detail.get("type")
        if raw:
            message = str(raw)
    elif isinstance(detail, str) and detail:
        message = detail
    elif isinstance(detail, list):
        message = _format_detail_list(detail)

    if not message:
        for key in ("message", "detail", "error_description", "msg"):
            value = parsed.get(key)
            if isinstance(value, str) and value:
                message = value
                break
            if isinstance(value, list):
                formatted = _format_detail_list(value)
                if formatted:
                    message = formatted
                    break
    if not message:
        message = text
    if param and param.lower() not in message.lower():
        message = f"{message} ({param})" if message else param
    return message[:400], param, code


def raise_for_status(status: int, body: bytes, headers, url: str) -> None:
    """Turn an HTTP error into a specific, actionable exception."""
    host = urlsplit(url).netloc or url
    message, param, code = _parse_error_body(body)
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

    raise APIError(
        f"request failed: {detail}",
        hint=hint,
        status=status,
        param=param,
        code=code,
    )


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


def _should_retry_status(status: int, *, replay_safe: bool) -> bool:
    """Return whether an HTTP status is safe to send again.

    409 is never retried: it means the request conflicts with one already
    accepted. 500/502/504 are retried only for idempotent calls, because a
    completion may already be generating. 408, 425, 429, and 503 explicitly
    ask the client to try again.
    """
    if status in _SAFE_RETRY_STATUS:
        return True
    if replay_safe and status in _IDEMPOTENT_RETRY_STATUS:
        return True
    return False


def _exception_reason(exc: BaseException) -> BaseException:
    reason = getattr(exc, "reason", None)
    if reason is None or reason is exc:
        return exc
    if isinstance(reason, BaseException):
        return reason
    return exc


def _ssl_failure(exc: BaseException) -> bool:
    reason = _exception_reason(exc)
    return isinstance(exc, ssl.SSLError) or isinstance(reason, ssl.SSLError)


def _failed_before_send(exc: BaseException) -> bool:
    """True when the TCP connection never came up, so the POST was not sent.

    urllib reports a connect timeout and a timeout while waiting for the
    status line as the same ``URLError``. Timeouts are therefore treated as
    possibly sent. Connection refused, unreachable hosts, and DNS failures
    happen before any request bytes leave the machine.
    """
    reason = _exception_reason(exc)
    if isinstance(reason, ConnectionRefusedError):
        return True
    if isinstance(reason, socket.gaierror):
        return True
    errno_value = getattr(reason, "errno", None)
    if errno_value in _BEFORE_SEND_ERRNOS:
        return True
    text = str(reason).lower()
    if "connection refused" in text or "actively refused" in text:
        return True
    return False


def _should_retry_network(exc: BaseException, *, replay_safe: bool) -> bool:
    if _ssl_failure(exc):
        return False
    if replay_safe:
        return True
    return _failed_before_send(exc)


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
    *,
    replay_safe: bool,
):
    """Open a URL, retrying only failures that are safe to send again.

    ``replay_safe`` is true for idempotent calls such as listing models.
    Completion POSTs pass false: once the request has been written, a 5xx or
    a dropped connection can mean the provider already started the generation.
    A connection refused before the handshake still retries, because nothing
    was sent. A failure while reading a response body is not handled here and
    is not retried.
    """
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
            if attempt < retries and _should_retry_status(exc.code, replay_safe=replay_safe):
                delay = _retry_delay(exc.headers, attempt)
                if delay is not None:
                    time.sleep(delay)
                    continue
            raise_for_status(exc.code, body, exc.headers, url)
        except (error.URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
            if attempt < retries and _should_retry_network(exc, replay_safe=replay_safe):
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
    """POST JSON and return an open streaming response.

    The POST is not replayed after the server has seen it. Retries are limited
    to 408/425/429/503 and to connection failures that happen before send.
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return _open("POST", url, data, headers, timeout, retries, replay_safe=False)


def request_json(
    url: str,
    headers: dict[str, str],
    payload: dict | None = None,
    *,
    method: str = "GET",
    timeout: float = 60.0,
    retries: int = MAX_RETRIES,
):
    """Perform a request and decode a JSON object or array.

    GET/HEAD may retry server errors. Other methods follow the completion
    rule: do not replay after the request was sent. A read error after the
    status line is never retried here, because the server already accepted
    the call.
    """
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    replay_safe = method.upper() in {"GET", "HEAD", "OPTIONS"}
    response = _open(method, url, data, headers, timeout, retries, replay_safe=replay_safe)
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


def bound_read_timeout(response, seconds: float) -> bool:
    """Best-effort cap on later socket reads. Returns whether a socket was found.

    This does not abort a request the server has already accepted. It only
    asks the local socket to wake up so cancellation and the overall deadline
    can be noticed between reads.
    """

    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None or not hasattr(sock, "settimeout"):
        return False
    try:
        sock.settimeout(max(0.1, float(seconds)))
    except OSError:
        return False
    return True


def iter_sse(response, *, keep_reading=None):
    """Yield decoded SSE data payloads, translating read errors.

    ``keep_reading`` may return true after a short socket timeout so the
    caller can keep waiting until its own deadline. A false result, or no
    callback, ends the stream. This is cooperative: it does not promise to
    stop the provider.
    """

    while True:
        try:
            yield from _iter_sse(response)
            return
        except (socket.timeout, TimeoutError) as exc:
            if keep_reading is not None and keep_reading():
                continue
            raise NetworkError(
                f"connection broke while streaming: {exc}",
                hint="The local read timed out. The provider may still be generating; "
                "this does not cancel the request on the server.",
            ) from None
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise NetworkError(
                f"connection broke while streaming: {exc}",
                hint="The provider may have timed out. Retry, or use --timeout with a larger value.",
            ) from None
