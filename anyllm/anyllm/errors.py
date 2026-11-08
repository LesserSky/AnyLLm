"""User-facing error types.

Every error carries a short ``message`` plus an optional ``hint`` that tells the
user what to do next. The CLI prints both, so errors should stay readable.
"""

from __future__ import annotations


class AnyLLMError(Exception):
    """Base class for errors that should be shown without a traceback."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class UsageError(AnyLLMError):
    """The command line was used incorrectly."""


class ConfigError(AnyLLMError):
    """The saved configuration or key store could not be used."""


class AuthError(AnyLLMError):
    """The provider rejected the credential (HTTP 401/403)."""


class NotFoundError(AnyLLMError):
    """The provider could not find the endpoint or model (HTTP 404)."""


class RateLimitError(AnyLLMError):
    """The provider is rate limiting or the quota is exhausted."""


class NetworkError(AnyLLMError):
    """The provider could not be reached."""


class APIError(AnyLLMError):
    """The provider returned an error response."""

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        status: int | None = None,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.status = status
        self.param = param
        self.code = code


class ProviderError(AnyLLMError):
    """The provider reported an error inside a stream."""
