"""Small, dependency-free signed-session and CSRF helpers for the review console."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

SESSION_COOKIE: Final = "cadence_review_session"
SESSION_MAX_AGE_SECONDS: Final = 8 * 60 * 60
READONLY_SESSION_MAX_AGE_SECONDS: Final = 2 * 60 * 60
READONLY_SESSION_MIN_AGE_SECONDS: Final = 5 * 60
SessionRole: TypeAlias = Literal["administrator", "reviewer"]


class AuthenticationError(ValueError):
    """Raised when a signed review-console session cannot be trusted."""


@dataclass(frozen=True)
class ReviewSession:
    actor: str
    csrf_token: str
    issued_at: int
    expires_at: int
    role: SessionRole

    @property
    def can_mutate(self) -> bool:
        return self.role == "administrator"


@dataclass(frozen=True)
class TunnelBasicAuth:
    """Ephemeral outer authentication for a temporary public tunnel."""

    username: str
    password: str

    def __post_init__(self) -> None:
        if not self.username or len(self.username) > 100:
            raise ValueError("tunnel username must contain between 1 and 100 characters")
        if len(self.password) < 24:
            raise ValueError("tunnel password must contain at least 24 characters")

    def verify(self, authorization: str | None) -> bool:
        if authorization is None or not authorization.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
            supplied_username, supplied_password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError):
            return False
        return hmac.compare_digest(supplied_username, self.username) and hmac.compare_digest(
            supplied_password, self.password
        )


class LoginAttemptLimiter:
    """Small process-local limiter for administrator-secret failures."""

    def __init__(self, maximum_failures: int = 5, window_seconds: int = 300) -> None:
        if maximum_failures < 1 or window_seconds < 1:
            raise ValueError("login rate-limit values must be positive")
        self.maximum_failures = maximum_failures
        self.window_seconds = window_seconds
        self._failures: deque[float] = deque()
        self._lock = threading.Lock()

    def allowed(self, *, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            self._discard_expired(timestamp)
            return len(self._failures) < self.maximum_failures

    def record_failure(self, *, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            self._discard_expired(timestamp)
            self._failures.append(timestamp)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()

    def _discard_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._failures and self._failures[0] <= cutoff:
            self._failures.popleft()


def configured_secret(explicit: str | None = None) -> str:
    """Return the administrator secret without ever supplying an unsafe default."""
    secret = explicit or os.getenv("CADENCE_REVIEW_ADMIN_SECRET")
    if not secret:
        raise ValueError("review UI requires auth_secret or CADENCE_REVIEW_ADMIN_SECRET")
    if len(secret) < 32:
        raise ValueError("review UI administrator secret must contain at least 32 characters")
    return secret


def configured_reviewer_secret(explicit: str | None = None) -> str | None:
    """Return optional runtime-only read-only access secret."""
    secret = explicit or os.getenv("CADENCE_REVIEW_READONLY_SECRET")
    if not secret:
        return None
    if len(secret) < 32:
        raise ValueError("review UI read-only secret must contain at least 32 characters")
    return secret


def configured_reviewer_session_max_age(explicit: int | None = None) -> int:
    """Return bounded read-only session lifetime."""
    raw = explicit
    if raw is None:
        environment_value = os.getenv("CADENCE_REVIEW_READONLY_MAX_AGE_SECONDS")
        if environment_value:
            try:
                raw = int(environment_value)
            except ValueError as exc:
                raise ValueError(
                    "CADENCE_REVIEW_READONLY_MAX_AGE_SECONDS must be an integer"
                ) from exc
    seconds = READONLY_SESSION_MAX_AGE_SECONDS if raw is None else raw
    if not READONLY_SESSION_MIN_AGE_SECONDS <= seconds <= READONLY_SESSION_MAX_AGE_SECONDS:
        raise ValueError("read-only session lifetime must be between 300 and 7200 seconds")
    return seconds


class SessionSigner:
    """Issue and verify compact HMAC-signed session cookies."""

    def __init__(self, secret: str, *, max_age_seconds: int = SESSION_MAX_AGE_SECONDS) -> None:
        if len(secret) < 32:
            raise ValueError("review UI administrator secret must contain at least 32 characters")
        self._key = hashlib.sha256(f"cadence-review-session:{secret}".encode()).digest()
        self.max_age_seconds = max_age_seconds

    def issue(
        self,
        actor: str,
        *,
        role: SessionRole = "administrator",
        max_age_seconds: int | None = None,
    ) -> tuple[str, ReviewSession]:
        normalized_actor = actor.strip()
        if not normalized_actor or len(normalized_actor) > 100:
            raise ValueError("actor must contain between 1 and 100 characters")
        lifetime = self.max_age_seconds if max_age_seconds is None else max_age_seconds
        if not 1 <= lifetime <= self.max_age_seconds:
            raise ValueError("session lifetime exceeds signer maximum")
        issued_at = int(time.time())
        session = ReviewSession(
            actor=normalized_actor,
            csrf_token=secrets.token_urlsafe(32),
            issued_at=issued_at,
            expires_at=issued_at + lifetime,
            role=role,
        )
        payload = json.dumps(
            {
                "actor": session.actor,
                "csrf": session.csrf_token,
                "issued_at": session.issued_at,
                "expires_at": session.expires_at,
                "role": session.role,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        encoded = _encode(payload)
        signature = _encode(hmac.digest(self._key, encoded.encode(), "sha256"))
        return f"{encoded}.{signature}", session

    def verify(self, token: str, *, now: int | None = None) -> ReviewSession:
        try:
            encoded, supplied_signature = token.split(".", 1)
            expected_signature = _encode(hmac.digest(self._key, encoded.encode(), "sha256"))
            if not hmac.compare_digest(supplied_signature, expected_signature):
                raise AuthenticationError("invalid session signature")
            raw = json.loads(_decode(encoded))
            session = ReviewSession(
                actor=str(raw["actor"]),
                csrf_token=str(raw["csrf"]),
                issued_at=int(raw["issued_at"]),
                expires_at=int(raw["expires_at"]),
                role=cast(SessionRole, str(raw["role"])),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, AuthenticationError):
                raise
            raise AuthenticationError("invalid session") from exc

        current_time = int(time.time()) if now is None else now
        age = current_time - session.issued_at
        if (
            age < -60
            or age > self.max_age_seconds
            or current_time > session.expires_at
            or session.expires_at <= session.issued_at
            or session.expires_at - session.issued_at > self.max_age_seconds
        ):
            raise AuthenticationError("session expired")
        if (
            not session.actor
            or len(session.actor) > 100
            or not session.csrf_token
            or session.role not in {"administrator", "reviewer"}
        ):
            raise AuthenticationError("invalid session claims")
        return session


def verify_administrator_secret(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode(), expected.encode())


def verify_csrf(supplied: str | None, session: ReviewSession) -> bool:
    return supplied is not None and hmac.compare_digest(supplied, session.csrf_token)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
