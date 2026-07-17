"""Small, dependency-free signed-session and CSRF helpers for the review console."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Final

SESSION_COOKIE: Final = "cadence_review_session"
SESSION_MAX_AGE_SECONDS: Final = 8 * 60 * 60


class AuthenticationError(ValueError):
    """Raised when a signed review-console session cannot be trusted."""


@dataclass(frozen=True)
class ReviewSession:
    actor: str
    csrf_token: str
    issued_at: int


def configured_secret(explicit: str | None = None) -> str:
    """Return the administrator secret without ever supplying an unsafe default."""
    secret = explicit or os.getenv("CADENCE_REVIEW_ADMIN_SECRET")
    if not secret:
        raise ValueError(
            "review UI requires auth_secret or CADENCE_REVIEW_ADMIN_SECRET"
        )
    if len(secret) < 32:
        raise ValueError("review UI administrator secret must contain at least 32 characters")
    return secret


class SessionSigner:
    """Issue and verify compact HMAC-signed session cookies."""

    def __init__(self, secret: str, *, max_age_seconds: int = SESSION_MAX_AGE_SECONDS) -> None:
        if len(secret) < 32:
            raise ValueError("review UI administrator secret must contain at least 32 characters")
        self._key = hashlib.sha256(f"cadence-review-session:{secret}".encode()).digest()
        self.max_age_seconds = max_age_seconds

    def issue(self, actor: str) -> tuple[str, ReviewSession]:
        normalized_actor = actor.strip()
        if not normalized_actor or len(normalized_actor) > 100:
            raise ValueError("actor must contain between 1 and 100 characters")
        session = ReviewSession(
            actor=normalized_actor,
            csrf_token=secrets.token_urlsafe(32),
            issued_at=int(time.time()),
        )
        payload = json.dumps(
            {
                "actor": session.actor,
                "csrf": session.csrf_token,
                "issued_at": session.issued_at,
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
            expected_signature = _encode(
                hmac.digest(self._key, encoded.encode(), "sha256")
            )
            if not hmac.compare_digest(supplied_signature, expected_signature):
                raise AuthenticationError("invalid session signature")
            raw = json.loads(_decode(encoded))
            session = ReviewSession(
                actor=str(raw["actor"]),
                csrf_token=str(raw["csrf"]),
                issued_at=int(raw["issued_at"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, AuthenticationError):
                raise
            raise AuthenticationError("invalid session") from exc

        current_time = int(time.time()) if now is None else now
        age = current_time - session.issued_at
        if age < -60 or age > self.max_age_seconds:
            raise AuthenticationError("session expired")
        if not session.actor or len(session.actor) > 100 or not session.csrf_token:
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
