from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from cadence.common.config import CadenceConfig, load_config
from cadence.review.app import create_app
from cadence.review.auth import (
    SESSION_COOKIE,
    AuthenticationError,
    ReviewSession,
    SessionSigner,
    TunnelBasicAuth,
)
from cadence.review.models import StaleRevisionError

SECRET = "test-administrator-secret-that-is-long-enough"
READONLY_SECRET = "test-read-only-secret-that-is-long-enough"


class FakeRegistry:
    def __init__(self, source: SimpleNamespace, segment: SimpleNamespace | None = None) -> None:
        self.source = source
        self.segment = segment

    def get_source(self, source_id: object) -> SimpleNamespace:
        if str(source_id) != str(self.source.source_id):
            raise KeyError(source_id)
        return self.source

    def get_segment(self, segment_id: object) -> SimpleNamespace:
        if self.segment is None or str(segment_id) != str(self.segment.segment_id):
            raise KeyError(segment_id)
        return self.segment


class FakeService:
    def __init__(self, source: SimpleNamespace) -> None:
        self.registry = FakeRegistry(source)
        self.source = source
        self.queue: list[object] = []

    def review_queue(self) -> list[object]:
        return self.queue

    def list_datasets(self) -> list[object]:
        return []

    def list_audit_events(
        self, *, entity_type: str | None = None, entity_id: str | None = None
    ) -> list[object]:
        return []

    def list_segments(self, source_id: object) -> list[object]:
        self.registry.get_source(source_id)
        return []

    def set_source_approval(
        self,
        source_id: object,
        status: object,
        *,
        actor: str,
        reason: str,
        evidence_reference: object,
        expected_revision: int,
    ) -> SimpleNamespace:
        del status, actor, reason, evidence_reference
        source = self.registry.get_source(source_id)
        if expected_revision != source.revision:
            raise StaleRevisionError(
                "source",
                UUID(str(source.source_id)),
                expected_revision=expected_revision,
                actual_revision=source.revision,
            )
        return source


def _config(tmp_path: Path) -> CadenceConfig:
    base = load_config("configs/test.yaml")
    paths = base.paths.model_copy(update={"intake_root": tmp_path / "private"})
    return base.model_copy(update={"paths": paths})


def _source(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source_id=uuid4(),
        schema_version="0.1.0",
        url="https://sources.example.invalid/private-launch",
        submitted_by="private-operator",
        collection_method="user-submitted-url",
        submitted_at="2026-07-17T00:00:00Z",
        title="Private launch metadata",
        publisher_or_creator="Example creator",
        platform="sources.example.invalid",
        duration_seconds=42.0,
        content_length_bytes=1234,
        inspection_status="supported",
        download_status="not_requested",
        rights_status="unverified",
        source_approval="pending",
        download_approval="pending",
        eligible_for_training=False,
        license_notes="must-not-reach-read-only-reviewer",
        checksum_sha256="a" * 64,
        normalized_path=path,
        storage_path=None,
        revision=3,
    )


def _client(tmp_path: Path, source_path: Path) -> tuple[TestClient, FakeService]:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    service = FakeService(_source(source_path))
    app = create_app(
        _config(tmp_path),
        service=service,  # type: ignore[arg-type]
        auth_secret=SECRET,
    )
    return TestClient(app), service


def _login(client: TestClient) -> str:
    response = client.post(
        "/login",
        data={"actor": "reviewer", "secret": SECRET},
        follow_redirects=False,
    )
    assert response.status_code == 303
    token = client.cookies.get(SESSION_COOKIE)
    assert token is not None
    return SessionSigner(SECRET).verify(token).csrf_token


def _readonly_login(client: TestClient) -> ReviewSession:
    response = client.post(
        "/login",
        data={"actor": "assistant-reviewer", "secret": READONLY_SECRET},
        follow_redirects=False,
    )
    assert response.status_code == 303
    token = client.cookies.get(SESSION_COOKIE)
    assert token is not None
    return SessionSigner(SECRET).verify(token)


def _basic_header(username: str, password: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"authorization": f"Basic {encoded}"}


def test_health_is_public_but_review_endpoints_require_authentication(tmp_path: Path) -> None:
    media = tmp_path / "private" / "source.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"media")
    client, service = _client(tmp_path, media)

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/").status_code == 200  # TestClient follows the login redirect.
    assert client.get("/api/v1/review/queue").status_code == 401
    assert client.get(f"/api/v1/media/{service.source.source_id}").status_code == 401


def test_console_style_is_embedded_and_allowed_by_csp(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, tmp_path / "private" / "source.mp4")

    login_page = client.get("/login")
    assert login_page.status_code == 200
    assert '<link rel="stylesheet"' not in login_page.text
    style = login_page.text.split("<style>", 1)[1].split("</style>", 1)[0]
    style_hash = base64.b64encode(hashlib.sha256(style.encode()).digest()).decode()
    assert f"'sha256-{style_hash}'" in login_page.headers["content-security-policy"]
    assert ".centered-card" in style

    _login(client)
    queue_page = client.get("/")
    assert queue_page.status_code == 200
    assert 'class="page-heading"' in queue_page.text
    assert 'class="panel queue-panel"' in queue_page.text
    assert 'class="empty-state"' in queue_page.text


def test_login_cookie_is_signed_http_only_and_strict(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, tmp_path / "private" / "source.mp4")
    failed = client.post(
        "/login",
        data={"actor": "reviewer", "secret": "not-the-secret"},
        follow_redirects=False,
    )
    assert failed.status_code == 401

    response = client.post(
        "/login",
        data={"actor": "reviewer", "secret": SECRET},
        follow_redirects=False,
    )
    cookie = response.headers["set-cookie"].lower()
    assert response.status_code == 303
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    token = client.cookies[SESSION_COOKIE]
    session = SessionSigner(SECRET).verify(token)
    assert session.actor == "reviewer"
    assert session.role == "administrator"
    assert session.can_mutate is True
    client.cookies.set(SESSION_COOKIE, f"{token[:-1]}x")
    assert client.get("/api/v1/review/queue").status_code == 401


def test_readonly_reviewer_sees_only_allowlisted_source_metadata(tmp_path: Path) -> None:
    media = tmp_path / "private" / "source.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"private-media-must-not-be-served")
    service = FakeService(_source(media))
    app = create_app(
        _config(tmp_path),
        service=service,  # type: ignore[arg-type]
        auth_secret=SECRET,
        reviewer_secret=READONLY_SECRET,
        reviewer_session_max_age_seconds=600,
    )
    client = TestClient(app)

    session = _readonly_login(client)
    assert session.role == "reviewer"
    assert session.can_mutate is False
    assert session.expires_at - session.issued_at == 600

    service.queue = [
        SimpleNamespace(entity_type="source", entity_id=service.source.source_id),
        SimpleNamespace(entity_type="segment", entity_id=uuid4()),
    ]
    queue = client.get("/api/v1/review/queue")
    assert queue.status_code == 200
    assert [item["entity_type"] for item in queue.json()] == ["source"]

    page = client.get(f"/sources/{service.source.source_id}")
    assert page.status_code == 200
    assert "Private launch metadata" in page.text
    assert "Example creator" in page.text
    assert str(service.source.url) in page.text
    assert "Read-only reviewer" in page.text
    assert "must-not-reach-read-only-reviewer" not in page.text
    assert "/source-decision" not in page.text
    assert "/download-decision" not in page.text
    assert "/eligibility" not in page.text
    assert "<video" not in page.text
    assert "Audit trail" not in page.text

    response = client.get(f"/api/v1/sources/{service.source.source_id}")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"source"}
    assert payload["source"]["title"] == "Private launch metadata"
    assert payload["source"]["url"] == service.source.url
    forbidden_fields = {
        "normalized_path",
        "storage_path",
        "storage_uri",
        "normalized_uri",
        "checksum_sha256",
        "license_notes",
        "canonical_url",
        "error_state",
    }
    assert forbidden_fields.isdisjoint(payload["source"])
    assert "audit_events" not in payload


def test_readonly_reviewer_cannot_mutate_or_access_sensitive_routes(
    tmp_path: Path,
) -> None:
    media = tmp_path / "private" / "source.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"private-media-must-not-be-served")
    service = FakeService(_source(media))
    app = create_app(
        _config(tmp_path),
        service=service,  # type: ignore[arg-type]
        auth_secret=SECRET,
        reviewer_secret=READONLY_SECRET,
        reviewer_session_max_age_seconds=600,
    )
    client = TestClient(app)
    session = _readonly_login(client)
    headers = {"x-csrf-token": session.csrf_token}
    source_id = service.source.source_id

    mutations = [
        (
            f"/api/v1/sources/{source_id}/rights",
            {"status": "licensed", "reason": "forbidden", "expected_revision": 3},
        ),
        (
            f"/api/v1/sources/{source_id}/source-decision",
            {"decision": "approved", "reason": "forbidden", "expected_revision": 3},
        ),
        (
            f"/api/v1/sources/{source_id}/download-decision",
            {"decision": "approved", "reason": "forbidden", "expected_revision": 3},
        ),
        (
            f"/api/v1/sources/{source_id}/eligibility",
            {"eligible": True, "reason": "forbidden", "expected_revision": 3},
        ),
        (
            f"/api/v1/segments/{uuid4()}/decision",
            {"decision": "approved", "reason": "forbidden", "expected_revision": 0},
        ),
        (
            "/api/v1/datasets/build",
            {"dataset_name": "forbidden", "reason": "forbidden", "expected_revision": 0},
        ),
    ]
    for endpoint, payload in mutations:
        assert client.post(endpoint, json=payload, headers=headers).status_code == 403

    assert client.get(f"/api/v1/media/{source_id}").status_code == 403
    assert client.get(f"/api/v1/sources/{source_id}/segments").status_code == 403
    assert client.get(f"/segments/{uuid4()}").status_code == 403
    assert client.get("/datasets/private-build").status_code == 403
    assert service.source.revision == 3


def test_readonly_session_expiry_and_secret_separation(tmp_path: Path) -> None:
    signer = SessionSigner(SECRET)
    token, session = signer.issue(
        "assistant-reviewer",
        role="reviewer",
        max_age_seconds=300,
    )
    assert signer.verify(token, now=session.expires_at).role == "reviewer"
    with pytest.raises(AuthenticationError, match="expired"):
        signer.verify(token, now=session.expires_at + 1)

    service = FakeService(_source(tmp_path / "private" / "source.mp4"))
    with pytest.raises(ValueError, match="must be different"):
        create_app(
            _config(tmp_path),
            service=service,  # type: ignore[arg-type]
            auth_secret=SECRET,
            reviewer_secret=SECRET,
        )


def test_mutations_are_post_only_and_require_valid_csrf(tmp_path: Path) -> None:
    client, service = _client(tmp_path, tmp_path / "private" / "source.mp4")
    csrf = _login(client)
    endpoint = f"/api/v1/sources/{service.source.source_id}/source-decision"
    body = {
        "decision": "approved",
        "reason": "Relevant launch sequence",
        "expected_revision": 3,
    }

    assert client.get(endpoint).status_code == 405
    assert client.post(endpoint, json=body).status_code == 403
    assert client.post(endpoint, json=body, headers={"x-csrf-token": "wrong"}).status_code == 403
    response = client.post(endpoint, json=body, headers={"x-csrf-token": csrf})
    assert response.status_code == 200
    assert response.json()["revision"] == 3


def test_server_rendered_form_mutation_returns_to_record(tmp_path: Path) -> None:
    client, service = _client(tmp_path, tmp_path / "private" / "source.mp4")
    csrf = _login(client)
    response = client.post(
        f"/api/v1/sources/{service.source.source_id}/source-decision",
        data={
            "decision": "approved",
            "reason": "Relevant launch sequence",
            "expected_revision": "3",
            "csrf_token": csrf,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/sources/{service.source.source_id}"


def test_training_eligibility_form_is_locked_until_normalization(tmp_path: Path) -> None:
    client, service = _client(tmp_path, tmp_path / "private" / "source.mp4")
    _login(client)

    page = client.get(f"/sources/{service.source.source_id}")

    assert page.status_code == 200
    assert "Training eligibility locked" in page.text
    assert "requires a normalized download" not in page.text
    assert f"/api/v1/sources/{service.source.source_id}/eligibility" not in page.text


def test_stale_revision_returns_conflict_details(tmp_path: Path) -> None:
    client, service = _client(tmp_path, tmp_path / "private" / "source.mp4")
    csrf = _login(client)
    response = client.post(
        f"/api/v1/sources/{service.source.source_id}/source-decision",
        json={
            "decision": "approved",
            "reason": "Review from an old browser tab",
            "expected_revision": 2,
        },
        headers={"x-csrf-token": csrf},
    )
    assert response.status_code == 409
    assert response.json()["actual_revision"] == 3
    assert response.json()["expected_revision"] == 2


def test_registered_media_supports_bounded_ranges(tmp_path: Path) -> None:
    media = tmp_path / "private" / "sources" / "normalized" / "source.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"0123456789")
    client, service = _client(tmp_path, media)
    _login(client)

    response = client.get(
        f"/api/v1/media/{service.source.source_id}",
        headers={"range": "bytes=2-5"},
    )
    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.headers["cache-control"] == "private, no-store"
    assert (
        client.get(
            f"/api/v1/media/{service.source.source_id}",
            headers={"range": "bytes=100-200"},
        ).status_code
        == 416
    )


def test_registered_paths_outside_private_intake_root_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "not-private" / "source.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"private media")
    client, service = _client(tmp_path, outside)
    _login(client)

    response = client.get(f"/api/v1/media/{service.source.source_id}")
    assert response.status_code == 403
    assert "outside" in response.json()["detail"]


def test_media_route_never_interprets_entity_id_as_a_path(tmp_path: Path) -> None:
    media = tmp_path / "private" / "source.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"media")
    client, _ = _client(tmp_path, media)
    _login(client)

    assert client.get("/api/v1/media/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_tunnel_mode_requires_outer_auth_and_forces_secure_cookie(tmp_path: Path) -> None:
    media = tmp_path / "private" / "source.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"media")
    service = FakeService(_source(media))
    tunnel_auth = TunnelBasicAuth(username="cadence", password="p" * 24)
    app = create_app(
        _config(tmp_path),
        service=service,  # type: ignore[arg-type]
        auth_secret=SECRET,
        secure_cookies=True,
        tunnel_basic_auth=tunnel_auth,
        allowed_hosts=("testserver",),
        login_max_failures=2,
    )
    client = TestClient(app)
    outer_headers = _basic_header(tunnel_auth.username, tunnel_auth.password)

    unauthorized = client.get("/healthz")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"].startswith("Basic ")
    assert client.get("/healthz", headers=outer_headers).status_code == 200

    login = client.post(
        "/login",
        data={"actor": "reviewer", "secret": SECRET},
        headers=outer_headers,
        follow_redirects=False,
    )
    assert login.status_code == 303
    assert "secure" in login.headers["set-cookie"].lower()


def test_tunnel_mode_rate_limits_administrator_secret_failures(tmp_path: Path) -> None:
    service = FakeService(_source(tmp_path / "private" / "source.mp4"))
    tunnel_auth = TunnelBasicAuth(username="cadence", password="p" * 24)
    app = create_app(
        _config(tmp_path),
        service=service,  # type: ignore[arg-type]
        auth_secret=SECRET,
        tunnel_basic_auth=tunnel_auth,
        allowed_hosts=("testserver",),
        login_max_failures=2,
    )
    client = TestClient(app)
    headers = _basic_header(tunnel_auth.username, tunnel_auth.password)
    body = {"actor": "reviewer", "secret": "incorrect"}

    assert client.post("/login", data=body, headers=headers).status_code == 401
    assert client.post("/login", data=body, headers=headers).status_code == 401
    limited = client.post("/login", data=body, headers=headers)
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "300"


def test_tunnel_mode_rejects_unexpected_local_host_header(tmp_path: Path) -> None:
    service = FakeService(_source(tmp_path / "private" / "source.mp4"))
    tunnel_auth = TunnelBasicAuth(username="cadence", password="p" * 24)
    app = create_app(
        _config(tmp_path),
        service=service,  # type: ignore[arg-type]
        auth_secret=SECRET,
        tunnel_basic_auth=tunnel_auth,
        allowed_hosts=("127.0.0.1",),
    )
    client = TestClient(app)
    headers = {
        **_basic_header(tunnel_auth.username, tunnel_auth.password),
        "host": "unexpected.invalid",
    }

    assert client.get("/healthz", headers=headers).status_code == 400
