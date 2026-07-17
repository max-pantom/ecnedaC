from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from cadence.common.config import CadenceConfig, load_config
from cadence.review.app import create_app
from cadence.review.auth import SESSION_COOKIE, SessionSigner
from cadence.review.models import StaleRevisionError

SECRET = "test-administrator-secret-that-is-long-enough"


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

    def review_queue(self) -> list[object]:
        return []

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


def test_health_is_public_but_review_endpoints_require_authentication(tmp_path: Path) -> None:
    media = tmp_path / "private" / "source.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"media")
    client, service = _client(tmp_path, media)

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/").status_code == 200  # TestClient follows the login redirect.
    assert client.get("/api/v1/review/queue").status_code == 401
    assert client.get(f"/api/v1/media/{service.source.source_id}").status_code == 401


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
    assert SessionSigner(SECRET).verify(token).actor == "reviewer"
    client.cookies.set(SESSION_COOKIE, f"{token[:-1]}x")
    assert client.get("/api/v1/review/queue").status_code == 401


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
    assert (
        client.post(endpoint, json=body, headers={"x-csrf-token": "wrong"}).status_code
        == 403
    )
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
    assert client.get(
        f"/api/v1/media/{service.source.source_id}",
        headers={"range": "bytes=100-200"},
    ).status_code == 416


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
