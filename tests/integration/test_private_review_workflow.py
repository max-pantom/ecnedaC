from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cadence.common.config import CadenceConfig, load_config
from cadence.dataset.downloaders import (
    DownloaderChain,
    DownloadResult,
    SourceInspection,
)
from cadence.dataset.media import MediaMetadata
from cadence.dataset.records import SourceRecord
from cadence.dataset.service import DatasetIntakeService
from cadence.ingestion.fixtures import generate_fixtures
from cadence.ingestion.manifest import load_manifest
from cadence.review.app import create_app
from cadence.review.auth import SESSION_COOKIE, SessionSigner

ADMIN_SECRET = "synthetic-review-administrator-secret"


class SyntheticDownloader:
    name = "synthetic-copy"

    def __init__(self, fixture: Path, metadata: MediaMetadata) -> None:
        self.fixture = fixture
        self.metadata = metadata

    def inspect(self, url: str) -> SourceInspection:
        del url
        return SourceInspection(
            supported=True,
            method=self.name,
            title="Synthetic review acceptance fixture",
            publisher_or_creator="Cadence tests",
            platform="fixtures.cadence.invalid",
            duration_seconds=self.metadata.duration_seconds,
            content_length_bytes=self.fixture.stat().st_size,
        )

    def download(self, source: SourceRecord, destination: Path) -> DownloadResult:
        del source
        destination.write_bytes(self.fixture.read_bytes())
        return DownloadResult(
            path=destination,
            bytes_written=destination.stat().st_size,
            method=self.name,
        )


class SyntheticMedia:
    def __init__(self, metadata: MediaMetadata) -> None:
        self.metadata = metadata

    def probe(self, path: Path) -> MediaMetadata:
        assert path.is_file()
        return self.metadata

    def normalize(self, source: Path, destination: Path) -> MediaMetadata:
        destination.write_bytes(source.read_bytes())
        return self.probe(destination)

    def extract_segment(
        self,
        source: Path,
        destination: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> MediaMetadata:
        del start_seconds, duration_seconds
        destination.write_bytes(source.read_bytes())
        return self.probe(destination)


def _config(tmp_path: Path) -> CadenceConfig:
    base = load_config("configs/test.yaml")
    return base.model_copy(
        update={"paths": base.paths.model_copy(update={"intake_root": tmp_path / "private-intake"})}
    )


def _login(client: TestClient) -> str:
    response = client.post(
        "/login",
        data={"actor": "acceptance-reviewer", "secret": ADMIN_SECRET},
        follow_redirects=False,
    )
    assert response.status_code == 303
    token = client.cookies.get(SESSION_COOKIE)
    assert token is not None
    return SessionSigner(ADMIN_SECRET).verify(token).csrf_token


def test_synthetic_submission_is_reviewed_and_built_through_private_console(
    tmp_path: Path,
) -> None:
    fixture_manifest = generate_fixtures(tmp_path / "generated-fixtures")
    fixture_entry = load_manifest(fixture_manifest)[0]
    fixture = fixture_entry.path
    assert fixture is not None
    metadata = MediaMetadata(
        duration_seconds=fixture_entry.duration_s,
        fps=fixture_entry.fps,
        width=fixture_entry.width,
        height=fixture_entry.height,
        audio_sample_rate=fixture_entry.audio_sample_rate,
        has_video=fixture_entry.has_video,
        has_audio=fixture_entry.has_audio,
    )

    config = _config(tmp_path)
    service = DatasetIntakeService(
        config,
        downloaders=DownloaderChain([SyntheticDownloader(fixture, metadata)]),
        media=SyntheticMedia(metadata),
    )
    source, created = service.add_source(
        "https://fixtures.cadence.invalid/review-console.mp4",
        submitted_by="synthetic-acceptance",
    )
    assert created is True
    inspected = service.inspect_source(source.source_id)

    client = TestClient(create_app(config, service=service, auth_secret=ADMIN_SECRET))
    csrf = _login(client)
    headers = {"x-csrf-token": csrf}

    rights = client.post(
        f"/api/v1/sources/{source.source_id}/rights",
        json={
            "status": "licensed",
            "license_notes": "Synthetic fixture permission",
            "reason": "Synthetic fixture is permitted for acceptance testing",
            "evidence_reference": "private://evidence/SYNTHETIC-001",
            "expected_revision": inspected.revision,
        },
        headers=headers,
    )
    assert rights.status_code == 200

    source_decision = client.post(
        f"/api/v1/sources/{source.source_id}/source-decision",
        json={
            "decision": "approved",
            "reason": "Synthetic launch sequence is relevant",
            "expected_revision": rights.json()["revision"],
        },
        headers=headers,
    )
    assert source_decision.status_code == 200

    download_decision = client.post(
        f"/api/v1/sources/{source.source_id}/download-decision",
        json={
            "decision": "approved",
            "reason": "Synthetic fixture acquisition is authorized",
            "expected_revision": source_decision.json()["revision"],
        },
        headers=headers,
    )
    assert download_decision.status_code == 200

    downloaded = service.download_source(source.source_id)
    assert downloaded.normalized_path is not None
    assert downloaded.normalized_path.is_relative_to(config.paths.intake_root)

    eligibility = client.post(
        f"/api/v1/sources/{source.source_id}/eligibility",
        json={
            "eligible": True,
            "reason": "All synthetic training gates are satisfied",
            "expected_revision": downloaded.revision,
        },
        headers=headers,
    )
    assert eligibility.status_code == 200
    assert eligibility.json()["eligible_for_training"] is True

    segments = service.suggest_source_segments(source.source_id)
    assert segments
    segment = segments[0]
    segment_decision = client.post(
        f"/api/v1/segments/{segment.segment_id}/decision",
        json={
            "decision": "approved",
            "reason": "Aligned synthetic motion and audio event",
            "expected_revision": segment.revision,
        },
        headers=headers,
    )
    assert segment_decision.status_code == 200

    stale = client.post(
        f"/api/v1/sources/{source.source_id}/source-decision",
        json={
            "decision": "approved",
            "reason": "Decision submitted from a stale page",
            "expected_revision": eligibility.json()["revision"],
        },
        headers=headers,
    )
    assert stale.status_code == 409
    assert stale.json()["actual_revision"] > stale.json()["expected_revision"]

    built = client.post(
        "/api/v1/datasets/build",
        json={
            "dataset_name": "synthetic-review",
            "reason": "Synthetic end-to-end acceptance build",
            "evidence_reference": "private://build/SYNTHETIC-001",
            "expected_revision": 0,
        },
        headers=headers,
    )
    assert built.status_code == 200
    dataset = built.json()
    assert dataset["version"] == 1

    manifest_path = Path(dataset["manifest_path"])
    report_path = Path(dataset["report_path"])
    assert manifest_path.is_relative_to(config.paths.intake_root)
    assert report_path.is_relative_to(config.paths.intake_root)
    assert manifest_path.is_file()
    assert report_path.is_file()
    manifest = load_manifest(manifest_path)
    assert len(manifest) == 1
    assert manifest[0].source_asset_id == source.source_id
    assert manifest[0].eligible_for_contrastive is True

    events = service.list_audit_events()
    assert [event.action for event in events] == [
        "rights_updated",
        "source_approval_updated",
        "download_approval_updated",
        "training_eligibility_updated",
        "segment_approval_updated",
        "dataset_built",
    ]
    assert all(event.actor == "acceptance-reviewer" for event in events)
    assert service.dataset_report("synthetic-review")["version"] == 1
    report_page = client.get("/datasets/synthetic-review")
    assert report_page.status_code == 200
    assert "immutable" in report_page.text.lower()
    assert "synthetic-review" in report_page.text
