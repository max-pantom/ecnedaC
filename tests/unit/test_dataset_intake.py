from pathlib import Path

import pytest

from cadence.common.config import CadenceConfig, load_config
from cadence.dataset.downloaders import DownloaderChain, DownloadResult, SourceInspection
from cadence.dataset.media import MediaMetadata
from cadence.dataset.records import (
    ApprovalStatus,
    DownloadStatus,
    InspectionStatus,
    RightsStatus,
    SourceRecord,
)
from cadence.dataset.service import DatasetIntakeService


class FakeDownloader:
    name = "fake"

    def __init__(self, fixture: Path, *, supported: bool = True, fail: bool = False) -> None:
        self.fixture = fixture
        self.supported = supported
        self.fail = fail

    def inspect(self, url: str) -> SourceInspection:
        return SourceInspection(
            self.supported,
            self.name,
            title="Launch film",
            publisher_or_creator="Fixture Studio",
            platform="fixture.invalid",
            duration_seconds=8.0,
            content_length_bytes=self.fixture.stat().st_size,
            error=None if self.supported else "unsupported fixture URL",
        )

    def download(self, source: SourceRecord, destination: Path) -> DownloadResult:
        if self.fail:
            raise RuntimeError("simulated download failure")
        destination.write_bytes(self.fixture.read_bytes())
        return DownloadResult(destination, destination.stat().st_size, self.name)


class FakeMedia:
    metadata = MediaMetadata(8.0, 30.0, 1920, 1080, 16000, True, True)

    def probe(self, path: Path) -> MediaMetadata:
        return self.metadata

    def normalize(self, source: Path, destination: Path) -> MediaMetadata:
        destination.write_bytes(source.read_bytes())
        return self.metadata

    def extract_segment(
        self, source: Path, destination: Path, start_seconds: float, duration_seconds: float
    ) -> MediaMetadata:
        destination.write_bytes(source.read_bytes())
        return self.metadata


def make_test_config(tmp_path: Path) -> CadenceConfig:
    base = load_config("configs/test.yaml")
    return base.model_copy(
        update={"paths": base.paths.model_copy(update={"intake_root": tmp_path / "intake"})}
    )


def service(tmp_path: Path, downloader: FakeDownloader) -> DatasetIntakeService:
    return DatasetIntakeService(
        make_test_config(tmp_path),
        downloaders=DownloaderChain([downloader]),
        media=FakeMedia(),
    )


def test_single_and_batch_url_intake_are_deduplicated(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
    fixture.write_bytes(b"fixture")
    intake = service(tmp_path, FakeDownloader(fixture))
    source, created = intake.add_source(
        "HTTPS://Example.COM/launch.mp4#fragment", submitted_by="aven"
    )
    duplicate, duplicate_created = intake.add_source(
        "https://example.com/launch.mp4", submitted_by="user"
    )
    assert created is True
    assert duplicate_created is False
    assert duplicate.source_id == source.source_id
    assert source.rights_status == RightsStatus.UNVERIFIED
    assert source.eligible_for_training is False

    batch = tmp_path / "urls.txt"
    batch.write_text(
        "https://example.com/launch.mp4\nhttps://example.com/second.mp4\nnot-a-url\n",
        encoding="utf-8",
    )
    assert intake.add_batch(batch, submitted_by="aven") == {
        "added": 1,
        "duplicates": 1,
        "invalid": 1,
    }


def test_unsupported_source_is_tracked_not_crashed(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"fixture")
    intake = service(tmp_path, FakeDownloader(fixture, supported=False))
    source, _ = intake.add_source("https://example.com/page", submitted_by="aven")
    inspected = intake.inspect_source(source.source_id)
    assert inspected.inspection_status == InspectionStatus.UNSUPPORTED
    assert inspected.download_status == DownloadStatus.UNSUPPORTED
    assert "unsupported" in (inspected.error_state or "")


def test_approval_rights_and_training_eligibility_are_separate(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
    fixture.write_bytes(b"fixture")
    intake = service(tmp_path, FakeDownloader(fixture))
    source, _ = intake.add_source("https://example.com/a.mp4", submitted_by="aven")
    intake.inspect_source(source.source_id)
    approved = intake.set_source_approval(source.source_id, ApprovalStatus.APPROVED)
    assert approved.download_approval == ApprovalStatus.PENDING
    with pytest.raises(ValueError, match="rights"):
        intake.set_training_eligibility(source.source_id, True)
    intake.set_download_approval(source.source_id, ApprovalStatus.APPROVED)
    intake.set_rights(source.source_id, RightsStatus.USER_OWNED, license_notes="Owned by user")
    with pytest.raises(ValueError, match="normalized"):
        intake.set_training_eligibility(source.source_id, True)


def test_rejection_revokes_download_and_training_approval(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
    fixture.write_bytes(b"fixture")
    intake = service(tmp_path, FakeDownloader(fixture))
    source, _ = intake.add_source("https://example.com/a.mp4", submitted_by="aven")
    rejected = intake.set_source_approval(source.source_id, ApprovalStatus.REJECTED)
    assert rejected.download_approval == ApprovalStatus.REJECTED
    assert rejected.eligible_for_training is False


def test_download_failure_is_recorded_and_retry_succeeds(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
    fixture.write_bytes(b"fixture")
    downloader = FakeDownloader(fixture, fail=True)
    intake = service(tmp_path, downloader)
    source, _ = intake.add_source("https://example.com/a.mp4", submitted_by="aven")
    intake.inspect_source(source.source_id)
    intake.set_source_approval(source.source_id, ApprovalStatus.APPROVED)
    intake.set_download_approval(source.source_id, ApprovalStatus.APPROVED)
    failed = intake.download_source(source.source_id)
    assert failed.download_status == DownloadStatus.FAILED
    assert "simulated" in (failed.error_state or "")
    downloader.fail = False
    recovered = intake.download_source(source.source_id)
    assert recovered.download_status == DownloadStatus.NORMALIZED
    assert recovered.error_state is None


def test_duplicate_checksum_reuses_normalized_source(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
    fixture.write_bytes(b"same-content")
    intake = service(tmp_path, FakeDownloader(fixture))
    sources = []
    for name in ("a", "b"):
        source, _ = intake.add_source(f"https://example.com/{name}.mp4", submitted_by="aven")
        intake.inspect_source(source.source_id)
        intake.set_source_approval(source.source_id, ApprovalStatus.APPROVED)
        intake.set_download_approval(source.source_id, ApprovalStatus.APPROVED)
        sources.append(intake.download_source(source.source_id))
    assert sources[0].download_status == DownloadStatus.NORMALIZED
    assert sources[1].download_status == DownloadStatus.DUPLICATE
    assert sources[1].duplicate_of_source_id == sources[0].source_id
