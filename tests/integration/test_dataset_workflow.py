import json
from pathlib import Path
from uuid import UUID

import av

from cadence.common.config import CadenceConfig, load_config
from cadence.dataset.downloaders import DownloaderChain, DownloadResult, SourceInspection
from cadence.dataset.media import MediaMetadata
from cadence.dataset.records import ApprovalStatus, RightsStatus, SourceRecord
from cadence.dataset.service import DatasetIntakeService
from cadence.ingestion.fixtures import generate_fixtures
from cadence.ingestion.manifest import load_manifest


def probe_with_av(path: Path) -> MediaMetadata:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        duration = float(container.duration or 0) / av.time_base
        return MediaMetadata(
            duration,
            float(video.average_rate or 8),
            video.width,
            video.height,
            int(audio.rate or 8000),
            True,
            True,
        )


class FixtureDownloader:
    name = "fixture-copy"

    def __init__(self, fixture: Path) -> None:
        self.fixture = fixture

    def inspect(self, url: str) -> SourceInspection:
        metadata = probe_with_av(self.fixture)
        return SourceInspection(
            True,
            self.name,
            title="Generated launch fixture",
            publisher_or_creator="Cadence",
            platform="fixtures.cadence.invalid",
            duration_seconds=metadata.duration_seconds,
            content_length_bytes=self.fixture.stat().st_size,
        )

    def download(self, source: SourceRecord, destination: Path) -> DownloadResult:
        destination.write_bytes(self.fixture.read_bytes())
        return DownloadResult(destination, destination.stat().st_size, self.name)


class FixtureMedia:
    def probe(self, path: Path) -> MediaMetadata:
        return probe_with_av(path)

    def normalize(self, source: Path, destination: Path) -> MediaMetadata:
        destination.write_bytes(source.read_bytes())
        return self.probe(destination)

    def extract_segment(
        self, source: Path, destination: Path, start_seconds: float, duration_seconds: float
    ) -> MediaMetadata:
        destination.write_bytes(source.read_bytes())
        return self.probe(destination)


def workflow_config(tmp_path: Path) -> CadenceConfig:
    base = load_config("configs/test.yaml")
    return base.model_copy(
        update={"paths": base.paths.model_copy(update={"intake_root": tmp_path / "intake"})}
    )


def test_segment_approval_build_manifest_and_report(tmp_path: Path) -> None:
    fixture_manifest = generate_fixtures(tmp_path / "fixtures")
    fixture = load_manifest(fixture_manifest)[0].path
    assert fixture is not None
    service = DatasetIntakeService(
        workflow_config(tmp_path),
        downloaders=DownloaderChain([FixtureDownloader(fixture)]),
        media=FixtureMedia(),
    )
    source, _ = service.add_source(
        "https://fixtures.cadence.invalid/launch.mp4", submitted_by="aven"
    )
    service.inspect_source(source.source_id)
    service.set_source_approval(source.source_id, ApprovalStatus.APPROVED)
    service.set_download_approval(source.source_id, ApprovalStatus.APPROVED)
    service.set_rights(
        source.source_id, RightsStatus.VERIFIED_PERMITTED, license_notes="Synthetic fixture"
    )
    downloaded = service.download_source(source.source_id)
    assert downloaded.eligible_for_training is False
    service.set_training_eligibility(source.source_id, True)
    segments = service.suggest_source_segments(source.source_id)
    assert segments
    assert all(segment.checksum_after_extraction for segment in segments)
    approved = service.set_segment_approval(segments[0].segment_id, ApprovalStatus.APPROVED)
    assert approved.approval_status == ApprovalStatus.APPROVED
    dataset = service.build_dataset("launch-pilot")
    manifest = load_manifest(dataset.manifest_path)
    assert len(manifest) == 1
    assert manifest[0].eligible_for_contrastive is True
    assert manifest[0].source_asset_id == source.source_id
    report = service.dataset_report("launch-pilot")
    assert report["version"] == 1
    assert report["rights_counts"] == {"verified_permitted": 1}
    second = service.build_dataset("launch-pilot")
    assert second.version == 2
    assert service.dataset_report("launch-pilot")["version"] == 2


def test_unverified_source_cannot_enter_dataset(tmp_path: Path) -> None:
    fixture_manifest = generate_fixtures(tmp_path / "fixtures")
    fixture = load_manifest(fixture_manifest)[0].path
    assert fixture is not None
    service = DatasetIntakeService(
        workflow_config(tmp_path),
        downloaders=DownloaderChain([FixtureDownloader(fixture)]),
        media=FixtureMedia(),
    )
    source, _ = service.add_source(
        "https://fixtures.cadence.invalid/unverified.mp4", submitted_by="aven"
    )
    service.inspect_source(source.source_id)
    service.set_source_approval(source.source_id, ApprovalStatus.APPROVED)
    service.set_download_approval(source.source_id, ApprovalStatus.APPROVED)
    service.download_source(source.source_id)
    segments = service.suggest_source_segments(source.source_id)
    service.set_segment_approval(segments[0].segment_id, ApprovalStatus.APPROVED)
    with __import__("pytest").raises(ValueError, match="no approved segments"):
        service.build_dataset("blocked-pilot")


def test_legacy_import_preserves_identity_but_quarantines_decisions(tmp_path: Path) -> None:
    fixture_manifest = generate_fixtures(tmp_path / "fixtures")
    fixture = load_manifest(fixture_manifest)[0].path
    assert fixture is not None
    service = DatasetIntakeService(
        workflow_config(tmp_path),
        downloaders=DownloaderChain([FixtureDownloader(fixture)]),
        media=FixtureMedia(),
    )
    legacy_id = UUID("11111111-1111-1111-1111-111111111111")
    pilot_dir = tmp_path / "legacy-pilot"
    pilot_dir.mkdir()
    (pilot_dir / "sources.jsonl").write_text(
        json.dumps(
            {
                "source_asset_id": str(legacy_id),
                "source_url": "https://example.com/legacy-launch",
                "submitted_by": "aven",
                "collection_method": "legacy-user-submitted-url",
                "license_status": "licensed",
                "rights_status": "licensed",
                "source_state": "approved_source",
                "eligible_for_training": True,
                "duration_s": 8.0,
                "creator": "Example Studio",
            }
        )
        + "\n{malformed}\n",
        encoding="utf-8",
    )

    preview = service.import_legacy_pilot(
        pilot_dir,
        submitted_by="migration-operator",
    )
    assert preview["executed"] is False
    assert preview["would_add"] == 1
    assert preview["invalid"] == 1
    assert service.list_sources() == []

    imported = service.import_legacy_pilot(
        pilot_dir,
        submitted_by="migration-operator",
        execute=True,
    )
    assert imported["added"] == 1
    source = service.registry.get_source(legacy_id)
    assert source.publisher_or_creator == "Example Studio"
    assert source.collection_method == "legacy-pilot:legacy-user-submitted-url"
    assert source.rights_status == RightsStatus.UNVERIFIED
    assert source.source_approval == ApprovalStatus.PENDING
    assert source.download_approval == ApprovalStatus.PENDING
    assert source.eligible_for_training is False
