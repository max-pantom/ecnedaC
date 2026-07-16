from pathlib import Path

from cadence.ingestion.dataset_pilot import (
    SourceRecord,
    approve_sources,
    load_source_queue,
    write_candidate_sources,
)
from cadence.cli import main


def test_url_candidate_queue_records_submitter_and_rights_state(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    records = write_candidate_sources(
        pilot_dir,
        ["https://example.com/launch-video"],
        submitted_by="max",
        collection_method="user-submitted-url",
    )

    assert len(records) == 1
    source = records[0]
    assert source.source_url == "https://example.com/launch-video"
    assert source.submitted_by == "max"
    assert source.source_state == "candidate"
    assert source.rights_status == "unverified"
    assert source.eligible_for_training is False
    assert source.eligible_for_contrastive is False
    assert source.download_status == "not_downloaded"


def test_batch_url_intake_deduplicates_and_approval_does_not_grant_training_rights(
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    write_candidate_sources(
        pilot_dir,
        [
            "https://example.com/a",
            "https://example.com/a",
            "https://example.com/b",
        ],
        submitted_by="max",
        collection_method="user-submitted-url",
    )

    queue = load_source_queue(pilot_dir)
    assert len(queue) == 2
    approved = approve_sources(pilot_dir, [queue[0].source_asset_id])
    first = next(source for source in approved if source.source_asset_id == queue[0].source_asset_id)
    assert first.source_state == "approved_source"
    assert first.rights_status == "unverified"
    assert first.eligible_for_training is False


def test_source_record_accepts_legacy_and_new_rights_fields(tmp_path: Path) -> None:
    source = SourceRecord(
        source_asset_id="66b2f650-36bb-5745-881c-3623ea5f190b",
        source_url="https://example.com/a",
        media_path=None,
        storage_uri=None,
        creator=None,
        publisher=None,
        submitted_by="max",
        collection_method="unit-test",
        license_status="unverified-research-quarantine",
        rights_status="unverified",
        source_state="candidate",
        source_rejection_reason=None,
        download_status="not_downloaded",
        domain="launch-video-sound-design",
        checksum_sha256="",
        duration_s=0.0,
        fps=0.0,
        width=0,
        height=0,
        audio_sample_rate=0,
        has_video=False,
        has_audio=False,
        eligible_for_contrastive=False,
        eligible_for_training=False,
        added_at="2026-07-16T00:00:00+00:00",
    )
    assert source.rights_status == "unverified"


def test_dataset_cli_source_approve_preserves_unverified_training_exclusion(
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    assert main([
        "dataset", "source", "add",
        "https://example.com/launch-video",
        "--pilot-dir", str(pilot_dir),
        "--submitted-by", "max",
    ]) == 0
    queue = load_source_queue(pilot_dir)
    assert main([
        "dataset", "source", "approve",
        "--pilot-dir", str(pilot_dir),
        "--source", str(queue[0].source_asset_id),
    ]) == 0
    approved = load_source_queue(pilot_dir)[0]
    assert approved.source_state == "approved_source"
    assert approved.rights_status == "unverified"
    assert approved.eligible_for_training is False
