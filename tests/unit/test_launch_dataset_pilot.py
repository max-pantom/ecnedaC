from pathlib import Path

from cadence.ingestion.fixtures import _write_mp4
from cadence.ingestion.dataset_pilot import (
    approve_segments,
    build_pilot_manifest,
    inspect_source,
    load_report,
    suggest_segments,
    write_source_record,
)
from cadence.ingestion.manifest import ManifestEntry


def test_unverified_launch_clip_is_quarantined_by_default(tmp_path: Path) -> None:
    media = tmp_path / "launch.mp4"
    _write_mp4(media, duration_s=6.0, fps=8, sample_rate=8000, event_s=3.0)
    source = write_source_record(
        tmp_path / "pilot",
        media_path=media,
        source_url="https://example.com/launch-film",
        creator="Example Studio",
        collection_method="unit-test-local-file",
        license_status="unverified-research-quarantine",
    )

    inspected = inspect_source(source)
    assert inspected.has_video is True
    assert inspected.has_audio is True
    assert inspected.domain == "launch-video-sound-design"
    assert inspected.eligible_for_contrastive is False


def test_segment_suggestions_build_reviewable_manifest_and_report(tmp_path: Path) -> None:
    media = tmp_path / "launch.mp4"
    _write_mp4(media, duration_s=8.0, fps=8, sample_rate=8000, event_s=4.0)
    pilot_dir = tmp_path / "pilot"
    source = write_source_record(
        pilot_dir,
        media_path=media,
        source_url="https://example.com/launch-film",
        creator="Example Studio",
        collection_method="unit-test-local-file",
        license_status="synthetic-generated",
    )

    candidates = suggest_segments(pilot_dir, source.source_asset_id, min_duration_s=4.0, max_duration_s=6.0)
    assert candidates
    assert candidates[0].review_status == "candidate"
    assert candidates[0].start_s >= 0
    assert 4.0 <= candidates[0].duration_s <= 6.0
    assert candidates[0].motion_intensity >= 0
    assert candidates[0].audio_activity >= 0

    approve_segments(pilot_dir, [candidates[0].clip_asset_id])
    manifest_path = build_pilot_manifest(pilot_dir, dataset_id="pilot-launch-v0")
    entries = [ManifestEntry.model_validate_json(line) for line in manifest_path.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0].domain == "launch-video-sound-design"
    assert entries[0].clip_start_s == candidates[0].start_s
    assert entries[0].clip_end_s == candidates[0].end_s
    assert entries[0].eligible_for_contrastive is True

    report = load_report(pilot_dir, dataset_id="pilot-launch-v0")
    assert report["source_videos"] == 1
    assert report["candidate_segments"] >= 1
    assert report["approved_segments"] == 1
    assert report["missing_modality_count"] == 0
    assert report["total_duration_s"] == entries[0].duration_s
