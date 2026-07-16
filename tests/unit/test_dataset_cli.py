from pathlib import Path

import pytest

from cadence.cli import main
from cadence.ingestion.fixtures import _write_mp4


def test_dataset_cli_source_segment_build_report_flow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = tmp_path / "launch.mp4"
    pilot_dir = tmp_path / "pilot"
    _write_mp4(media, duration_s=8.0, fps=8, sample_rate=8000, event_s=4.0)

    assert main([
        "dataset", "source", "add",
        "--pilot-dir", str(pilot_dir),
        "--media-path", str(media),
        "--source-url", "https://example.com/launch-film",
        "--creator", "Example Studio",
        "--collection-method", "unit-test-local-file",
        "--license-status", "synthetic-generated",
    ]) == 0
    output = capsys.readouterr().out
    assert "source_asset_id" in output

    assert main([
        "dataset", "segments", "suggest",
        "--pilot-dir", str(pilot_dir),
        "--source", "all",
        "--min-duration", "4",
        "--max-duration", "6",
    ]) == 0
    candidates = (pilot_dir / "segments.jsonl").read_text().splitlines()
    assert candidates

    assert main([
        "dataset", "segments", "approve",
        "--pilot-dir", str(pilot_dir),
        "--clip", "all",
    ]) == 0
    assert main([
        "dataset", "build", "pilot-launch-v0",
        "--pilot-dir", str(pilot_dir),
    ]) == 0
    assert (pilot_dir / "datasets" / "pilot-launch-v0" / "manifest.jsonl").is_file()

    assert main([
        "dataset", "report", "pilot-launch-v0",
        "--pilot-dir", str(pilot_dir),
    ]) == 0
    report_output = capsys.readouterr().out
    assert "approved_segments" in report_output


def test_dataset_cli_accepts_batch_urls_as_candidate_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot_dir = tmp_path / "pilot"
    assert main([
        "dataset", "source", "add",
        "https://example.com/a",
        "https://example.com/b",
        "--pilot-dir", str(pilot_dir),
        "--submitted-by", "max",
    ]) == 0
    output = capsys.readouterr().out
    assert "https://example.com/a" in output
    assert "source_state" in output
    assert "candidate" in output
    assert "eligible_for_training" in output


def test_dataset_cli_segments_suggest_skips_url_only_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = tmp_path / "launch.mp4"
    pilot_dir = tmp_path / "pilot"
    _write_mp4(media, duration_s=8.0, fps=8, sample_rate=8000, event_s=4.0)
    assert main([
        "dataset", "source", "add",
        "https://example.com/url-only",
        "--pilot-dir", str(pilot_dir),
        "--submitted-by", "max",
    ]) == 0
    assert main([
        "dataset", "source", "add",
        "--pilot-dir", str(pilot_dir),
        "--media-path", str(media),
        "--source-url", "https://fixtures.cadence.invalid/local",
        "--collection-method", "unit-test-local-file",
        "--license-status", "synthetic-generated",
    ]) == 0
    assert main([
        "dataset", "segments", "suggest",
        "--pilot-dir", str(pilot_dir),
        "--source", "all",
    ]) == 0
    output = capsys.readouterr().out
    assert "launch_video_candidate" in output
