from pathlib import Path

import pytest

from cadence.cli import main
from cadence.ingestion.fixtures import _write_mp4


def test_pilot_cli_source_segment_build_report_flow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = tmp_path / "launch.mp4"
    pilot_dir = tmp_path / "pilot"
    _write_mp4(media, duration_s=8.0, fps=8, sample_rate=8000, event_s=4.0)

    assert (
        main(
            [
                "pilot",
                "source",
                "add",
                "--pilot-dir",
                str(pilot_dir),
                "--media-path",
                str(media),
                "--source-url",
                "https://example.com/launch-film",
                "--creator",
                "Example Studio",
                "--collection-method",
                "unit-test-local-file",
                "--license-status",
                "synthetic-generated",
            ]
        )
        == 0
    )
    assert "source_asset_id" in capsys.readouterr().out

    assert (
        main(
            [
                "pilot",
                "segments",
                "suggest",
                "--pilot-dir",
                str(pilot_dir),
                "--source",
                "all",
                "--min-duration",
                "4",
                "--max-duration",
                "6",
            ]
        )
        == 0
    )
    assert (pilot_dir / "segments.jsonl").read_text().splitlines()

    assert (
        main(
            [
                "pilot",
                "segments",
                "approve",
                "--pilot-dir",
                str(pilot_dir),
                "--clip",
                "all",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "pilot",
                "build",
                "pilot-launch-v0",
                "--pilot-dir",
                str(pilot_dir),
            ]
        )
        == 0
    )
    assert (pilot_dir / "datasets" / "pilot-launch-v0" / "manifest.jsonl").is_file()

    assert (
        main(
            [
                "pilot",
                "report",
                "pilot-launch-v0",
                "--pilot-dir",
                str(pilot_dir),
            ]
        )
        == 0
    )
    assert "approved_segments" in capsys.readouterr().out


def test_pilot_cli_accepts_batch_urls_as_candidate_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot_dir = tmp_path / "pilot"
    assert (
        main(
            [
                "pilot",
                "source",
                "add",
                "https://example.com/a",
                "https://example.com/b",
                "--pilot-dir",
                str(pilot_dir),
                "--submitted-by",
                "max",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "https://example.com/a" in output
    assert '"source_state": "candidate"' in output
    assert '"eligible_for_training": false' in output


def test_pilot_cli_segments_suggest_skips_url_only_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = tmp_path / "launch.mp4"
    pilot_dir = tmp_path / "pilot"
    _write_mp4(media, duration_s=8.0, fps=8, sample_rate=8000, event_s=4.0)
    assert (
        main(
            [
                "pilot",
                "source",
                "add",
                "https://example.com/url-only",
                "--pilot-dir",
                str(pilot_dir),
                "--submitted-by",
                "max",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "pilot",
                "source",
                "add",
                "--pilot-dir",
                str(pilot_dir),
                "--media-path",
                str(media),
                "--source-url",
                "https://fixtures.cadence.invalid/local",
                "--collection-method",
                "unit-test-local-file",
                "--license-status",
                "synthetic-generated",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "pilot",
                "segments",
                "suggest",
                "--pilot-dir",
                str(pilot_dir),
                "--source",
                "all",
            ]
        )
        == 0
    )
    assert "launch_video_candidate" in capsys.readouterr().out
