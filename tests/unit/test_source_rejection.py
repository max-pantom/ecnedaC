from pathlib import Path

from cadence.cli import main
from cadence.ingestion.dataset_pilot import load_source_queue


def test_pilot_cli_source_reject_records_reason_and_keeps_training_disabled(
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    assert (
        main(
            [
                "pilot",
                "source",
                "add",
                "https://example.com/not-launch-video",
                "--pilot-dir",
                str(pilot_dir),
                "--submitted-by",
                "max",
            ]
        )
        == 0
    )
    source = load_source_queue(pilot_dir)[0]
    assert (
        main(
            [
                "pilot",
                "source",
                "reject",
                "--pilot-dir",
                str(pilot_dir),
                "--source",
                str(source.source_asset_id),
                "--reason",
                "outside-launch-video-domain",
            ]
        )
        == 0
    )
    rejected = load_source_queue(pilot_dir)[0]
    assert rejected.source_state == "rejected"
    assert rejected.source_rejection_reason == "outside-launch-video-domain"
    assert rejected.eligible_for_training is False
    assert rejected.eligible_for_contrastive is False
