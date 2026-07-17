from pathlib import Path
from uuid import uuid4

import pytest

from cadence.common.config import load_config
from cadence.ingestion.fixtures import generate_fixtures
from cadence.ingestion.manifest import load_manifest, write_manifest
from cadence.training.checkpoint import inspect_checkpoint
from cadence.training.runner import run_contrastive_training
from cadence.training.synthetic import run_synthetic_training


def test_synthetic_optimizer_checkpoint_and_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic.pt"
    config = load_config("configs/test.yaml")
    first = run_synthetic_training(config, checkpoint_path=checkpoint)
    assert first["optimizer_updated"] is True
    assert first["position"]["global_step"] == 1
    metadata = inspect_checkpoint(checkpoint)
    assert metadata["position"]["next_sample_offset"] == 2
    resumed = run_synthetic_training(
        config, checkpoint_path=checkpoint, resume_from=checkpoint
    )
    assert resumed["position"]["global_step"] == 2


def test_manifest_training_entry_point_uses_microbatch_buffer(tmp_path: Path) -> None:
    manifest = generate_fixtures(tmp_path / "fixtures")
    eligible_manifest = tmp_path / "eligible.jsonl"
    write_manifest(load_manifest(manifest)[:3], eligible_manifest)
    base = load_config("configs/test.yaml")
    config = base.model_copy(
        update={"paths": base.paths.model_copy(update={
            "manifest_path": eligible_manifest,
            "checkpoint_dir": tmp_path / "checkpoints",
        })}
    )
    result = run_contrastive_training(config)
    assert result["position"]["global_step"] == 1
    assert Path(result["checkpoint"]).is_file()


def test_manifest_training_reports_complete_validation_retrieval(
    tmp_path: Path,
) -> None:
    manifest = generate_fixtures(tmp_path / "fixtures")
    originals = load_manifest(manifest)[:2]
    entries = [
        entry.model_copy(update={"split": "train"}) for entry in originals
    ] + [
        entry.model_copy(
            update={
                "asset_id": uuid4(),
                "source_asset_id": uuid4(),
                "split": "validation",
            }
        )
        for entry in originals
    ]
    bounded_manifest = tmp_path / "bounded.jsonl"
    write_manifest(entries, bounded_manifest)
    base = load_config("configs/test.yaml")
    config = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "manifest_path": bounded_manifest,
                    "checkpoint_dir": tmp_path / "checkpoints",
                }
            ),
            "training": base.training.model_copy(
                update={"evaluation_interval_steps": 1}
            ),
        }
    )

    result = run_contrastive_training(config)

    assert result["validation"]["sample_count"] == 2
    assert result["validation"]["chance"] == 0.5
    assert "video_to_audio" in result["validation"]
    assert "audio_to_video" in result["validation"]


def test_manifest_training_rejects_checksum_drift_before_optimization(
    tmp_path: Path,
) -> None:
    manifest = generate_fixtures(tmp_path / "fixtures")
    eligible_manifest = tmp_path / "eligible.jsonl"
    entries = load_manifest(manifest)[:2]
    write_manifest(entries, eligible_manifest)
    assert entries[0].path is not None
    entries[0].path.write_bytes(entries[0].path.read_bytes() + b"corrupt")
    base = load_config("configs/test.yaml")
    config = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "manifest_path": eligible_manifest,
                    "checkpoint_dir": tmp_path / "checkpoints",
                }
            )
        }
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        run_contrastive_training(config)
