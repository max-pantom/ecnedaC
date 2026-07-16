from pathlib import Path

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
