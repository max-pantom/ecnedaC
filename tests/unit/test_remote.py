import pytest

from cadence.common.config import load_config
from cadence.remote.job import RemoteJob, remote_command, run_remote_action


def test_remote_job_rejects_non_commit() -> None:
    values = {
        "git_commit": "UNCOMMITTED",
        "python_version": "3.12",
        "dependency_group": "training-gpu",
        "dependency_index": "https://download.pytorch.org/whl/cu126",
        "dependency_lock_hash": "a" * 64,
        "configuration_hash": "b" * 64,
        "configuration": {},
        "dataset_manifest_uri": "s3://bucket/manifest",
        "checkpoint_destination": "s3://bucket/checkpoint",
        "random_seed": 1,
        "requested_hardware": "RTX 4090 24GB",
        "maximum_budget_usd": 1,
        "maximum_runtime_minutes": 1,
        "created_at": "2026-01-01T00:00:00Z",
    }
    with pytest.raises(ValueError):
        RemoteJob.model_validate(values)


def test_every_remote_action_is_dry_by_default() -> None:
    config = load_config("configs/vps.yaml")
    for action in (
        "bootstrap_vps", "doctor_vps", "submit_job", "sync_checkpoints",
        "fetch_results", "terminate_gpu",
    ):
        assert run_remote_action(action, config, execute=False).startswith("DRY RUN:")
        assert remote_command(action, config)


def test_execute_requires_configuration() -> None:
    with pytest.raises(ValueError, match="missing"):
        run_remote_action("bootstrap_vps", load_config("configs/vps.yaml"), execute=True)

