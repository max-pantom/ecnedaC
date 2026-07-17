from pathlib import Path

import pytest
from pydantic import ValidationError

from cadence.common.config import load_config


def test_local_config_loads_and_resolves_paths() -> None:
    config = load_config("configs/local.yaml")
    assert config.runtime.device == "cpu"
    assert config.paths.checkpoint_dir.is_absolute()
    assert config.paths.checkpoint_dir == Path("artifacts/checkpoints").resolve()


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CADENCE_RUNTIME__SEED", "99")
    assert load_config("configs/local.yaml").runtime.seed == 99


@pytest.mark.parametrize(
    "key",
    [
        "CADENCE_REVIEW_ADMIN_SECRET",
        "CADENCE_REVIEW_SECURE_DEPLOYMENT",
        "CADENCE_REVIEW_TUNNEL_BASIC_USERNAME",
        "CADENCE_REVIEW_TUNNEL_BASIC_PASSWORD",
        "CADENCE_VPS_HOST",
        "CADENCE_VAST_INSTANCE_ID",
        "RUNPOD_API_KEY",
    ],
)
def test_runtime_only_environment_values_are_not_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    monkeypatch.setenv(key, "runtime-only-value")
    assert load_config("configs/local.yaml").runtime.profile == "local"


def test_malformed_environment_override_path_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CADENCE_RUNTIME____SEED", "99")
    with pytest.raises(ValueError, match="invalid environment override path"):
        load_config("configs/local.yaml")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("CADENCE_RUNTIME__DEVICE", "cuda"),
        ("CADENCE_RUNTIME__MAX_SAMPLES", "5"),
        ("CADENCE_RUNTIME__EPOCHS", "2"),
        ("CADENCE_RUNTIME__MICROBATCH_SIZE", "2"),
        ("CADENCE_RUNTIME__NUM_WORKERS", "2"),
        ("CADENCE_DATA__CLIP_SECONDS", "2.1"),
    ],
)
def test_local_safety_limits_reject_unsafe_values(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError, match="unsafe local configuration"):
        load_config("configs/local.yaml")


def test_unknown_configuration_field_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CADENCE_RUNTIME__MAGIC", "true")
    with pytest.raises(ValidationError):
        load_config("configs/local.yaml")


def test_vps_dataset_limits_are_hard_defaults() -> None:
    config = load_config("configs/vps.yaml")
    assert config.runtime.num_workers == 1
    assert config.paths.intake_root == Path("/srv/cadence/private")
    assert config.dataset_intake.maximum_working_storage_gb == 20.0
    assert config.dataset_intake.minimum_free_disk_gb == 15.0


def test_vps_intake_root_inside_repository_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CADENCE_PATHS__INTAKE_ROOT", "data/intake")
    with pytest.raises(ValueError, match="VPS intake_root must be outside the Git worktree"):
        load_config("configs/vps.yaml")


def test_test_profile_may_use_synthetic_intake_root_inside_repository() -> None:
    config = load_config("configs/test.yaml")
    assert config.paths.intake_root == Path("data/intake").resolve()


def test_gpu_profile_targets_bounded_runpod_a5000() -> None:
    remote = load_config("configs/gpu-24gb.yaml").remote

    assert remote.provider == "runpod"
    assert remote.requested_hardware == "NVIDIA RTX A5000 24GB"
    assert remote.runpod_gpu_type_id == "NVIDIA RTX A5000"
    assert remote.runpod_gpu_count == 1
    assert remote.maximum_budget_usd == 5
    assert remote.maximum_runtime_minutes == 240
    assert remote.synthetic_smoke_maximum_budget_usd == 1
    assert remote.synthetic_smoke_maximum_runtime_minutes == 30


def test_runpod_runtime_and_price_must_fit_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CADENCE_REMOTE__MAXIMUM_HOURLY_PRICE_USD", "2")

    with pytest.raises(ValidationError, match="exceed the first-run budget cap"):
        load_config("configs/gpu-24gb.yaml")


def test_first_run_final_step_must_include_full_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CADENCE_TRAINING__EVALUATION_INTERVAL_STEPS", "16")

    with pytest.raises(ValidationError, match="evaluate at the final optimizer step"):
        load_config("configs/first-run-v0.1.0.yaml")
