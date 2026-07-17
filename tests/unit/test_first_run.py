import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cadence.common.config import load_config
from cadence.training.first_run import (
    freeze_first_run,
    read_frozen_first_run,
    validate_frozen_first_run,
    write_frozen_first_run,
)

DATASET_HANDLE = "cad15-manifest-47924d80b058"


def test_exact_first_run_configuration_is_bounded() -> None:
    config = load_config("configs/first-run-v0.1.0.yaml")

    assert config.runtime.seed == 1337
    assert config.runtime.epochs == 20
    assert config.runtime.microbatch_size == 32
    assert config.runtime.contrastive_group_size == 32
    assert config.training.max_steps == 40
    assert config.training.precision == "amp-fp16"
    assert config.training.evaluation_interval_steps == 10
    assert config.first_run.expected_train_rows == 83
    assert config.first_run.expected_validation_rows == 22
    assert config.first_run.abort is not None
    assert config.first_run.abort.soft_stop_runtime_minutes == 210
    assert config.first_run.abort.hard_stop_runtime_minutes == 240


def test_first_run_linked_values_cannot_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CADENCE_TRAINING__MAX_STEPS", "41")
    with pytest.raises(ValidationError, match="invalid first-run freeze"):
        load_config("configs/first-run-v0.1.0.yaml")


def test_frozen_package_is_deterministic_sanitized_and_non_authorizing(
    tmp_path: Path,
) -> None:
    config = load_config("configs/first-run-v0.1.0.yaml")
    first = freeze_first_run(config, DATASET_HANDLE, require_clean=False)
    second = freeze_first_run(config, DATASET_HANDLE, require_clean=False)
    encoded = first.model_dump_json().lower()

    assert first == second
    assert first.launch_authorized is False
    assert first.private_locations_included is False
    assert first.credentials_included is False
    for forbidden in (
        "manifest_uri",
        "checkpoint_uri",
        "vps_host",
        "api_key",
        "s3://",
        "intake_root",
        "manifest_path",
    ):
        assert forbidden not in encoded

    output = tmp_path / "first-run.json"
    write_frozen_first_run(first, output)
    assert read_frozen_first_run(output) == first
    assert not output.with_suffix(".json.tmp").exists()
    assert json.loads(output.read_text())["launch_authorized"] is False


def test_frozen_package_validates_exact_inputs() -> None:
    config = load_config("configs/first-run-v0.1.0.yaml")
    package = freeze_first_run(config, DATASET_HANDLE, require_clean=False)

    result = validate_frozen_first_run(
        package,
        config,
        DATASET_HANDLE,
        require_clean=False,
    )

    assert result["compatible"] is True
    assert result["launch_authorized"] is False
    assert result["network_action"] is False


def test_frozen_package_rejects_dataset_or_configuration_drift() -> None:
    config = load_config("configs/first-run-v0.1.0.yaml")
    package = freeze_first_run(config, DATASET_HANDLE, require_clean=False)
    changed_training = config.training.model_copy(update={"learning_rate": 0.0002})
    changed_config = config.model_copy(update={"training": changed_training})

    with pytest.raises(ValueError, match="incompatible frozen first run"):
        validate_frozen_first_run(
            package,
            changed_config,
            DATASET_HANDLE,
            require_clean=False,
        )
    with pytest.raises(ValueError, match="incompatible frozen first run"):
        validate_frozen_first_run(
            package,
            config,
            "cad15-manifest-different",
            require_clean=False,
        )


def test_first_run_freeze_rejects_hidden_typed_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/first-run-v0.1.0.yaml")
    monkeypatch.setenv("CADENCE_TRAINING__LEARNING_RATE", "0.0002")

    with pytest.raises(ValueError, match="rejects typed environment overrides"):
        freeze_first_run(config, DATASET_HANDLE, require_clean=False)


def test_first_run_freeze_rejects_invalid_dataset_handle() -> None:
    config = load_config("configs/first-run-v0.1.0.yaml")

    with pytest.raises(ValidationError):
        freeze_first_run(config, "/srv/private/manifest.jsonl", require_clean=False)
