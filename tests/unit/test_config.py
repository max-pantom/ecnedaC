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

