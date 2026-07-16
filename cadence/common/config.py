"""Typed, environment-overridable Cadence configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfig(StrictModel):
    profile: Literal["local", "test", "vps", "gpu-24gb"]
    device: Literal["cpu", "cuda"]
    seed: int = 1337
    allow_unsafe_local: bool = False
    max_samples: int | None = None
    epochs: int = 1
    num_workers: int = 0
    microbatch_size: int = 1
    contrastive_group_size: int = 2


class DataConfig(StrictModel):
    clip_seconds: float = Field(gt=0)
    num_frames: int = Field(gt=0)
    frame_size: int = Field(gt=0)
    sample_rate: int = Field(gt=0)
    n_mels: int = Field(gt=1)
    n_fft: int = Field(gt=1)
    hop_length: int = Field(gt=0)
    split: Literal["train", "validation", "test"] = "train"


class EncoderConfig(StrictModel):
    video_base_channels: int = Field(gt=0)
    audio_base_channels: int = Field(gt=0)
    embed_dim: int = Field(gt=0)
    projection_dim: int = Field(gt=0)
    sequence_length: int = Field(gt=0)


class TrainingConfig(StrictModel):
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    temperature: float = Field(gt=0)
    checkpoint_interval_seconds: int = Field(gt=0, le=600)
    max_steps: int = Field(gt=0)


class PathsConfig(StrictModel):
    manifest_path: Path | None = None
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    report_dir: Path = Path("artifacts/reports")


class RemoteConfig(StrictModel):
    manifest_uri: str = "s3://cadence-placeholder/manifests/train.jsonl"
    checkpoint_uri: str = "s3://cadence-placeholder/checkpoints/"
    requested_hardware: str = "RTX 4090 24GB"
    dependency_group: Literal["training-gpu"] = "training-gpu"
    python_version: str = "3.12"
    maximum_budget_usd: float = Field(default=25.0, gt=0)
    maximum_runtime_minutes: int = Field(default=360, gt=0)
    vps_host: str | None = None
    vast_instance_id: str | None = None


class CadenceConfig(StrictModel):
    runtime: RuntimeConfig
    data: DataConfig
    encoders: EncoderConfig
    training: TrainingConfig
    paths: PathsConfig
    remote: RemoteConfig

    @model_validator(mode="after")
    def enforce_profile_safety(self) -> CadenceConfig:
        if self.runtime.profile not in {"local", "test"} or self.runtime.allow_unsafe_local:
            return self
        errors: list[str] = []
        if self.runtime.device != "cpu":
            errors.append("device must be cpu")
        if self.runtime.max_samples is None or self.runtime.max_samples > 4:
            errors.append("max_samples must be set and <= 4")
        if self.data.clip_seconds > 2:
            errors.append("clip_seconds must be <= 2")
        if self.runtime.epochs > 1:
            errors.append("epochs must be <= 1")
        if self.runtime.microbatch_size > 1:
            errors.append("microbatch_size must be <= 1")
        if self.runtime.num_workers > 1:
            errors.append("num_workers must be <= 1")
        if self.paths.manifest_path and _is_remote(str(self.paths.manifest_path)):
            errors.append("remote manifest paths are disabled")
        if errors:
            raise ValueError("unsafe local configuration: " + "; ".join(errors))
        return self


def _is_remote(value: str) -> bool:
    return "://" in value and not value.startswith("file://")


def _coerce_env_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_env_overrides(data: dict[str, Any]) -> None:
    prefix = "CADENCE_"
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        cursor: dict[str, Any] = data
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"environment override conflicts at {part}")
            cursor = child
        cursor[parts[-1]] = _coerce_env_value(raw)


def _resolve_paths(config: CadenceConfig, repo_root: Path) -> CadenceConfig:
    values = config.model_dump()
    for key in ("manifest_path", "checkpoint_dir", "report_dir"):
        value = values["paths"].get(key)
        if value is not None:
            path = Path(value).expanduser()
            values["paths"][key] = path if path.is_absolute() else (repo_root / path).resolve()
    return CadenceConfig.model_validate(values)


def load_config(path: str | Path, *, repo_root: str | Path | None = None) -> CadenceConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    _apply_env_overrides(raw)
    config = CadenceConfig.model_validate(raw)
    root = Path(repo_root).resolve() if repo_root else config_path.parent.parent
    return _resolve_paths(config, root)

