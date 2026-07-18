"""Typed, environment-overridable Cadence configuration."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    precision: Literal["fp32", "amp-fp16"] = "fp32"
    evaluation_interval_steps: int | None = Field(default=None, gt=0)


class FirstRunSuccessConfig(StrictModel):
    required_optimizer_steps: int = Field(gt=0)
    require_finite_loss_and_gradients: Literal[True] = True
    require_full_validation_retrieval: Literal[True] = True
    minimum_recall_at_1_over_chance: float = Field(default=1.0, ge=1.0)
    require_final_checkpoint: Literal[True] = True
    require_fresh_process_resume: Literal[True] = True


class FirstRunAbortConfig(StrictModel):
    maximum_oom_events: Literal[0] = 0
    maximum_decoder_errors: Literal[0] = 0
    maximum_checksum_mismatches: Literal[0] = 0
    abort_on_nonfinite_loss_or_gradient: Literal[True] = True
    abort_on_checkpoint_failure: Literal[True] = True
    abort_on_compatibility_mismatch: Literal[True] = True
    soft_stop_runtime_minutes: int = Field(gt=0)
    hard_stop_runtime_minutes: int = Field(gt=0)
    hard_stop_budget_usd: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_stop_order(self) -> FirstRunAbortConfig:
        if self.soft_stop_runtime_minutes >= self.hard_stop_runtime_minutes:
            raise ValueError("soft stop must occur before the hard runtime stop")
        return self


class FirstRunConfig(StrictModel):
    enabled: bool = False
    specification_version: Literal["0.1.0"] = "0.1.0"
    expected_train_rows: int = Field(default=0, ge=0)
    expected_validation_rows: int = Field(default=0, ge=0)
    expected_test_rows: int = Field(default=0, ge=0)
    success: FirstRunSuccessConfig | None = None
    abort: FirstRunAbortConfig | None = None


class PathsConfig(StrictModel):
    manifest_path: Path | None = None
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    report_dir: Path = Path("artifacts/reports")
    intake_root: Path = Path("data/intake")


class DatasetIntakeConfig(StrictModel):
    maximum_working_storage_gb: float = Field(default=20.0, gt=0)
    minimum_free_disk_gb: float = Field(default=15.0, ge=0)
    unknown_download_reservation_gb: float = Field(default=2.0, gt=0)
    segment_min_seconds: float = Field(default=4.0, gt=0)
    segment_max_seconds: float = Field(default=10.0, gt=0)
    segment_target_seconds: float = Field(default=6.0, gt=0)
    maximum_suggestions_per_source: int = Field(default=12, gt=0)
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"

    @model_validator(mode="after")
    def validate_segment_durations(self) -> DatasetIntakeConfig:
        if not self.segment_min_seconds <= self.segment_target_seconds <= self.segment_max_seconds:
            raise ValueError("segment target must be between segment minimum and maximum")
        return self


class RemoteConfig(StrictModel):
    provider: Literal["runpod", "vast"] = "runpod"
    artifact_transport: Literal["vps-ssh"] = "vps-ssh"
    dataset_snapshot_handle: str = Field(
        default="cadence-private-snapshot",
        pattern=r"^[a-z0-9][a-z0-9-]{7,127}$",
    )
    checkpoint_run_handle: str = Field(
        default="cadence-first-run",
        pattern=r"^[a-z0-9][a-z0-9-]{7,127}$",
    )
    checkpoint_retention_count: int = Field(default=4, ge=1, le=4)
    requested_hardware: str = "NVIDIA RTX A5000 24GB"
    dependency_group: Literal["training-gpu"] = "training-gpu"
    python_version: str = "3.12"
    maximum_budget_usd: float = Field(default=2.0, gt=0, le=2.0)
    maximum_runtime_minutes: int = Field(default=240, gt=0, le=240)
    synthetic_smoke_maximum_budget_usd: float = Field(default=1.0, gt=0, le=1.0)
    synthetic_smoke_maximum_runtime_minutes: int = Field(default=30, gt=0, le=30)
    maximum_hourly_price_usd: float = Field(default=0.30, gt=0)
    runpod_gpu_type_id: Literal["NVIDIA RTX A5000"] = "NVIDIA RTX A5000"
    runpod_gpu_count: Literal[1] = 1
    runpod_cloud_type: Literal["COMMUNITY", "SECURE"] = "COMMUNITY"
    runpod_image_name: str = Field(
        default="pytorch/pytorch:2.11.0-cuda12.6-cudnn9-devel",
        min_length=1,
    )
    runpod_container_disk_gb: int = Field(default=50, ge=20, le=100)
    runpod_volume_gb: int = Field(default=0, ge=0, le=100)
    runpod_pod_name: str = Field(
        default="cadence-bounded-a5000",
        pattern=r"^[a-z0-9][a-z0-9-]{2,62}$",
    )
    vps_host: str | None = None
    vast_instance_id: str | None = None

    @model_validator(mode="after")
    def validate_provider_limits(self) -> RemoteConfig:
        if self.provider == "runpod" and self.requested_hardware != "NVIDIA RTX A5000 24GB":
            raise ValueError("RunPod readiness is restricted to NVIDIA RTX A5000 24GB")
        maximum_compute_cost = (
            self.maximum_hourly_price_usd * self.maximum_runtime_minutes / 60
        )
        if maximum_compute_cost > self.maximum_budget_usd:
            raise ValueError("maximum runtime and hourly price exceed the first-run budget cap")
        smoke_compute_cost = (
            self.maximum_hourly_price_usd
            * self.synthetic_smoke_maximum_runtime_minutes
            / 60
        )
        if smoke_compute_cost > self.synthetic_smoke_maximum_budget_usd:
            raise ValueError("smoke runtime and hourly price exceed the smoke budget cap")
        return self


class VpsOperationsConfig(StrictModel):
    backup_retention_count: int = Field(default=7, ge=1, le=30)
    review_health_url: str = "http://127.0.0.1:8787/healthz"

    @field_validator("review_health_url")
    @classmethod
    def require_loopback_health_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query:
            raise ValueError("review health URL must be plain loopback HTTP without credentials")
        if parsed.path != "/healthz" or parsed.fragment:
            raise ValueError("review health URL must target /healthz")
        host = parsed.hostname
        if host is None:
            raise ValueError("review health URL requires a host")
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if not loopback:
            raise ValueError("review health URL must use a loopback host")
        return value


class CadenceConfig(StrictModel):
    runtime: RuntimeConfig
    data: DataConfig
    encoders: EncoderConfig
    training: TrainingConfig
    paths: PathsConfig
    dataset_intake: DatasetIntakeConfig
    remote: RemoteConfig
    first_run: FirstRunConfig = Field(default_factory=FirstRunConfig)
    vps_operations: VpsOperationsConfig = Field(default_factory=VpsOperationsConfig)

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

    @model_validator(mode="after")
    def enforce_first_run_freeze(self) -> CadenceConfig:
        spec = self.first_run
        if not spec.enabled:
            return self
        errors: list[str] = []
        if self.runtime.profile != "gpu-24gb" or self.runtime.device != "cuda":
            errors.append("first run requires the gpu-24gb CUDA profile")
        if self.training.precision != "amp-fp16":
            errors.append("first run precision must be amp-fp16")
        if self.training.evaluation_interval_steps is None:
            errors.append("first run requires an evaluation interval")
        elif self.training.max_steps % self.training.evaluation_interval_steps != 0:
            errors.append("first run must evaluate at the final optimizer step")
        if spec.expected_train_rows < self.runtime.contrastive_group_size:
            errors.append("first run needs at least one complete contrastive group")
        if spec.expected_validation_rows < 2:
            errors.append("first run needs at least two validation rows")
        if spec.success is None or spec.abort is None:
            errors.append("first run requires success and abort rules")
        else:
            expected_optimizer_steps = self.runtime.epochs * (
                spec.expected_train_rows // self.runtime.contrastive_group_size
            )
            if spec.success.required_optimizer_steps != expected_optimizer_steps:
                errors.append(
                    "required_optimizer_steps must match epochs and complete groups"
                )
            if self.training.max_steps != spec.success.required_optimizer_steps:
                errors.append("max_steps must equal required_optimizer_steps")
            if self.remote.maximum_runtime_minutes != spec.abort.hard_stop_runtime_minutes:
                errors.append("remote runtime must equal the hard runtime stop")
            if self.remote.maximum_budget_usd != spec.abort.hard_stop_budget_usd:
                errors.append("remote budget must equal the hard budget stop")
        if errors:
            raise ValueError("invalid first-run freeze: " + "; ".join(errors))
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
        override = key[len(prefix) :]
        # Typed configuration overrides always use a double underscore between
        # model fields. Single-level CADENCE_* names belong to runtime consumers
        # such as review authentication, tunnel controls, and remote credentials.
        if "__" not in override:
            continue
        parts = override.lower().split("__")
        if any(not part for part in parts):
            raise ValueError(f"invalid environment override path: {key}")
        cursor: dict[str, Any] = data
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"environment override conflicts at {part}")
            cursor = child
        cursor[parts[-1]] = _coerce_env_value(raw)


def _resolve_paths(config: CadenceConfig, repo_root: Path) -> CadenceConfig:
    values = config.model_dump()
    for key in ("manifest_path", "checkpoint_dir", "report_dir", "intake_root"):
        value = values["paths"].get(key)
        if value is not None:
            path = Path(value).expanduser()
            values["paths"][key] = path if path.is_absolute() else (repo_root / path).resolve()
    return CadenceConfig.model_validate(values)


def _validate_private_runtime_paths(config: CadenceConfig, repo_root: Path) -> None:
    if (
        config.runtime.profile == "vps"
        and config.paths.intake_root.is_relative_to(repo_root.resolve())
    ):
        raise ValueError(
            "VPS intake_root must be outside the Git worktree; "
            "use /srv/cadence/private or another private absolute path"
        )


def load_config(path: str | Path, *, repo_root: str | Path | None = None) -> CadenceConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    _apply_env_overrides(raw)
    config = CadenceConfig.model_validate(raw)
    root = Path(repo_root).resolve() if repo_root else config_path.parent.parent
    resolved = _resolve_paths(config, root)
    _validate_private_runtime_paths(resolved, root)
    return resolved
