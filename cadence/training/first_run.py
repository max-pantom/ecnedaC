"""Immutable, sanitized first real-training run specification."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cadence.common.config import CadenceConfig
from cadence.common.repro import file_hash, git_commit, stable_hash


class FrozenFirstRun(BaseModel):
    """Public-safe binding for an approved private dataset snapshot and exact code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dependency_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_snapshot_handle: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{7,127}$")
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: dict[str, Any]
    private_locations_included: Literal[False] = False
    credentials_included: Literal[False] = False
    launch_authorized: Literal[False] = False


def bounded_first_run_snapshot(config: CadenceConfig) -> dict[str, Any]:
    """Return only training controls; omit paths, URIs, hosts, and credentials."""

    return {
        "runtime": config.runtime.model_dump(mode="json"),
        "data": config.data.model_dump(mode="json"),
        "encoders": config.encoders.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "remote": {
            "provider": config.remote.provider,
            "artifact_transport": config.remote.artifact_transport,
            "dataset_snapshot_handle": config.remote.dataset_snapshot_handle,
            "checkpoint_run_handle": config.remote.checkpoint_run_handle,
            "checkpoint_retention_count": config.remote.checkpoint_retention_count,
            "requested_hardware": config.remote.requested_hardware,
            "dependency_group": config.remote.dependency_group,
            "python_version": config.remote.python_version,
            "maximum_budget_usd": config.remote.maximum_budget_usd,
            "maximum_runtime_minutes": config.remote.maximum_runtime_minutes,
            "maximum_hourly_price_usd": config.remote.maximum_hourly_price_usd,
            "runpod_gpu_type_id": config.remote.runpod_gpu_type_id,
            "runpod_gpu_count": config.remote.runpod_gpu_count,
            "runpod_cloud_type": config.remote.runpod_cloud_type,
            "runpod_image_name": config.remote.runpod_image_name,
            "runpod_container_disk_gb": config.remote.runpod_container_disk_gb,
            "runpod_volume_gb": config.remote.runpod_volume_gb,
        },
        "first_run": config.first_run.model_dump(mode="json"),
    }


def _require_clean_repository(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout.strip():
        raise ValueError("first-run freeze requires a clean Git worktree")


def freeze_first_run(
    config: CadenceConfig,
    dataset_snapshot_handle: str,
    *,
    repo_root: str | Path = ".",
    require_clean: bool = True,
) -> FrozenFirstRun:
    """Bind a validated first-run configuration to code, lock, and dataset handle."""

    config = CadenceConfig.model_validate(config.model_dump())
    if not config.first_run.enabled:
        raise ValueError("first-run configuration must be enabled")
    hidden_overrides = sorted(
        key for key in os.environ if key.startswith("CADENCE_") and "__" in key
    )
    if hidden_overrides:
        raise ValueError(
            "first-run freeze rejects typed environment overrides: "
            + ", ".join(hidden_overrides)
        )
    root = Path(repo_root).resolve()
    commit = git_commit(root)
    if commit == "UNCOMMITTED":
        raise ValueError("first-run freeze requires a committed Git revision")
    if require_clean:
        _require_clean_repository(root)
    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        raise ValueError("uv.lock is required for first-run freeze")
    snapshot = bounded_first_run_snapshot(config)
    configuration_hash = stable_hash(snapshot)
    lock_hash = file_hash(lock_path)
    identity: dict[str, Any] = {
        "schema_version": "0.1.0",
        "git_commit": commit,
        "dependency_lock_hash": lock_hash,
        "dataset_snapshot_handle": dataset_snapshot_handle,
        "configuration_hash": configuration_hash,
        "configuration": snapshot,
        "private_locations_included": False,
        "credentials_included": False,
        "launch_authorized": False,
    }
    return FrozenFirstRun(
        git_commit=commit,
        dependency_lock_hash=lock_hash,
        dataset_snapshot_handle=dataset_snapshot_handle,
        configuration_hash=configuration_hash,
        package_hash=stable_hash(identity),
        configuration=snapshot,
    )


def validate_frozen_first_run(
    package: FrozenFirstRun,
    config: CadenceConfig,
    dataset_snapshot_handle: str,
    *,
    repo_root: str | Path = ".",
    require_clean: bool = True,
) -> dict[str, object]:
    """Reject drift in any frozen input without authorizing a launch."""

    expected = freeze_first_run(
        config,
        dataset_snapshot_handle,
        repo_root=repo_root,
        require_clean=require_clean,
    )
    mismatches = [
        field
        for field in (
            "git_commit",
            "dependency_lock_hash",
            "dataset_snapshot_handle",
            "configuration_hash",
            "package_hash",
            "configuration",
        )
        if getattr(package, field) != getattr(expected, field)
    ]
    if mismatches:
        raise ValueError("incompatible frozen first run: " + ", ".join(mismatches))
    return {
        "compatible": True,
        "git_commit": package.git_commit,
        "configuration_hash": package.configuration_hash,
        "package_hash": package.package_hash,
        "launch_authorized": False,
        "network_action": False,
    }


def write_frozen_first_run(package: FrozenFirstRun, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(package.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def read_frozen_first_run(path: str | Path) -> FrozenFirstRun:
    return FrozenFirstRun.model_validate_json(Path(path).read_text(encoding="utf-8"))
