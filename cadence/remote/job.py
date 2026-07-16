"""Remote training job package generation."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cadence.common.config import CadenceConfig
from cadence.common.repro import git_commit, stable_hash


class RemoteJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    python_version: Literal["3.12"]
    dependency_group: Literal["training-gpu"]
    dependency_index: str
    dependency_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: dict[str, Any]
    dataset_manifest_uri: str
    checkpoint_destination: str
    random_seed: int
    requested_hardware: str
    maximum_budget_usd: float = Field(gt=0)
    maximum_runtime_minutes: int = Field(gt=0)
    created_at: datetime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_remote_job(
    config: CadenceConfig, *, repo_root: str | Path = ".", require_clean: bool = True
) -> RemoteJob:
    root = Path(repo_root).resolve()
    commit = git_commit(root)
    if commit == "UNCOMMITTED":
        raise ValueError("remote jobs require a committed Git revision")
    if require_clean:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
        )
        if result.stdout.strip():
            raise ValueError("remote jobs require a clean Git worktree")
    lock = root / "uv.lock"
    if not lock.is_file():
        raise ValueError("uv.lock is required for remote jobs")
    configuration = config.model_dump(mode="json")
    if config.remote.python_version != "3.12":
        raise ValueError("GPU remote jobs require Python 3.12")
    return RemoteJob(
        git_commit=commit,
        python_version="3.12",
        dependency_group=config.remote.dependency_group,
        dependency_index="https://download.pytorch.org/whl/cu126",
        dependency_lock_hash=_sha256(lock),
        configuration_hash=stable_hash(configuration),
        configuration=configuration,
        dataset_manifest_uri=config.remote.manifest_uri,
        checkpoint_destination=config.remote.checkpoint_uri,
        random_seed=config.runtime.seed,
        requested_hardware=config.remote.requested_hardware,
        maximum_budget_usd=config.remote.maximum_budget_usd,
        maximum_runtime_minutes=config.remote.maximum_runtime_minutes,
        created_at=datetime.now(UTC),
    )


def remote_command(action: str, config: CadenceConfig) -> list[str]:
    host = config.remote.vps_host or os.getenv("CADENCE_VPS_HOST")
    instance = config.remote.vast_instance_id or os.getenv("CADENCE_VAST_INSTANCE_ID")
    commands: dict[str, list[str]] = {
        "bootstrap_vps": ["ssh", host or "<missing-vps-host>", "mkdir -p cadence-jobs"],
        "doctor_vps": ["ssh", host or "<missing-vps-host>", "python3 --version && git --version"],
        "submit_job": [
            "ssh",
            host or "<missing-vps-host>",
            "cadence train-contrastive --config configs/gpu-24gb.yaml",
        ],
        "sync_checkpoints": [
            "aws", "s3", "sync", config.remote.checkpoint_uri, "artifacts/checkpoints"
        ],
        "fetch_results": ["aws", "s3", "sync", config.remote.checkpoint_uri, "artifacts/reports"],
        "terminate_gpu": ["vastai", "destroy", "instance", instance or "<missing-instance-id>"],
    }
    if action not in commands:
        raise ValueError(f"unknown remote action: {action}")
    return commands[action]


def run_remote_action(action: str, config: CadenceConfig, *, execute: bool) -> str:
    command = remote_command(action, config)
    rendered = shlex.join(command)
    if not execute:
        return f"DRY RUN: {rendered}"
    if any(part.startswith("<missing-") for part in command):
        raise ValueError("remote action is missing required host or instance configuration")
    if action == "terminate_gpu" and not os.getenv("VAST_API_KEY"):
        raise ValueError("VAST_API_KEY is required to terminate a GPU instance")
    subprocess.run(command, check=True)
    return f"EXECUTED: {rendered}"


def write_remote_job(job: RemoteJob, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(job.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
