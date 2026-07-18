"""Public-safe, dry-run-first VPS/GPU artifact transfer plans."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cadence.common.repro import stable_hash

TransferAction = Literal[
    "dataset-pull",
    "checkpoint-push",
    "checkpoint-pull",
    "report-push",
    "report-pull",
]
CredentialRole = Literal["dataset-read", "checkpoint-read-write"]

_HANDLE_PATTERN = r"^[a-z0-9][a-z0-9-]{7,127}$"
_ARTIFACT_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_APPROVAL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$"
_MAX_ARTIFACT_BYTES = 2 * 1024**3

VPS_HOST_ENV = "CADENCE_GPU_VPS_HOST"
VPS_PORT_ENV = "CADENCE_GPU_VPS_PORT"
VPS_USER_ENV = "CADENCE_GPU_VPS_USER"
VPS_KNOWN_HOSTS_ENV = "CADENCE_GPU_VPS_KNOWN_HOSTS_FILE"
DATASET_KEY_ENV = "CADENCE_GPU_VPS_DATASET_KEY_FILE"
CHECKPOINT_KEY_ENV = "CADENCE_GPU_VPS_CHECKPOINT_KEY_FILE"


class VpsTransferPlan(BaseModel):
    """Sanitized plan; runtime connection details are intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    transport: Literal["vps-ssh"] = "vps-ssh"
    action: TransferAction
    credential_role: CredentialRole
    dataset_snapshot_handle: str | None = Field(default=None, pattern=_HANDLE_PATTERN)
    run_handle: str | None = Field(default=None, pattern=_HANDLE_PATTERN)
    artifact_name: str = Field(pattern=_ARTIFACT_PATTERN)
    maximum_artifact_bytes: Literal[2147483648] = 2147483648
    maximum_parallel_transfers: Literal[1] = 1
    requires_strict_host_key_checking: Literal[True] = True
    requires_explicit_execute: Literal[True] = True
    requires_human_approval: Literal[True] = True
    runtime_environment_variables: tuple[str, ...]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_action_contract(self) -> VpsTransferPlan:
        if self.action == "dataset-pull":
            if self.dataset_snapshot_handle is None or self.run_handle is not None:
                raise ValueError("dataset-pull requires only a dataset snapshot handle")
            if self.credential_role != "dataset-read":
                raise ValueError("dataset-pull requires the read-only dataset credential")
            if not self.artifact_name.endswith(".tar.zst"):
                raise ValueError("dataset exports require a .tar.zst extension")
        else:
            if self.run_handle is None or self.dataset_snapshot_handle is not None:
                raise ValueError("checkpoint/report actions require only a run handle")
            if self.credential_role != "checkpoint-read-write":
                raise ValueError(
                    "checkpoint/report actions require the checkpoint credential"
                )
        if self.action.startswith("checkpoint") and not self.artifact_name.endswith(
            (".pt", ".pth", ".ckpt")
        ):
            raise ValueError("checkpoint artifacts require a checkpoint file extension")
        if self.action.startswith("report") and not self.artifact_name.endswith(".json"):
            raise ValueError("report artifacts require a .json extension")
        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def build_vps_transfer_plan(
    action: TransferAction,
    *,
    dataset_snapshot_handle: str | None = None,
    run_handle: str | None = None,
    artifact_name: str | None = None,
) -> VpsTransferPlan:
    """Build a deterministic plan without reading environment variables."""

    if action == "dataset-pull":
        credential_role: CredentialRole = "dataset-read"
        resolved_name = artifact_name or "dataset-snapshot.tar.zst"
        credential_env = DATASET_KEY_ENV
    else:
        credential_role = "checkpoint-read-write"
        if artifact_name is None:
            raise ValueError(f"{action} requires an artifact name")
        resolved_name = artifact_name
        credential_env = CHECKPOINT_KEY_ENV
    identity = {
        "schema_version": "0.1.0",
        "transport": "vps-ssh",
        "action": action,
        "credential_role": credential_role,
        "dataset_snapshot_handle": dataset_snapshot_handle,
        "run_handle": run_handle,
        "artifact_name": resolved_name,
        "maximum_artifact_bytes": _MAX_ARTIFACT_BYTES,
        "maximum_parallel_transfers": 1,
    }
    return VpsTransferPlan(
        action=action,
        credential_role=credential_role,
        dataset_snapshot_handle=dataset_snapshot_handle,
        run_handle=run_handle,
        artifact_name=resolved_name,
        runtime_environment_variables=(
            VPS_HOST_ENV,
            VPS_PORT_ENV,
            VPS_USER_ENV,
            VPS_KNOWN_HOSTS_ENV,
            credential_env,
        ),
        plan_hash=stable_hash(identity),
    )


def execute_vps_transfer_plan(
    plan: VpsTransferPlan,
    *,
    local_path: str | Path,
    execute: bool,
    approval_reference: str | None = None,
) -> dict[str, object]:
    """Execute one bounded SCP transfer after runtime and human gates."""

    if not execute:
        return {
            "mode": "dry-run",
            "network_action": False,
            "credentials_read": False,
            "plan": plan.to_dict(),
        }
    if approval_reference is None or re.fullmatch(
        _APPROVAL_PATTERN, approval_reference
    ) is None:
        raise ValueError("executed transfer requires an opaque human approval reference")

    runtime = _load_runtime(plan)
    requested_path = Path(local_path).expanduser()
    if requested_path.is_symlink():
        raise ValueError("local artifact path must not be a symlink")
    path = requested_path.resolve()
    if plan.action.endswith("-push"):
        _validate_upload(path)
        checksum = _sha256(path)
        _upload(plan, path, checksum, runtime)
    else:
        _download(plan, path, runtime)
        checksum = _sha256(path)
    return {
        "mode": "executed",
        "network_action": True,
        "action": plan.action,
        "plan_hash": plan.plan_hash,
        "artifact_name": plan.artifact_name,
        "size_bytes": path.stat().st_size,
        "sha256_computed": True,
    }


def _load_runtime(plan: VpsTransferPlan) -> tuple[str, int, str, Path, Path]:
    values = {name: os.getenv(name, "") for name in plan.runtime_environment_variables}
    missing = sorted(name for name, value in values.items() if not value)
    if missing:
        raise ValueError("missing VPS transfer runtime configuration: " + ", ".join(missing))
    host = values[VPS_HOST_ENV]
    user = values[VPS_USER_ENV]
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", host) is None:
        raise ValueError("invalid VPS host")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,31}", user) is None:
        raise ValueError("invalid VPS user")
    try:
        port = int(values[VPS_PORT_ENV])
    except ValueError:
        raise ValueError("invalid VPS port") from None
    if not 1 <= port <= 65535:
        raise ValueError("invalid VPS port")
    known_hosts = _require_regular_file(
        Path(values[VPS_KNOWN_HOSTS_ENV]), "known-hosts file", private=False
    )
    key_env = DATASET_KEY_ENV if plan.credential_role == "dataset-read" else CHECKPOINT_KEY_ENV
    key_file = _require_regular_file(Path(values[key_env]), "SSH key", private=True)
    return host, port, user, known_hosts, key_file


def _require_regular_file(path: Path, label: str, *, private: bool) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must be a regular, non-symlink file")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular, non-symlink file")
    if private and stat.S_IMODE(resolved.stat().st_mode) & 0o077:
        raise ValueError(f"{label} must not be accessible by group or other users")
    return resolved


def _validate_upload(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("upload source must be a regular, non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_ARTIFACT_BYTES:
        raise ValueError("upload source is empty or exceeds the 2 GiB artifact limit")


def _ssh_options(runtime: tuple[str, int, str, Path, Path]) -> list[str]:
    _, port, _, known_hosts, key_file = runtime
    return [
        "-B",
        "-q",
        "-P",
        str(port),
        "-i",
        str(key_file),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]


def _remote_path(plan: VpsTransferPlan, *, partial: bool = False) -> str:
    if plan.action == "dataset-pull":
        assert plan.dataset_snapshot_handle is not None
        return f"datasets/{plan.dataset_snapshot_handle}/{plan.artifact_name}"
    assert plan.run_handle is not None
    category = "checkpoints" if plan.action.startswith("checkpoint") else "reports"
    suffix = ".partial" if partial else ""
    return f"runs/{plan.run_handle}/{category}/{plan.artifact_name}{suffix}"


def _remote(runtime: tuple[str, int, str, Path, Path], path: str) -> str:
    host, _, user, _, _ = runtime
    return f"{user}@{host}:{path}"


def _download(
    plan: VpsTransferPlan,
    local_path: Path,
    runtime: tuple[str, int, str, Path, Path],
) -> None:
    local_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = local_path.with_name(f".{local_path.name}.partial")
    temporary.unlink(missing_ok=True)
    command = [
        "scp",
        *_ssh_options(runtime),
        _remote(runtime, _remote_path(plan)),
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        size = temporary.stat().st_size
        if size <= 0 or size > _MAX_ARTIFACT_BYTES:
            raise ValueError("download is empty or exceeds the 2 GiB artifact limit")
        temporary.chmod(0o600)
        os.replace(temporary, local_path)
    finally:
        temporary.unlink(missing_ok=True)


def _upload(
    plan: VpsTransferPlan,
    local_path: Path,
    checksum: str,
    runtime: tuple[str, int, str, Path, Path],
) -> None:
    upload = [
        "scp",
        *_ssh_options(runtime),
        str(local_path),
        _remote(runtime, _remote_path(plan, partial=True)),
    ]
    subprocess.run(upload, check=True)
    host, port, user, known_hosts, key_file = runtime
    commit = [
        "ssh",
        "-p",
        str(port),
        "-i",
        str(key_file),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        f"{user}@{host}",
        "cadence-private-transfer",
        "commit-upload",
        "--run-handle",
        plan.run_handle or "",
        "--artifact-kind",
        "checkpoint" if plan.action == "checkpoint-push" else "report",
        "--artifact-name",
        plan.artifact_name,
        "--sha256",
        checksum,
    ]
    subprocess.run(commit, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
