"""Sanitized, dry-run-first controls for the private Cadence VPS."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.request import urlopen
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cadence.common.config import CadenceConfig
from cadence.common.data_policy import check_repository_data_policy
from cadence.common.repro import git_commit
from cadence.dataset.records import RegistryState
from cadence.storage.base import directory_size

_BACKUP_ID = re.compile(r"^vps-metadata-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
_MAX_BACKUP_MEMBER_BYTES = 64 * 1024 * 1024
_METADATA_ROOTS = ("datasets", "manifests", "reports")


class BackupFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class BackupIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    backup_id: str = Field(pattern=_BACKUP_ID.pattern)
    created_at: datetime
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: tuple[BackupFile, ...]


def prepare_private_runtime(config: CadenceConfig, *, execute: bool) -> dict[str, object]:
    """Prepare owner-only runtime directories, or return the plan without writing."""

    _require_vps(config)
    if not execute:
        return {
            "executed": False,
            "operation": "prepare-private-runtime",
            "directories": 2,
            "directory_mode": "0700",
            "file_mode": "0600",
        }

    root = config.paths.intake_root.resolve()
    backup_root = root / "backups"
    for directory in (root, backup_root):
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        directory.chmod(0o700)
    return {
        "executed": True,
        "operation": "prepare-private-runtime",
        "directories": 2,
        "directory_mode": "0700",
        "file_mode": "0600",
    }


def create_metadata_backup(
    config: CadenceConfig,
    *,
    repo_root: str | Path = ".",
    execute: bool,
) -> dict[str, object]:
    """Create an atomic private metadata backup, never a media archive."""

    _require_vps(config)
    if not execute:
        return {
            "executed": False,
            "operation": "metadata-backup",
            "includes": ["registry", "dataset-metadata"],
            "excludes": ["source-media", "segments", "credentials"],
            "retention_count": config.vps_operations.backup_retention_count,
        }

    prepare_private_runtime(config, execute=True)
    root = config.paths.intake_root.resolve()
    backup_root = root / "backups"
    commit = git_commit(Path(repo_root).resolve())
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("metadata backups require a Git checkout with an exact commit")

    now = datetime.now(UTC)
    backup_id = f"vps-metadata-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    payloads = _metadata_payloads(root)
    file_records = tuple(
        BackupFile(
            path=relative,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        for relative, payload in sorted(payloads.items())
    )
    index = BackupIndex(
        backup_id=backup_id,
        created_at=now,
        git_commit=commit,
        files=file_records,
    )
    index_payload = index.model_dump_json(indent=2).encode() + b"\n"

    output = backup_root / f"{backup_id}.tar.gz"
    temporary = backup_root / f".{backup_id}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            with tarfile.open(fileobj=handle, mode="w:gz") as archive:
                _add_archive_bytes(archive, "backup.json", index_payload, now)
                for relative, payload in sorted(payloads.items()):
                    _add_archive_bytes(archive, relative, payload, now)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        output.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)

    pruned = _prune_backups(
        backup_root, retention_count=config.vps_operations.backup_retention_count
    )
    return {
        "executed": True,
        "operation": "metadata-backup",
        "backup_id": backup_id,
        "archive_sha256": _sha256_file(output),
        "file_count": len(file_records),
        "retention_count": config.vps_operations.backup_retention_count,
        "pruned_count": pruned,
        "media_included": False,
    }


def rehearse_metadata_restore(
    config: CadenceConfig,
    backup_id: str,
) -> dict[str, object]:
    """Validate and restore a backup into an isolated temporary directory."""

    _require_vps(config)
    if _BACKUP_ID.fullmatch(backup_id) is None:
        raise ValueError("invalid backup ID")
    archive_path = config.paths.intake_root.resolve() / "backups" / f"{backup_id}.tar.gz"
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError("backup ID does not identify a regular private archive")

    payloads: dict[str, bytes] = {}
    total_size = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            relative = _safe_archive_path(member.name)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("backup contains a non-regular member")
            if member.size > _MAX_BACKUP_MEMBER_BYTES:
                raise ValueError("backup member exceeds the metadata size limit")
            total_size += member.size
            if total_size > _MAX_BACKUP_MEMBER_BYTES:
                raise ValueError("backup exceeds the metadata size limit")
            if relative in payloads:
                raise ValueError("backup contains duplicate members")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("backup member could not be read")
            payloads[relative] = extracted.read(_MAX_BACKUP_MEMBER_BYTES + 1)
            if len(payloads[relative]) != member.size:
                raise ValueError("backup member size does not match its archive record")

    index_payload = payloads.pop("backup.json", None)
    if index_payload is None:
        raise ValueError("backup index is missing")
    index = BackupIndex.model_validate_json(index_payload)
    if index.backup_id != backup_id:
        raise ValueError("backup index ID does not match the requested archive")

    indexed = {item.path: item for item in index.files}
    if set(indexed) != set(payloads):
        raise ValueError("backup index does not match archive members")
    for relative, payload in payloads.items():
        record = indexed[relative]
        if len(payload) != record.size_bytes:
            raise ValueError("backup file size verification failed")
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError("backup file checksum verification failed")

    registry_payload = payloads.get("registry.json")
    if registry_payload is None:
        raise ValueError("backup registry is missing")
    state = RegistryState.model_validate_json(registry_payload)
    _validate_dataset_metadata(payloads)

    with tempfile.TemporaryDirectory(prefix="cadence-restore-rehearsal-") as temporary:
        rehearsal_root = Path(temporary).resolve()
        rehearsal_root.chmod(0o700)
        for relative, payload in payloads.items():
            destination = _contained_destination(rehearsal_root, relative)
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            _chmod_parents(destination.parent, rehearsal_root)
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            destination.chmod(0o600)
        permissions = audit_owner_only_permissions(rehearsal_root)
        if not permissions["passed"]:
            raise ValueError("restore rehearsal produced unsafe permissions")

    return {
        "operation": "metadata-restore-rehearsal",
        "backup_id": backup_id,
        "passed": True,
        "file_count": len(payloads),
        "source_count": len(state.sources),
        "segment_count": len(state.segments),
        "dataset_count": len(state.datasets),
        "production_state_modified": False,
    }


def run_vps_doctor(
    config: CadenceConfig,
    *,
    expected_commit: str,
    repo_root: str | Path = ".",
    require_health: bool = False,
) -> dict[str, object]:
    """Run read-only deployment checks and return sanitized evidence."""

    _require_vps(config)
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("expected commit must be a full 40-character Git SHA")
    root = Path(repo_root).resolve()
    checks: list[dict[str, str]] = []

    actual_commit = git_commit(root)
    _check(
        checks,
        "exact-git-commit",
        actual_commit == expected_commit,
        "exact approved commit" if actual_commit == expected_commit else "commit mismatch",
    )
    clean = _git_is_clean(root)
    _check(checks, "clean-worktree", clean, "clean" if clean else "uncommitted changes")

    policy = check_repository_data_policy(root)
    _check(
        checks,
        "repository-data-policy",
        policy.passed,
        "passed" if policy.passed else f"{len(policy.violations)} violation(s)",
    )

    required_tools = (
        "git",
        "uv",
        config.dataset_intake.ffmpeg_binary,
        config.dataset_intake.ffprobe_binary,
    )
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    _check(
        checks,
        "required-tools",
        not missing_tools,
        "available" if not missing_tools else f"{len(missing_tools)} missing",
    )

    intake_root = config.paths.intake_root.resolve()
    root_exists = intake_root.is_dir() and not intake_root.is_symlink()
    _check(
        checks,
        "private-runtime-root",
        root_exists,
        "present" if root_exists else "missing",
    )

    permission_report: dict[str, int | bool]
    if root_exists:
        permission_report = audit_owner_only_permissions(intake_root)
        _check(
            checks,
            "owner-only-permissions",
            bool(permission_report["passed"]),
            "0700 directories and 0600 files"
            if permission_report["passed"]
            else f"{permission_report['violation_count']} violation(s)",
        )
    else:
        permission_report = {
            "passed": False,
            "directory_count": 0,
            "file_count": 0,
            "violation_count": 1,
        }

    registry_ok = True
    if root_exists and (intake_root / "registry.json").exists():
        try:
            RegistryState.model_validate_json(
                (intake_root / "registry.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            registry_ok = False
    _check(
        checks,
        "registry-schema",
        registry_ok,
        "valid or not yet initialized" if registry_ok else "invalid",
    )

    storage = _sanitized_storage_report(config, intake_root) if root_exists else None
    if storage is not None:
        storage_ok = (
            int(storage["working_bytes"]) <= int(storage["maximum_working_bytes"])
            and int(storage["filesystem_free_bytes"]) >= int(storage["minimum_free_bytes"])
        )
        _check(
            checks,
            "storage-capacity",
            storage_ok,
            "within configured limits" if storage_ok else "configured limit exceeded",
        )

    if require_health:
        health_ok = _loopback_health_ok(config.vps_operations.review_health_url)
        _check(
            checks,
            "loopback-review-health",
            health_ok,
            "healthy" if health_ok else "unreachable or unhealthy",
        )
    else:
        checks.append(
            {
                "name": "loopback-review-health",
                "status": "skipped",
                "detail": "rerun with --require-health after starting the console",
            }
        )

    return {
        "passed": all(item["status"] != "failed" for item in checks),
        "git_commit": actual_commit,
        "checks": checks,
        "permissions": permission_report,
        "storage": storage,
        "remote_actions_executed": False,
    }


def audit_owner_only_permissions(root: Path) -> dict[str, int | bool]:
    """Check that a private runtime tree has no symlinks or broad permissions."""

    expected_owner = os.getuid()
    directories = 0
    files = 0
    violations = 0
    for path in (root, *root.rglob("*")):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            violations += 1
            continue
        if stat.S_ISLNK(metadata.st_mode):
            violations += 1
            continue
        if stat.S_ISDIR(metadata.st_mode):
            directories += 1
            expected_mode = 0o700
        elif stat.S_ISREG(metadata.st_mode):
            files += 1
            expected_mode = 0o600
        else:
            violations += 1
            continue
        if stat.S_IMODE(metadata.st_mode) != expected_mode or metadata.st_uid != expected_owner:
            violations += 1
    return {
        "passed": violations == 0,
        "directory_count": directories,
        "file_count": files,
        "violation_count": violations,
    }


def _require_vps(config: CadenceConfig) -> None:
    if config.runtime.profile != "vps":
        raise ValueError("VPS operations require a vps configuration profile")


def _metadata_payloads(root: Path) -> dict[str, bytes]:
    registry_path = root / "registry.json"
    if registry_path.exists():
        if registry_path.is_symlink() or not registry_path.is_file():
            raise ValueError("private registry must be a regular file")
        registry_payload = registry_path.read_bytes()
        RegistryState.model_validate_json(registry_payload)
    else:
        registry_payload = RegistryState().model_dump_json(indent=2).encode() + b"\n"
    payloads = {"registry.json": registry_payload}

    for directory_name in _METADATA_ROOTS:
        directory = root / directory_name
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("metadata root must be a contained directory")
        for path in sorted(directory.rglob("*")):
            if path.suffix not in {".json", ".jsonl"}:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("metadata backup refuses links and non-regular files")
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("metadata path escapes the private runtime root")
            relative = resolved.relative_to(root).as_posix()
            payload = resolved.read_bytes()
            if len(payload) > _MAX_BACKUP_MEMBER_BYTES:
                raise ValueError("metadata file exceeds the backup size limit")
            payloads[relative] = payload
    if sum(len(payload) for payload in payloads.values()) > _MAX_BACKUP_MEMBER_BYTES:
        raise ValueError("metadata backup exceeds the size limit")
    return payloads


def _add_archive_bytes(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes,
    created_at: datetime,
) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o600
    member.mtime = int(created_at.timestamp())
    archive.addfile(member, io.BytesIO(payload))


def _prune_backups(backup_root: Path, *, retention_count: int) -> int:
    backups = sorted(
        (
            path
            for path in backup_root.glob("vps-metadata-*.tar.gz")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    expired = backups[:-retention_count]
    for path in expired:
        path.unlink()
    return len(expired)


def _safe_archive_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise ValueError("backup contains an unsafe member path")
    return path.as_posix()


def _contained_destination(root: Path, relative: str) -> Path:
    destination = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError("restore destination escapes the rehearsal root")
    return destination


def _chmod_parents(path: Path, root: Path) -> None:
    current = path
    while True:
        current.chmod(0o700)
        if current == root:
            break
        current = current.parent


def _validate_dataset_metadata(payloads: dict[str, bytes]) -> None:
    for relative, payload in payloads.items():
        if relative == "registry.json":
            continue
        try:
            if relative.endswith(".jsonl"):
                for line in payload.decode().splitlines():
                    if line.strip():
                        json.loads(line)
            else:
                json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("backup contains invalid dataset metadata") from exc


def _git_is_clean(root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not result.stdout.strip()


def _check(checks: list[dict[str, str]], name: str, passed: bool, detail: str) -> None:
    checks.append(
        {
            "name": name,
            "status": "passed" if passed else "failed",
            "detail": detail,
        }
    )


def _sanitized_storage_report(config: CadenceConfig, root: Path) -> dict[str, int]:
    gib = 1024**3
    usage = shutil.disk_usage(root)
    maximum = round(config.dataset_intake.maximum_working_storage_gb * gib)
    minimum = round(config.dataset_intake.minimum_free_disk_gb * gib)
    working = directory_size(root)
    return {
        "working_bytes": working,
        "maximum_working_bytes": maximum,
        "filesystem_free_bytes": usage.free,
        "minimum_free_bytes": minimum,
        "remaining_working_bytes": max(0, maximum - working),
    }


def _loopback_health_ok(url: str) -> bool:
    try:
        with urlopen(url, timeout=3) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read(1024))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and len(payload) == 1
        and payload.get("status") == "ok"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
