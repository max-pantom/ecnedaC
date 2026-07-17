from __future__ import annotations

import json
import stat
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cadence.common.config import VpsOperationsConfig, load_config
from cadence.dataset.records import IntakeRegistry, RegistryState, SourceRecord
from cadence.operations import vps
from cadence.operations.vps import (
    audit_owner_only_permissions,
    create_metadata_backup,
    prepare_private_runtime,
    rehearse_metadata_restore,
    run_vps_doctor,
)

COMMIT = "a" * 40


def vps_config(tmp_path: Path, *, retention_count: int = 7):
    base = load_config("configs/vps.yaml")
    return base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={"intake_root": tmp_path / "private"}
            ),
            "vps_operations": base.vps_operations.model_copy(
                update={"backup_retention_count": retention_count}
            ),
        }
    )


def test_prepare_is_dry_by_default_and_execute_sets_owner_only_modes(
    tmp_path: Path,
) -> None:
    config = vps_config(tmp_path)

    plan = prepare_private_runtime(config, execute=False)
    assert plan["executed"] is False
    assert not config.paths.intake_root.exists()

    result = prepare_private_runtime(config, execute=True)
    assert result["executed"] is True
    assert stat.S_IMODE(config.paths.intake_root.stat().st_mode) == 0o700
    assert stat.S_IMODE((config.paths.intake_root / "backups").stat().st_mode) == 0o700


def test_backup_and_restore_rehearsal_validate_private_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = vps_config(tmp_path)
    prepare_private_runtime(config, execute=True)
    registry = IntakeRegistry(config.paths.intake_root)
    source = SourceRecord.from_submission("https://example.com/private.mp4", "operator")

    def seed(state: RegistryState) -> None:
        state.sources[str(source.source_id)] = source

    registry.mutate(seed)
    dataset_root = config.paths.intake_root / "datasets" / "synthetic" / "v0001"
    dataset_root.mkdir(parents=True, mode=0o700)
    for directory in (config.paths.intake_root / "datasets", dataset_root.parent, dataset_root):
        directory.chmod(0o700)
    report = dataset_root / "report.json"
    report.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    report.chmod(0o600)
    manifest = dataset_root / "manifest.jsonl"
    manifest.write_text(json.dumps({"synthetic": True}) + "\n", encoding="utf-8")
    manifest.chmod(0o600)
    media = config.paths.intake_root / "sources.mp4"
    media.write_bytes(b"not real media")
    media.chmod(0o600)
    monkeypatch.setattr(vps, "git_commit", lambda _: COMMIT)

    backup = create_metadata_backup(config, repo_root=tmp_path, execute=True)
    assert backup["executed"] is True
    assert backup["media_included"] is False
    backup_id = str(backup["backup_id"])
    archive = config.paths.intake_root / "backups" / f"{backup_id}.tar.gz"
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert "registry.json" in names
    assert "datasets/synthetic/v0001/report.json" in names
    assert "sources.mp4" not in names

    rehearsal = rehearse_metadata_restore(config, backup_id)
    assert rehearsal == {
        "operation": "metadata-restore-rehearsal",
        "backup_id": backup_id,
        "passed": True,
        "file_count": 3,
        "source_count": 1,
        "segment_count": 0,
        "dataset_count": 0,
        "production_state_modified": False,
    }
    assert registry.get_source(source.source_id) == source


def test_backup_retention_prunes_oldest_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = vps_config(tmp_path, retention_count=2)
    monkeypatch.setattr(vps, "git_commit", lambda _: COMMIT)

    results = [
        create_metadata_backup(config, repo_root=tmp_path, execute=True)
        for _ in range(3)
    ]

    archives = list((config.paths.intake_root / "backups").glob("*.tar.gz"))
    assert len(archives) == 2
    assert results[-1]["pruned_count"] == 1


def test_restore_rehearsal_rejects_archive_path_traversal(tmp_path: Path) -> None:
    config = vps_config(tmp_path)
    prepare_private_runtime(config, execute=True)
    backup_id = "vps-metadata-20260717T120000Z-deadbeef"
    archive = config.paths.intake_root / "backups" / f"{backup_id}.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("escape", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname="../escape")
    archive.chmod(0o600)

    with pytest.raises(ValueError, match="unsafe member path"):
        rehearse_metadata_restore(config, backup_id)


def test_permission_audit_rejects_broad_file_mode(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    unsafe = root / "registry.json"
    unsafe.write_text("{}", encoding="utf-8")
    unsafe.chmod(0o644)

    report = audit_owner_only_permissions(root)

    assert report["passed"] is False
    assert report["violation_count"] == 1


def test_doctor_returns_sanitized_checks_without_remote_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = vps_config(tmp_path)
    prepare_private_runtime(config, execute=True)
    monkeypatch.setattr(vps, "git_commit", lambda _: COMMIT)
    monkeypatch.setattr(vps, "_git_is_clean", lambda _: True)
    monkeypatch.setattr(
        vps,
        "check_repository_data_policy",
        lambda _: SimpleNamespace(passed=True, violations=()),
    )
    monkeypatch.setattr(vps.shutil, "which", lambda _: "/synthetic/tool")
    monkeypatch.setattr(
        vps,
        "_sanitized_storage_report",
        lambda *_: {
            "working_bytes": 1,
            "maximum_working_bytes": 2,
            "filesystem_free_bytes": 2,
            "minimum_free_bytes": 1,
            "remaining_working_bytes": 1,
        },
    )

    report = run_vps_doctor(
        config,
        expected_commit=COMMIT,
        repo_root=tmp_path,
    )

    assert report["passed"] is True
    assert report["remote_actions_executed"] is False
    assert report["git_commit"] == COMMIT
    rendered = json.dumps(report)
    assert str(config.paths.intake_root) not in rendered
    assert any(
        item["status"] == "skipped"
        for item in report["checks"]  # type: ignore[union-attr]
    )


def test_vps_health_configuration_requires_loopback() -> None:
    with pytest.raises(ValueError, match="loopback"):
        VpsOperationsConfig(review_health_url="https://public.example/healthz")


def test_private_stack_script_is_dry_run_by_default() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/vps/prepare_private_stack.sh",
            "--expected-commit",
            COMMIT,
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "DRY RUN:" in result.stdout
    assert "sync locked" in result.stdout
