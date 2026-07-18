from pathlib import Path
from urllib.error import HTTPError

import pytest

from cadence.common.config import load_config
from cadence.remote.job import (
    RemoteJob,
    package_remote_job,
    remote_command,
    run_remote_action,
)
from cadence.remote.runpod import build_runpod_plan, execute_runpod_plan
from cadence.remote.vps_transport import (
    CHECKPOINT_KEY_ENV,
    DATASET_KEY_ENV,
    build_vps_transfer_plan,
    execute_vps_transfer_plan,
)


def test_remote_job_rejects_non_commit() -> None:
    values = {
        "provider": "runpod",
        "git_commit": "UNCOMMITTED",
        "python_version": "3.12",
        "dependency_group": "training-gpu",
        "dependency_index": "https://download.pytorch.org/whl/cu126",
        "dependency_lock_hash": "a" * 64,
        "configuration_hash": "b" * 64,
        "configuration": {},
        "artifact_transport": "vps-ssh",
        "dataset_snapshot_handle": "cadence-test-snapshot",
        "checkpoint_run_handle": "cadence-test-run",
        "checkpoint_retention_count": 4,
        "random_seed": 1,
        "requested_hardware": "NVIDIA RTX A5000 24GB",
        "maximum_budget_usd": 1,
        "maximum_runtime_minutes": 1,
        "maximum_hourly_price_usd": 0.30,
        "synthetic_smoke_maximum_budget_usd": 1,
        "synthetic_smoke_maximum_runtime_minutes": 1,
        "created_at": "2026-01-01T00:00:00Z",
    }
    with pytest.raises(ValueError):
        RemoteJob.model_validate(values)


def test_remote_job_package_omits_private_and_runtime_locators() -> None:
    job = package_remote_job(
        load_config("configs/gpu-24gb.yaml"),
        require_clean=False,
    )
    encoded = job.model_dump_json().lower()

    assert job.artifact_transport == "vps-ssh"
    assert job.checkpoint_retention_count == 4
    for forbidden in (
        "s3://",
        "manifest_uri",
        "checkpoint_uri",
        "vps_host",
        "vast_instance_id",
        "intake_root",
        "manifest_path",
        "cadence_gpu_vps_host",
    ):
        assert forbidden not in encoded


def test_every_remote_action_is_dry_by_default() -> None:
    config = load_config("configs/vps.yaml")
    for action in (
        "bootstrap_vps", "doctor_vps", "submit_job", "sync_checkpoints",
        "fetch_results",
    ):
        assert run_remote_action(action, config, execute=False).startswith("DRY RUN:")
        assert remote_command(action, config)


def test_legacy_vast_termination_cannot_target_runpod() -> None:
    with pytest.raises(ValueError, match=r"legacy Vast\.ai"):
        remote_command("terminate_gpu", load_config("configs/gpu-24gb.yaml"))


def test_execute_requires_configuration() -> None:
    with pytest.raises(ValueError, match="missing"):
        run_remote_action("bootstrap_vps", load_config("configs/vps.yaml"), execute=True)


def test_legacy_storage_actions_cannot_execute() -> None:
    config = load_config("configs/vps.yaml")
    for action in ("sync_checkpoints", "fetch_results"):
        with pytest.raises(ValueError, match="gpu-transfer"):
            run_remote_action(action, config, execute=True)


def test_runpod_create_plan_is_bounded_and_public_safe() -> None:
    config = load_config("configs/gpu-24gb.yaml")

    plan = build_runpod_plan("create", config)
    encoded = str(plan.to_dict()).lower()

    assert plan.requested_hardware == "NVIDIA RTX A5000 24GB"
    assert plan.maximum_budget_usd == 2
    assert plan.maximum_runtime_minutes == 240
    assert plan.maximum_hourly_price_usd == 0.30
    assert plan.request_body is not None
    assert plan.request_body["gpuTypeIds"] == ["NVIDIA RTX A5000"]
    assert plan.request_body["gpuCount"] == 1
    assert plan.request_body["ports"] == []
    assert plan.request_body["supportPublicIp"] is False
    assert "env" not in plan.request_body
    assert "runtime-only-test-key" not in encoded
    assert "bearer" not in encoded
    assert "manifest_uri" not in encoded
    assert "checkpoint_uri" not in encoded
    assert any("billable volume storage" in warning for warning in plan.warnings)


def test_vps_transfer_plans_are_sanitized_and_use_separate_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CADENCE_GPU_VPS_HOST", "private.example.invalid")
    monkeypatch.setenv(DATASET_KEY_ENV, "/private/dataset-key")
    monkeypatch.setenv(CHECKPOINT_KEY_ENV, "/private/checkpoint-key")

    dataset = build_vps_transfer_plan(
        "dataset-pull",
        dataset_snapshot_handle="cad15-manifest-47924d80b058",
    )
    checkpoint = build_vps_transfer_plan(
        "checkpoint-push",
        run_handle="cadence-first-run-v0-1-0",
        artifact_name="step-000010.pt",
    )
    encoded = f"{dataset.to_dict()} {checkpoint.to_dict()}".lower()

    assert dataset.credential_role == "dataset-read"
    assert DATASET_KEY_ENV in dataset.runtime_environment_variables
    assert CHECKPOINT_KEY_ENV not in dataset.runtime_environment_variables
    assert checkpoint.credential_role == "checkpoint-read-write"
    assert CHECKPOINT_KEY_ENV in checkpoint.runtime_environment_variables
    assert DATASET_KEY_ENV not in checkpoint.runtime_environment_variables
    assert "private.example.invalid" not in encoded
    assert "/private/" not in encoded


def test_vps_transfer_dry_run_reads_no_credentials_or_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_environment(name: str, default: str = "") -> str:
        del name, default
        raise AssertionError("dry run read runtime environment")

    def fail_subprocess(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("dry run contacted the network")

    monkeypatch.setattr("cadence.remote.vps_transport.os.getenv", fail_environment)
    monkeypatch.setattr("cadence.remote.vps_transport.subprocess.run", fail_subprocess)
    plan = build_vps_transfer_plan(
        "dataset-pull",
        dataset_snapshot_handle="cad15-manifest-47924d80b058",
    )

    result = execute_vps_transfer_plan(
        plan,
        local_path=tmp_path / "dataset.tar.zst",
        execute=False,
    )

    assert result["network_action"] is False
    assert result["credentials_read"] is False


@pytest.mark.parametrize(
    ("action", "kwargs"),
    [
        ("dataset-pull", {"dataset_snapshot_handle": "../../private"}),
        (
            "checkpoint-push",
            {"run_handle": "cadence-valid-run", "artifact_name": "../checkpoint.pt"},
        ),
        (
            "report-pull",
            {"run_handle": "cadence-valid-run", "artifact_name": "report.txt"},
        ),
    ],
)
def test_vps_transfer_rejects_traversal_and_wrong_artifact_types(
    action: str,
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        build_vps_transfer_plan(action, **kwargs)  # type: ignore[arg-type]


def test_vps_transfer_execution_requires_human_approval(tmp_path: Path) -> None:
    plan = build_vps_transfer_plan(
        "checkpoint-push",
        run_handle="cadence-first-run-v0-1-0",
        artifact_name="step-000010.pt",
    )

    with pytest.raises(ValueError, match="approval"):
        execute_vps_transfer_plan(
            plan,
            local_path=tmp_path / "step-000010.pt",
            execute=True,
        )


def test_vps_transfer_execution_rejects_missing_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in (
        "CADENCE_GPU_VPS_HOST",
        "CADENCE_GPU_VPS_PORT",
        "CADENCE_GPU_VPS_USER",
        "CADENCE_GPU_VPS_KNOWN_HOSTS_FILE",
        CHECKPOINT_KEY_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    checkpoint = tmp_path / "step-000010.pt"
    checkpoint.write_bytes(b"checkpoint")
    plan = build_vps_transfer_plan(
        "checkpoint-push",
        run_handle="cadence-first-run-v0-1-0",
        artifact_name=checkpoint.name,
    )

    with pytest.raises(ValueError, match="missing VPS transfer runtime"):
        execute_vps_transfer_plan(
            plan,
            local_path=checkpoint,
            execute=True,
            approval_reference="cad35-test-approval",
        )


def test_vps_checkpoint_upload_is_strict_atomic_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("host key\n", encoding="utf-8")
    key = tmp_path / "checkpoint_key"
    key.write_text("private key\n", encoding="utf-8")
    key.chmod(0o600)
    checkpoint = tmp_path / "step-000010.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("CADENCE_GPU_VPS_HOST", "vps.example.invalid")
    monkeypatch.setenv("CADENCE_GPU_VPS_PORT", "22")
    monkeypatch.setenv("CADENCE_GPU_VPS_USER", "cadence_checkpoint")
    monkeypatch.setenv("CADENCE_GPU_VPS_KNOWN_HOSTS_FILE", str(known_hosts))
    monkeypatch.setenv(CHECKPOINT_KEY_ENV, str(key))
    commands: list[list[str]] = []

    def record(command: list[str], *, check: bool) -> object:
        assert check is True
        commands.append(command)
        return object()

    monkeypatch.setattr("cadence.remote.vps_transport.subprocess.run", record)
    plan = build_vps_transfer_plan(
        "checkpoint-push",
        run_handle="cadence-first-run-v0-1-0",
        artifact_name=checkpoint.name,
    )

    result = execute_vps_transfer_plan(
        plan,
        local_path=checkpoint,
        execute=True,
        approval_reference="cad35-test-approval",
    )
    encoded_commands = " ".join(part for command in commands for part in command)
    encoded_result = str(result)

    assert [command[0] for command in commands] == ["scp", "ssh"]
    assert "StrictHostKeyChecking=yes" in encoded_commands
    assert "step-000010.pt.partial" in encoded_commands
    assert "commit-upload" in encoded_commands
    assert result["network_action"] is True
    assert "vps.example.invalid" not in encoded_result
    assert str(key) not in encoded_result


def test_vps_transfer_rejects_overexposed_ssh_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("host key\n", encoding="utf-8")
    key = tmp_path / "dataset_key"
    key.write_text("private key\n", encoding="utf-8")
    key.chmod(0o644)
    monkeypatch.setenv("CADENCE_GPU_VPS_HOST", "vps.example.invalid")
    monkeypatch.setenv("CADENCE_GPU_VPS_PORT", "22")
    monkeypatch.setenv("CADENCE_GPU_VPS_USER", "cadence_dataset")
    monkeypatch.setenv("CADENCE_GPU_VPS_KNOWN_HOSTS_FILE", str(known_hosts))
    monkeypatch.setenv(DATASET_KEY_ENV, str(key))
    plan = build_vps_transfer_plan(
        "dataset-pull",
        dataset_snapshot_handle="cad15-manifest-47924d80b058",
    )

    with pytest.raises(ValueError, match="group or other"):
        execute_vps_transfer_plan(
            plan,
            local_path=tmp_path / "dataset.tar.zst",
            execute=True,
            approval_reference="cad35-test-approval",
        )


def test_every_runpod_action_is_dry_without_credentials_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("dry run contacted the network")

    monkeypatch.setattr("cadence.remote.runpod.urlopen", fail_network)
    config = load_config("configs/gpu-24gb.yaml")
    plans = (
        build_runpod_plan("search", config),
        build_runpod_plan("create", config),
        build_runpod_plan("inspect", config, pod_id="pod_test_123"),
        build_runpod_plan("terminate", config, pod_id="pod_test_123"),
    )

    for plan in plans:
        result = execute_runpod_plan(plan, execute=False)
        assert result["mode"] == "dry-run"
        assert result["network_action"] is False


def test_executed_runpod_actions_require_runtime_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    plan = build_runpod_plan("search", load_config("configs/gpu-24gb.yaml"))

    with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
        execute_runpod_plan(plan, execute=True)


def test_runpod_mutations_require_separate_human_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "runtime-only-test-key")
    config = load_config("configs/gpu-24gb.yaml")

    with pytest.raises(ValueError, match="approval"):
        execute_runpod_plan(build_runpod_plan("create", config), execute=True)
    with pytest.raises(ValueError, match="confirm-termination"):
        execute_runpod_plan(
            build_runpod_plan("terminate", config, pod_id="pod_test_123"),
            execute=True,
            approval_reference="human-gate-1234",
        )


def test_runpod_create_is_idempotent_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_request(method: str, endpoint: str, api_key: str, body: object) -> object:
        del endpoint, api_key, body
        calls.append(method)
        return [{"name": "cadence-bounded-a5000"}]

    monkeypatch.setenv("RUNPOD_API_KEY", "runtime-only-test-key")
    monkeypatch.setattr("cadence.remote.runpod._request_json", fake_request)
    plan = build_runpod_plan("create", load_config("configs/gpu-24gb.yaml"))

    result = execute_runpod_plan(
        plan,
        execute=True,
        approval_reference="human-gate-1234",
    )

    assert result["status"] == "already-exists"
    assert calls == ["GET"]


def test_runpod_termination_is_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_request(method: str, endpoint: str, api_key: str, body: object) -> object:
        del api_key, body
        calls.append(method)
        if method == "GET":
            raise HTTPError(endpoint, 404, "not found", {}, None)
        return {}

    monkeypatch.setenv("RUNPOD_API_KEY", "runtime-only-test-key")
    monkeypatch.setattr("cadence.remote.runpod._request_json", fake_request)
    plan = build_runpod_plan(
        "terminate",
        load_config("configs/gpu-24gb.yaml"),
        pod_id="pod_test_123",
    )

    result = execute_runpod_plan(
        plan,
        execute=True,
        approval_reference="human-gate-1234",
        confirm_termination=True,
    )

    assert result["status"] == "terminated-and-verified"
    assert calls == ["DELETE", "GET"]


def test_runpod_over_price_creation_attempts_immediate_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_request(method: str, endpoint: str, api_key: str, body: object) -> object:
        del endpoint, api_key, body
        calls.append(method)
        if calls == ["GET"]:
            return []
        if calls == ["GET", "POST"]:
            return {"id": "pod_test_123", "costPerHr": "0.31"}
        return {}

    monkeypatch.setenv("RUNPOD_API_KEY", "runtime-only-test-key")
    monkeypatch.setattr("cadence.remote.runpod._request_json", fake_request)
    plan = build_runpod_plan("create", load_config("configs/gpu-24gb.yaml"))

    with pytest.raises(RuntimeError, match="exceeded the approved hourly price"):
        execute_runpod_plan(
            plan,
            execute=True,
            approval_reference="human-gate-1234",
        )

    assert calls == ["GET", "POST", "DELETE"]
