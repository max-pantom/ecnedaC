from urllib.error import HTTPError

import pytest

from cadence.common.config import load_config
from cadence.remote.job import RemoteJob, remote_command, run_remote_action
from cadence.remote.runpod import build_runpod_plan, execute_runpod_plan


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
        "dataset_manifest_uri": "s3://bucket/manifest",
        "checkpoint_destination": "s3://bucket/checkpoint",
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
