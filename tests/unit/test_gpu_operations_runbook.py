from pathlib import Path

from cadence.common.config import load_config


def test_gpu_operations_runbook_covers_every_required_gate() -> None:
    text = Path("docs/operations/gpu-private-operations.md").read_text(encoding="utf-8")

    required = (
        "Human gate",
        "Aven — remote operation",
        "outbound SSH/SCP",
        "15 GB free",
        "four rotating checkpoints plus the final checkpoint",
        "Runtime-only secret lifecycle",
        "integrity verification",
        "Checkpoints and clean stopping",
        "Abort and stop rules",
        "Fresh-process resume and recovery",
        "Verified termination and cleanup",
        "Incident evidence boundary",
        "Dataset contents never come",
        "No credential, RunPod Pod, transfer, spend",
    )
    for phrase in required:
        assert phrase.lower() in text.lower()


def test_gpu_operations_runbook_matches_tightened_budget_controls() -> None:
    config = load_config("configs/first-run-v0.1.0.yaml")
    runbook = Path("docs/operations/gpu-private-operations.md").read_text(
        encoding="utf-8"
    )

    assert config.remote.maximum_budget_usd == 2
    assert config.remote.maximum_runtime_minutes == 240
    assert config.remote.maximum_hourly_price_usd == 0.30
    assert config.first_run.abort is not None
    assert config.first_run.abort.hard_stop_budget_usd == 2
    assert "`$2` total or 240 minutes" in runbook


def test_gpu_operations_runbook_contains_no_operational_values() -> None:
    text = Path("docs/operations/gpu-private-operations.md").read_text(
        encoding="utf-8"
    ).lower()

    for forbidden in (
        "s3://",
        "/srv/cadence/private",
        "runpod_api_key=",
        "secret_access_key",
        "signed url:",
        "pod_test_",
        "cadence-admin-",
    ):
        assert forbidden not in text
