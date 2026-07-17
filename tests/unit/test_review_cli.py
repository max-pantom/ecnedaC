from __future__ import annotations

import json

import pytest

import cadence.review.share as review_share
from cadence.cli import _is_loopback_host, main
from cadence.review.share import (
    WORMKEY_VERSION,
    build_wormkey_share_plan,
    execute_wormkey_share,
    expiry_seconds,
)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_review_server_recognizes_loopback_hosts(host: str) -> None:
    assert _is_loopback_host(host) is True


def test_review_server_requires_runtime_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CADENCE_REVIEW_ADMIN_SECRET", raising=False)

    with pytest.raises(ValueError, match="at least 32 characters"):
        main(["review-serve", "--config", "configs/test.yaml"])


def test_non_loopback_review_server_requires_both_safety_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CADENCE_REVIEW_ADMIN_SECRET", "s" * 32)
    monkeypatch.delenv("CADENCE_REVIEW_SECURE_DEPLOYMENT", raising=False)

    with pytest.raises(ValueError, match="non-loopback"):
        main(
            [
                "review-serve",
                "--config",
                "configs/test.yaml",
                "--host",
                "0.0.0.0",
                "--allow-non-loopback",
            ]
        )


def test_wormkey_share_plan_is_pinned_guarded_and_dry_run() -> None:
    plan = build_wormkey_share_plan("configs/vps.yaml", port=8787, expires="30m")

    assert plan.execute is False
    assert plan.package == f"wormkey@{WORMKEY_VERSION}"
    assert plan.command == (
        "npx",
        "--yes",
        f"wormkey@{WORMKEY_VERSION}",
        "http",
        "8787",
        "--expires",
        "30m",
    )
    assert any("Basic authentication" in protection for protection in plan.protections)


@pytest.mark.parametrize("expires", ["4m", "121m", "3h", "forever", "0m"])
def test_wormkey_share_rejects_unsafe_expiry(expires: str) -> None:
    with pytest.raises(ValueError, match="expiry"):
        expiry_seconds(expires)


def test_wormkey_share_cli_is_dry_run_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "review-share",
                "--config",
                "configs/test.yaml",
                "--expires",
                "15m",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["provider"] == "wormkey"
    assert output["execute"] is False
    assert output["expires"] == "15m"


def test_wormkey_execution_requires_vps_provisioned_admin_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CADENCE_REVIEW_ADMIN_SECRET", raising=False)
    plan = build_wormkey_share_plan("configs/test.yaml", execute=True)

    with pytest.raises(ValueError, match="auth_secret"):
        execute_wormkey_share(plan)


def test_external_tunnel_server_requires_ephemeral_outer_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CADENCE_REVIEW_ADMIN_SECRET", "s" * 32)
    monkeypatch.delenv("CADENCE_REVIEW_TUNNEL_BASIC_USERNAME", raising=False)
    monkeypatch.delenv("CADENCE_REVIEW_TUNNEL_BASIC_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="tunnel username"):
        main(
            [
                "review-serve",
                "--config",
                "configs/test.yaml",
                "--external-tunnel",
                "wormkey",
            ]
        )


def test_wormkey_execution_emits_only_relay_credentials_and_sanitizes_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeProcess:
        def __init__(self, output: list[str]) -> None:
            self.stdout = iter(output) if output else None
            self.return_code: int | None = None

        def poll(self) -> int | None:
            return self.return_code

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.return_code = 0
            return 0

        def terminate(self) -> None:
            self.return_code = 0

        def kill(self) -> None:
            self.return_code = -9

    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    processes = [
        FakeProcess([]),
        FakeProcess(
            [
                "https://wormkey.run/s/quiet-lime-82\n",
                "https://wormkey.run/.wormkey/owner?secret=must-not-leak\n",
            ]
        ),
    ]

    def fake_popen(command: tuple[str, ...], **kwargs: object) -> FakeProcess:
        calls.append((command, kwargs))
        return processes[len(calls) - 1]

    monkeypatch.setenv("CADENCE_REVIEW_ADMIN_SECRET", "s" * 32)
    monkeypatch.setenv("VAST_API_KEY", "must-not-reach-npx")
    monkeypatch.setattr(review_share.shutil, "which", lambda _: "/usr/bin/npx")
    monkeypatch.setattr(review_share.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(review_share, "_ensure_port_available", lambda *args: None)
    monkeypatch.setattr(review_share, "_wait_for_server", lambda *args: None)
    monkeypatch.setattr(review_share.secrets, "token_urlsafe", lambda _: "p" * 32)
    plan = build_wormkey_share_plan("configs/test.yaml", execute=True)

    assert execute_wormkey_share(plan) == 0

    relay = json.loads(capsys.readouterr().out)
    assert relay["url"] == "https://wormkey.run/s/quiet-lime-82"
    assert relay["basic_auth"] == {"username": "cadence", "password": "p" * 32}
    tunnel_command, tunnel_options = calls[1]
    assert tunnel_command[0] == "/usr/bin/npx"
    assert "--auth" not in tunnel_command
    tunnel_environment = tunnel_options["env"]
    assert isinstance(tunnel_environment, dict)
    assert "CADENCE_REVIEW_ADMIN_SECRET" not in tunnel_environment
    assert "VAST_API_KEY" not in tunnel_environment
