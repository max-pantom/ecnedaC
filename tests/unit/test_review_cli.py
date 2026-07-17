from __future__ import annotations

import pytest

from cadence.cli import _is_loopback_host, main


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
