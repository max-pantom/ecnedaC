"""Fail when Git tracks private Cadence data or generated training artifacts."""

from __future__ import annotations

from pathlib import Path

from cadence.common.data_policy import check_repository_data_policy


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    report = check_repository_data_policy(repo_root)
    if not report.passed:
        print("Git tracks files forbidden by the Cadence private-data policy:")
        for path in report.violations:
            print(f"- {path}")
        print("Keep real media, registries, manifests, and generated artifacts on the VPS.")
        return 1
    print("Cadence private-data policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
