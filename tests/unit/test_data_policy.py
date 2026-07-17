import subprocess
from pathlib import Path, PurePosixPath

from cadence.cli import build_parser, main
from cadence.common.data_policy import (
    ALLOWED_PLACEHOLDERS,
    check_repository_data_policy,
    policy_violations,
)


def test_policy_allows_only_declared_placeholders_in_private_roots() -> None:
    assert policy_violations(ALLOWED_PLACEHOLDERS) == ()
    assert policy_violations(
        [
            PurePosixPath("data/intake/registry.json"),
            PurePosixPath("data/pilots/launch-video/sources.jsonl"),
            PurePosixPath("examples/real-video.mp4"),
            PurePosixPath("private/registry.json"),
        ]
    ) == (
        PurePosixPath("data/intake/registry.json"),
        PurePosixPath("data/pilots/launch-video/sources.jsonl"),
        PurePosixPath("examples/real-video.mp4"),
        PurePosixPath("private/registry.json"),
    )


def test_repository_check_sees_force_added_private_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "code.py").write_text("value = 1\n", encoding="utf-8")
    private = tmp_path / "data" / "intake" / "registry.json"
    private.parent.mkdir(parents=True)
    private.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "code.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "--force", "data/intake/registry.json"], cwd=tmp_path, check=True)

    report = check_repository_data_policy(tmp_path)

    assert report.passed is False
    assert report.violations == (PurePosixPath("data/intake/registry.json"),)
    assert report.tracked_file_count == 2


def test_data_policy_cli_contract_and_failure_exit(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    private = tmp_path / "manifest.jsonl"
    private.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "--force", "manifest.jsonl"], cwd=tmp_path, check=True)

    args = build_parser().parse_args(
        ["data-policy", "check", "--repo-root", str(tmp_path)]
    )
    assert args.command == "data-policy"
    assert main(["data-policy", "check", "--repo-root", str(tmp_path)]) == 1
