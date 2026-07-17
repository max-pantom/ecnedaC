"""Repository guardrails for Cadence private dataset and training artifacts."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ALLOWED_PLACEHOLDERS = frozenset(
    {
        PurePosixPath("artifacts/checkpoints/.gitkeep"),
        PurePosixPath("artifacts/exports/.gitkeep"),
        PurePosixPath("artifacts/reports/.gitkeep"),
        PurePosixPath("data/cache/.gitkeep"),
        PurePosixPath("data/intake/.gitkeep"),
        PurePosixPath("data/manifests/.gitkeep"),
    }
)

PRIVATE_ROOTS = (
    PurePosixPath("artifacts"),
    PurePosixPath("data/cache"),
    PurePosixPath("data/intake"),
    PurePosixPath("data/manifests"),
    PurePosixPath("data/pilots"),
)

PRIVATE_SUFFIXES = frozenset(
    {
        ".aac",
        ".avi",
        ".ckpt",
        ".flac",
        ".jsonl",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".pt",
        ".pth",
        ".wav",
        ".webm",
    }
)


@dataclass(frozen=True)
class DataPolicyReport:
    """Typed result of checking Git's tracked index against the private-data policy."""

    repository_root: Path
    tracked_file_count: int
    violations: tuple[PurePosixPath, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "repository_root": str(self.repository_root),
            "tracked_file_count": self.tracked_file_count,
            "violations": [str(path) for path in self.violations],
        }


def find_repository_root(start: str | Path = ".") -> Path:
    """Resolve the Git worktree root containing ``start``."""

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(start).resolve(),
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def tracked_paths(repository_root: str | Path) -> tuple[PurePosixPath, ...]:
    """Return every path currently tracked by Git, including staged additions."""

    root = Path(repository_root).resolve()
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(
        PurePosixPath(value.decode("utf-8"))
        for value in result.stdout.split(b"\0")
        if value
    )


def policy_violations(paths: Iterable[PurePosixPath]) -> tuple[PurePosixPath, ...]:
    """Identify tracked paths forbidden by the Cadence private-data policy."""

    violations: set[PurePosixPath] = set()
    for path in paths:
        if path in ALLOWED_PLACEHOLDERS:
            continue
        inside_private_root = any(path == root or root in path.parents for root in PRIVATE_ROOTS)
        if (
            inside_private_root
            or path.name == "registry.json"
            or path.suffix.lower() in PRIVATE_SUFFIXES
        ):
            violations.add(path)
    return tuple(sorted(violations))


def check_repository_data_policy(
    repository_root: str | Path | None = None,
) -> DataPolicyReport:
    """Check a repository without reading any private file contents."""

    root = (
        find_repository_root()
        if repository_root is None
        else find_repository_root(repository_root)
    )
    paths = tracked_paths(root)
    return DataPolicyReport(
        repository_root=root,
        tracked_file_count=len(paths),
        violations=policy_violations(paths),
    )
