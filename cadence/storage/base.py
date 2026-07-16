"""Local pilot storage with hard capacity and free-space guards."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class StorageLimitError(RuntimeError):
    """Raised before an operation would violate a configured storage limit."""


@dataclass(frozen=True)
class StorageReport:
    root: str
    working_bytes: int
    maximum_working_bytes: int
    filesystem_free_bytes: int
    minimum_free_bytes: int
    remaining_working_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class ObjectStorage(Protocol):
    def report(self) -> StorageReport: ...

    def preflight(self, additional_bytes: int) -> StorageReport: ...

    def path_for(self, *parts: str) -> Path: ...


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


class LocalFilesystemStorage:
    def __init__(
        self,
        root: str | Path,
        *,
        maximum_working_bytes: int,
        minimum_free_bytes: int,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.maximum_working_bytes = maximum_working_bytes
        self.minimum_free_bytes = minimum_free_bytes

    def report(self) -> StorageReport:
        used = directory_size(self.root)
        free = shutil.disk_usage(self.root).free
        return StorageReport(
            root=str(self.root),
            working_bytes=used,
            maximum_working_bytes=self.maximum_working_bytes,
            filesystem_free_bytes=free,
            minimum_free_bytes=self.minimum_free_bytes,
            remaining_working_bytes=max(0, self.maximum_working_bytes - used),
        )

    def preflight(self, additional_bytes: int) -> StorageReport:
        if additional_bytes < 0:
            raise ValueError("additional_bytes must be non-negative")
        report = self.report()
        if report.working_bytes + additional_bytes > self.maximum_working_bytes:
            raise StorageLimitError(
                "operation would exceed maximum Cadence working storage "
                f"({self.maximum_working_bytes} bytes)"
            )
        if report.filesystem_free_bytes - additional_bytes < self.minimum_free_bytes:
            raise StorageLimitError(
                "operation would reduce VPS free disk below minimum "
                f"({self.minimum_free_bytes} bytes)"
            )
        return report

    def path_for(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError("storage path escapes the configured root")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class S3CompatibleStorage:
    """Future interface placeholder; credentials are intentionally not required yet."""

    def __init__(self, endpoint: str, bucket: str) -> None:
        self.endpoint = endpoint
        self.bucket = bucket

    def report(self) -> StorageReport:
        raise NotImplementedError("S3-compatible storage is not enabled in the pilot milestone")

    def preflight(self, additional_bytes: int) -> StorageReport:
        raise NotImplementedError("S3-compatible storage is not enabled in the pilot milestone")

    def path_for(self, *parts: str) -> Path:
        raise NotImplementedError("S3-compatible storage does not expose local paths")

