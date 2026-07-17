import stat
from pathlib import Path

import pytest

from cadence.storage.base import LocalFilesystemStorage, StorageLimitError


def test_working_storage_limit_rejects_before_write(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(
        tmp_path, maximum_working_bytes=10, minimum_free_bytes=0
    )
    (tmp_path / "existing.bin").write_bytes(b"12345678")
    with pytest.raises(StorageLimitError, match="maximum Cadence working storage"):
        storage.preflight(3)


def test_minimum_free_space_rejects_before_write(tmp_path: Path) -> None:
    free = __import__("shutil").disk_usage(tmp_path).free
    storage = LocalFilesystemStorage(
        tmp_path, maximum_working_bytes=free * 2, minimum_free_bytes=free + 1
    )
    with pytest.raises(StorageLimitError, match="free disk below minimum"):
        storage.preflight(0)


def test_storage_path_cannot_escape_root(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(
        tmp_path, maximum_working_bytes=1000, minimum_free_bytes=0
    )
    with pytest.raises(ValueError, match="escapes"):
        storage.path_for("..", "outside")


def test_storage_directories_are_owner_only(tmp_path: Path) -> None:
    root = tmp_path / "private"
    storage = LocalFilesystemStorage(
        root, maximum_working_bytes=1000, minimum_free_bytes=0
    )
    target = storage.path_for("sources", "normalized", "sample.mp4")

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.parent.parent.stat().st_mode) == 0o700
