"""Replaceable Cadence working-storage interfaces."""

from cadence.storage.base import LocalFilesystemStorage, StorageLimitError

__all__ = ["LocalFilesystemStorage", "StorageLimitError"]
