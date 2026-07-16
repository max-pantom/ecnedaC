"""Dependency-light reproducibility hashes and Git identity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        return hashlib.sha256(b"missing").hexdigest()
    return hashlib.sha256(target.read_bytes()).hexdigest()


def git_commit(repo_root: str | Path = ".") -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"

