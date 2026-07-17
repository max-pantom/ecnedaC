#!/usr/bin/env python3
"""Verify the locked GPU packages and official CPython 3.12 CUDA wheels without installing."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote

CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
PACKAGES = ("torch", "torchaudio")
PYTHON_TAG = "cp312-cp312"
PLATFORM_TAG = "manylinux_2_28_x86_64"


def _gpu_versions(project: dict[str, object]) -> dict[str, str]:
    groups = project.get("dependency-groups")
    if not isinstance(groups, dict):
        raise ValueError("dependency-groups table is missing")
    requirements = groups.get("training-gpu")
    if not isinstance(requirements, list):
        raise ValueError("training-gpu dependency group is missing")
    versions: dict[str, str] = {}
    for value in requirements:
        if not isinstance(value, str):
            continue
        decoded = unquote(value)
        match = re.match(
            r"^(torch|torchaudio) @ https://download-r2\.pytorch\.org/"
            r"whl/cu126/[^-]+-(\d+\.\d+\.\d+)\+cu126-",
            decoded,
        )
        if match:
            versions[match.group(1)] = match.group(2)
    if set(versions) != set(PACKAGES):
        raise ValueError("training-gpu must pin torch and torchaudio exactly")
    if len(set(versions.values())) != 1:
        raise ValueError("torch and torchaudio GPU versions must match")
    return versions


def _verify_lock(lock: dict[str, object], versions: dict[str, str]) -> None:
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock has no package records")
    for name, version in versions.items():
        matching = [
            item
            for item in packages
            if isinstance(item, dict)
            and item.get("name") == name
            and item.get("version") in {version, f"{version}+cu126"}
        ]
        if not matching:
            raise ValueError(f"uv.lock does not contain {name}=={version}")
        if not any(
            isinstance(item.get("source"), dict)
            and str(item["source"].get("url", "")).startswith(
                "https://download-r2.pytorch.org/whl/cu126/"
            )
            for item in matching
        ):
            raise ValueError(
                f"uv.lock does not resolve {name}=={version} from the official CUDA index"
            )


def _verify_official_indexes(versions: dict[str, str]) -> None:
    for name, version in versions.items():
        response = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                f"{CUDA_INDEX}/{name}/",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        page = unquote(response.stdout)
        expected = f"{name}-{version}+cu126-{PYTHON_TAG}-{PLATFORM_TAG}.whl"
        if expected not in page:
            raise ValueError(f"official CUDA index does not publish {expected}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        versions = _gpu_versions(tomllib.load(handle))
    with (root / "uv.lock").open("rb") as handle:
        _verify_lock(tomllib.load(handle), versions)
    _verify_official_indexes(versions)
    print(
        "verified torch and torchaudio "
        f"{next(iter(versions.values()))} CPython 3.12 Linux CUDA 12.6 wheels"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
