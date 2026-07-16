"""Versioned, provenance-bearing contrastive dataset manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

MANIFEST_SCHEMA_VERSION = "0.1.0"


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    asset_id: UUID
    source_asset_id: UUID
    path: Path | None = None
    storage_uri: str | None = None
    duration_s: float = Field(gt=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    audio_sample_rate: int = Field(gt=0)
    has_video: bool
    has_audio: bool
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: AnyHttpUrl
    license_status: str = Field(min_length=1, max_length=100)
    collection_method: str = Field(min_length=1, max_length=100)
    split: Literal["train", "validation", "test"]
    eligible_for_contrastive: bool
    domain: str = "ui-micro-interactions-product-reveals"

    @model_validator(mode="after")
    def validate_locator(self) -> ManifestEntry:
        if self.path is None and self.storage_uri is None:
            raise ValueError("either path or storage_uri is required")
        return self


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_split(source_asset_id: UUID, seed: int = 1337) -> str:
    digest = hashlib.sha256(f"{seed}:{source_asset_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def validate_source_level_splits(entries: Iterable[ManifestEntry]) -> None:
    seen: dict[UUID, str] = {}
    for entry in entries:
        existing = seen.setdefault(entry.source_asset_id, entry.split)
        if existing != entry.split:
            raise ValueError(
                f"source asset {entry.source_asset_id} leaks across {existing} and {entry.split}"
            )


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    manifest_path = Path(path)
    entries: list[ManifestEntry] = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entries.append(ManifestEntry.model_validate_json(line))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest entry at line {line_number}: {exc}") from exc
    validate_source_level_splits(entries)
    return entries


def write_manifest(entries: Iterable[ManifestEntry], path: str | Path) -> None:
    materialized = list(entries)
    validate_source_level_splits(materialized)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(entry.model_dump_json() + "\n" for entry in materialized), encoding="utf-8"
    )
