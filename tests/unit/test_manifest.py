from uuid import uuid4

import pytest
from pydantic import ValidationError

from cadence.ingestion.manifest import (
    ManifestEntry,
    deterministic_split,
    load_manifest,
    validate_source_level_splits,
    write_manifest,
)


def make_entry(
    path: str, *, source_id: object | None = None, split: str = "train"
) -> ManifestEntry:
    asset = uuid4()
    return ManifestEntry.model_validate({
        "asset_id": asset,
        "source_asset_id": source_id or asset,
        "path": path,
        "duration_s": 1.0,
        "fps": 8,
        "width": 32,
        "height": 32,
        "audio_sample_rate": 8000,
        "has_video": True,
        "has_audio": True,
        "checksum_sha256": "a" * 64,
        "source_url": "https://fixtures.cadence.invalid/example.mp4",
        "license_status": "synthetic-generated",
        "collection_method": "unit-test",
        "split": split,
        "eligible_for_contrastive": True,
    })


def test_manifest_round_trip(tmp_path: object) -> None:
    path = tmp_path / "manifest.jsonl"  # type: ignore[operator]
    entry = make_entry("/tmp/example.mp4")
    write_manifest([entry], path)
    assert load_manifest(path) == [entry]


def test_source_split_leak_rejected() -> None:
    source = uuid4()
    with pytest.raises(ValueError, match="leaks"):
        validate_source_level_splits([
            make_entry("/tmp/a", source_id=source, split="train"),
            make_entry("/tmp/b", source_id=source, split="validation"),
        ])


def test_deterministic_split() -> None:
    source = uuid4()
    assert deterministic_split(source, 7) == deterministic_split(source, 7)


def test_provenance_is_required() -> None:
    values = make_entry("/tmp/a").model_dump()
    del values["license_status"]
    with pytest.raises(ValidationError):
        ManifestEntry.model_validate(values)
