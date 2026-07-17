"""Read-only parsing for one-way imports from the removed pilot registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class LegacyPilotSource(BaseModel):
    """The small subset of a legacy source record safe to preserve."""

    model_config = ConfigDict(extra="ignore")

    source_asset_id: UUID
    source_url: str
    submitted_by: str = Field(default="unknown", min_length=1, max_length=100)
    collection_method: str = Field(default="legacy-pilot-import", min_length=1, max_length=100)
    creator: str | None = None
    publisher: str | None = None
    duration_s: float = Field(default=0.0, ge=0)
    checksum_sha256: str = ""
    added_at: str | None = None


def load_legacy_pilot_sources(
    pilot_dir: str | Path,
) -> tuple[tuple[LegacyPilotSource, ...], int]:
    """Parse legacy source rows, counting malformed rows without trusting their decisions."""

    source_file = Path(pilot_dir).expanduser().resolve() / "sources.jsonl"
    if not source_file.is_file():
        raise ValueError(f"legacy pilot source file does not exist: {source_file}")
    sources: list[LegacyPilotSource] = []
    invalid = 0
    for line in source_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("legacy source row must be an object")
            sources.append(LegacyPilotSource.model_validate(payload))
        except (json.JSONDecodeError, ValueError):
            invalid += 1
    return tuple(sources), invalid
