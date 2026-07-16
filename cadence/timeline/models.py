"""Canonical Creative Timeline models."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TIMELINE_SCHEMA_VERSION = "0.1.0"


class TimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    t_ms: int = Field(ge=0)
    type: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    duration_ms: int = Field(ge=0)
    intensity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=500)
    texture: str | None = Field(default=None, max_length=100)
    pan: float | None = Field(default=None, ge=-1.0, le=1.0)
    reverb: str | None = Field(default=None, max_length=100)

    @field_validator("type", "rationale", "texture", "reverb")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else value


class CreativeTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timeline_id: UUID
    source_video: str = Field(min_length=1, max_length=255)
    duration_ms: int = Field(gt=0)
    events: tuple[TimelineEvent, ...]
    schema_version: Literal["0.1.0"]

    @field_validator("source_video")
    @classmethod
    def strip_source_video(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_event_timing(self) -> CreativeTimeline:
        previous = -1
        for event in self.events:
            if event.t_ms < previous:
                raise ValueError("events must be ordered by t_ms")
            if event.t_ms + event.duration_ms > self.duration_ms:
                raise ValueError("event extends beyond timeline duration")
            previous = event.t_ms
        return self


def timeline_json_schema() -> dict[str, Any]:
    """Return the canonical checked-in JSON Schema representation."""
    schema = CreativeTimeline.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://cadence.local/schema/timeline.schema.json"
    return schema
