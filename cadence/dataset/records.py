"""Versioned persistent records for the dataset-intake workflow."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from cadence.review.models import AuditEvent


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source URL must be an absolute HTTP or HTTPS URL")
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


class RightsStatus(StrEnum):
    VERIFIED_PERMITTED = "verified_permitted"
    USER_OWNED = "user_owned"
    LICENSED = "licensed"
    UNVERIFIED = "unverified"
    RESTRICTED = "restricted"
    REJECTED = "rejected"
    REVOKED = "revoked"
    EXPIRED = "expired"


PERMITTED_RIGHTS = {
    RightsStatus.VERIFIED_PERMITTED,
    RightsStatus.USER_OWNED,
    RightsStatus.LICENSED,
}


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class InspectionStatus(StrEnum):
    NOT_INSPECTED = "not_inspected"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class DownloadStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    NORMALIZED = "normalized"
    DUPLICATE = "duplicate"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    INSPECTED = "inspected"
    NORMALIZED = "normalized"
    SEGMENTS_SUGGESTED = "segments_suggested"
    FAILED = "failed"


class SegmentCategory(StrEnum):
    OPENING_BUILDUP = "opening buildup"
    PRODUCT_REVEAL = "product reveal"
    FEATURE_MONTAGE = "feature montage"
    INTERFACE_REVEAL = "interface reveal"
    DEVICE_MOVEMENT = "device movement"
    KINETIC_TYPOGRAPHY = "kinetic typography"
    LOGO_RESOLUTION = "logo resolution"
    BRAND_LOCKUP = "brand lockup"
    TRANSITION = "transition"
    DELIBERATE_SILENCE = "deliberate silence"
    CINEMATIC_PRODUCT_SHOT = "cinematic product shot"


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    revision: int = Field(default=0, ge=0)
    source_id: UUID = Field(default_factory=uuid4)
    url: AnyHttpUrl
    canonical_url: str
    submitted_by: str = Field(min_length=1, max_length=100)
    collection_method: str = Field(default="user-submitted-url", min_length=1, max_length=100)
    submitted_at: datetime = Field(default_factory=utc_now)
    title: str | None = None
    publisher_or_creator: str | None = None
    platform: str | None = None
    duration_seconds: float | None = Field(default=None, gt=0)
    content_length_bytes: int | None = Field(default=None, ge=0)
    inspection_status: InspectionStatus = InspectionStatus.NOT_INSPECTED
    download_status: DownloadStatus = DownloadStatus.NOT_REQUESTED
    download_method: str | None = None
    storage_path: Path | None = None
    storage_uri: str | None = None
    normalized_path: Path | None = None
    normalized_uri: str | None = None
    checksum_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    duplicate_of_source_id: UUID | None = None
    rights_status: RightsStatus = RightsStatus.UNVERIFIED
    license_notes: str = Field(default="", max_length=1000)
    source_approval: ApprovalStatus = ApprovalStatus.PENDING
    download_approval: ApprovalStatus = ApprovalStatus.PENDING
    eligible_for_training: bool = False
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_state: str | None = None
    estimated_useful_segment_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def enforce_training_rights(self) -> SourceRecord:
        if self.eligible_for_training and self.rights_status not in PERMITTED_RIGHTS:
            raise ValueError("training eligibility requires verified, owned, or licensed rights")
        return self

    @classmethod
    def from_submission(
        cls,
        url: str,
        submitted_by: str,
        *,
        collection_method: str = "user-submitted-url",
    ) -> SourceRecord:
        canonical = canonicalize_url(url)
        return cls.model_validate(
            {
                "url": canonical,
                "canonical_url": canonical,
                "submitted_by": submitted_by,
                "collection_method": collection_method,
            }
        )


class SegmentCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    revision: int = Field(default=0, ge=0)
    segment_id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    motion_score: float = Field(ge=0, le=1)
    audio_activity_score: float = Field(ge=0, le=1)
    scene_boundary: bool
    scene_boundary_seconds: tuple[float, ...] = ()
    reason: str = Field(min_length=1, max_length=500)
    categories: tuple[SegmentCategory, ...] = ()
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    extracted_path: Path
    checksum_after_extraction: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_duration(self) -> SegmentCandidate:
        if abs((self.end_seconds - self.start_seconds) - self.duration_seconds) > 1e-3:
            raise ValueError("segment duration must equal end minus start")
        return self


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    dataset_id: UUID = Field(default_factory=uuid4)
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    version: int = Field(gt=0)
    created_at: datetime = Field(default_factory=utc_now)
    manifest_path: Path
    report_path: Path
    segment_ids: tuple[UUID, ...]
    source_ids: tuple[UUID, ...]
    total_bytes: int = Field(ge=0)
    split_counts: dict[str, int]


class RegistryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = "0.1.0"
    sources: dict[str, SourceRecord] = Field(default_factory=dict)
    segments: dict[str, SegmentCandidate] = Field(default_factory=dict)
    datasets: dict[str, DatasetRecord] = Field(default_factory=dict)
    audit_events: tuple[AuditEvent, ...] = ()


class IntakeRegistry:
    """Atomic single-file registry suitable for a one-worker pilot VPS."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.root.chmod(0o700)
        self.path = self.root / "registry.json"
        self.lock_path = self.root / ".registry.lock"

    def load(self) -> RegistryState:
        if not self.path.exists():
            return RegistryState()
        return RegistryState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, state: RegistryState) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        self.path.chmod(0o600)

    def mutate(self, callback: Callable[[RegistryState], None]) -> RegistryState:
        with self.lock_path.open("a+") as lock:
            self.lock_path.chmod(0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = self.load()
            callback(state)
            self.save(state)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return state

    def get_source(self, source_id: UUID | str) -> SourceRecord:
        key = str(source_id)
        source = self.load().sources.get(key)
        if source is None:
            raise KeyError(f"unknown source ID: {source_id}")
        return source

    def get_segment(self, segment_id: UUID | str) -> SegmentCandidate:
        key = str(segment_id)
        segment = self.load().segments.get(key)
        if segment is None:
            raise KeyError(f"unknown segment ID: {segment_id}")
        return segment
