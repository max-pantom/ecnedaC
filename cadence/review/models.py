"""Versioned contracts for human decisions and private audit history."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, TypeAlias
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

ReviewEntityType: TypeAlias = Literal["source", "segment", "dataset"]
ReviewAction: TypeAlias = Literal[
    "rights_updated",
    "source_approval_updated",
    "download_approval_updated",
    "training_eligibility_updated",
    "segment_approval_updated",
    "dataset_built",
]
ReviewStage: TypeAlias = Literal[
    "rights",
    "source_approval",
    "download_approval",
    "training_eligibility",
    "segment_approval",
]
RightsStatusValue: TypeAlias = Literal[
    "verified_permitted",
    "user_owned",
    "licensed",
    "unverified",
    "restricted",
    "rejected",
    "revoked",
    "expired",
]
ReviewDecisionValue: TypeAlias = Literal[
    "pending",
    "approved",
    "rejected",
    "eligible",
    "ineligible",
]
ReviewStateValue: TypeAlias = str | bool | int | float | None


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvidenceReference(StrictReviewModel):
    """Opaque pointer to evidence held outside Git and ordinary application logs."""

    schema_version: Literal["0.1.0"] = "0.1.0"
    reference: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=500)


class RightsDecision(StrictReviewModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    status: RightsStatusValue
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)
    evidence_reference: EvidenceReference | None = None
    expected_revision: int = Field(ge=0)
    decided_at: datetime = Field(default_factory=utc_now)


class ReviewDecision(StrictReviewModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    decision: ReviewDecisionValue
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)
    evidence_reference: EvidenceReference | None = None
    expected_revision: int = Field(ge=0)
    decided_at: datetime = Field(default_factory=utc_now)


class AuditEvent(StrictReviewModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    event_id: UUID = Field(default_factory=uuid4)
    entity_type: ReviewEntityType
    entity_id: UUID
    action: ReviewAction
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)
    evidence_reference: EvidenceReference | None = None
    occurred_at: datetime = Field(default_factory=utc_now)
    revision: int = Field(gt=0)
    prior_state: dict[str, ReviewStateValue]
    new_state: dict[str, ReviewStateValue]


class ReviewQueueItem(StrictReviewModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    entity_type: ReviewEntityType
    entity_id: UUID
    source_id: UUID
    revision: int = Field(ge=0)
    stage: ReviewStage
    status: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    submitted_at: datetime


class RecordRevision(StrictReviewModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    entity_type: ReviewEntityType
    entity_id: UUID
    revision: int = Field(ge=0)


class StaleRevisionError(RuntimeError):
    """Raised inside the registry lock when a review was based on stale state."""

    def __init__(
        self,
        entity_type: ReviewEntityType,
        entity_id: UUID,
        *,
        expected_revision: int,
        actual_revision: int,
    ) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"stale {entity_type} revision for {entity_id}: "
            f"expected {expected_revision}, actual {actual_revision}"
        )
