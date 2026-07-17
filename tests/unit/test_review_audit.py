import json
import stat
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cadence.common.config import load_config
from cadence.dataset.downloaders import DownloaderChain
from cadence.dataset.media import MediaMetadata
from cadence.dataset.records import (
    ApprovalStatus,
    DownloadStatus,
    InspectionStatus,
    IntakeRegistry,
    RegistryState,
    RightsStatus,
    SegmentCandidate,
    SourceRecord,
)
from cadence.dataset.service import DatasetIntakeService
from cadence.review.models import (
    EvidenceReference,
    RecordRevision,
    ReviewDecision,
    RightsDecision,
    StaleRevisionError,
)


class UnusedMedia:
    def probe(self, path: Path) -> MediaMetadata:
        raise AssertionError("media should not be read by review tests")

    def normalize(self, source: Path, destination: Path) -> MediaMetadata:
        raise AssertionError("media should not be normalized by review tests")

    def extract_segment(
        self, source: Path, destination: Path, start_seconds: float, duration_seconds: float
    ) -> MediaMetadata:
        raise AssertionError("segments should not be extracted by review tests")


def make_service(tmp_path: Path) -> DatasetIntakeService:
    base = load_config("configs/test.yaml")
    config = base.model_copy(
        update={"paths": base.paths.model_copy(update={"intake_root": tmp_path / "intake"})}
    )
    return DatasetIntakeService(
        config,
        downloaders=DownloaderChain([]),
        media=UnusedMedia(),
    )


def test_review_models_are_versioned_strict_contracts() -> None:
    evidence = EvidenceReference(reference="LIC-2026-014", description="Private contract index")
    rights = RightsDecision(
        status="licensed",
        actor="reviewer",
        reason="License covers model training",
        evidence_reference=evidence,
        expected_revision=3,
    )
    decision = ReviewDecision(
        decision="approved",
        actor="reviewer",
        reason="Relevant synchronized launch sequence",
        expected_revision=4,
    )
    revision = RecordRevision(entity_type="source", entity_id=uuid4(), revision=4)

    assert rights.schema_version == decision.schema_version == revision.schema_version == "0.1.0"
    with pytest.raises(ValidationError):
        EvidenceReference.model_validate({"reference": "LIC-1", "private_text": "forbidden"})


def test_old_registry_without_revisions_or_audit_history_loads(tmp_path: Path) -> None:
    registry = IntakeRegistry(tmp_path / "intake")
    source = SourceRecord.from_submission("https://example.com/old.mp4", "operator")
    segment = SegmentCandidate(
        source_id=source.source_id,
        start_seconds=0,
        end_seconds=4,
        duration_seconds=4,
        motion_score=0.5,
        audio_activity_score=0.5,
        scene_boundary=False,
        reason="Legacy candidate",
        extracted_path=tmp_path / "segment.mp4",
        checksum_after_extraction="a" * 64,
    )
    registry.path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "sources": {
                    str(source.source_id): source.model_dump(mode="json", exclude={"revision"})
                },
                "segments": {
                    str(segment.segment_id): segment.model_dump(
                        mode="json", exclude={"revision"}
                    )
                },
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = registry.load()

    assert loaded.sources[str(source.source_id)].revision == 0
    assert loaded.segments[str(segment.segment_id)].revision == 0
    assert loaded.audit_events == ()


def test_source_review_is_atomic_versioned_and_audited(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    source, _ = service.add_source("https://example.com/review.mp4", submitted_by="operator")
    evidence = EvidenceReference(reference="OWN-42")

    updated = service.set_rights(
        source.source_id,
        RightsStatus.USER_OWNED,
        license_notes="Owner confirmation is held privately",
        actor="aven",
        reason="Owner confirmed training permission",
        evidence_reference=evidence,
        expected_revision=0,
    )

    assert updated.revision == 1
    events = service.list_audit_events(entity_type="source", entity_id=source.source_id)
    assert len(events) == 1
    event = events[0]
    assert event.actor == "aven"
    assert event.action == "rights_updated"
    assert event.reason == "Owner confirmed training permission"
    assert event.evidence_reference == evidence
    assert event.revision == 1
    assert event.prior_state["rights_status"] == "unverified"
    assert event.new_state["rights_status"] == "user_owned"

    with pytest.raises(StaleRevisionError) as raised:
        service.set_source_approval(
            source.source_id,
            ApprovalStatus.APPROVED,
            actor="aven",
            reason="Relevant source",
            expected_revision=0,
        )
    assert raised.value.expected_revision == 0
    assert raised.value.actual_revision == 1
    assert service.registry.get_source(source.source_id).revision == 1
    assert len(service.list_audit_events()) == 1


def test_rights_downgrade_immediately_revokes_training_eligibility(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    source, _ = service.add_source("https://example.com/licensed.mp4", submitted_by="operator")
    eligible = source.model_copy(
        update={
            "revision": 7,
            "rights_status": RightsStatus.LICENSED,
            "source_approval": ApprovalStatus.APPROVED,
            "download_approval": ApprovalStatus.APPROVED,
            "download_status": DownloadStatus.NORMALIZED,
            "eligible_for_training": True,
        }
    )

    def seed(state: RegistryState) -> None:
        state.sources[str(source.source_id)] = eligible

    service.registry.mutate(seed)
    downgraded = service.set_rights(
        source.source_id,
        RightsStatus.UNVERIFIED,
        license_notes="Review reopened",
        actor="aven",
        reason="License scope is now uncertain",
        expected_revision=7,
    )

    assert downgraded.revision == 8
    assert downgraded.eligible_for_training is False
    event = service.list_audit_events(entity_id=source.source_id)[0]
    assert event.prior_state["eligible_for_training"] is True
    assert event.new_state["eligible_for_training"] is False


@pytest.mark.parametrize(
    "status",
    [
        RightsStatus.UNVERIFIED,
        RightsStatus.RESTRICTED,
        RightsStatus.REJECTED,
        RightsStatus.REVOKED,
        RightsStatus.EXPIRED,
    ],
)
def test_every_prohibited_rights_state_revokes_eligibility_and_download(
    tmp_path: Path, status: RightsStatus
) -> None:
    service = make_service(tmp_path)
    source, _ = service.add_source("https://example.com/revoke.mp4", submitted_by="operator")
    eligible = source.model_copy(
        update={
            "rights_status": RightsStatus.LICENSED,
            "source_approval": ApprovalStatus.APPROVED,
            "download_approval": ApprovalStatus.APPROVED,
            "download_status": DownloadStatus.NORMALIZED,
            "eligible_for_training": True,
        }
    )

    def seed(state: RegistryState) -> None:
        state.sources[str(source.source_id)] = eligible

    service.registry.mutate(seed)
    updated = service.set_rights(
        source.source_id,
        status,
        license_notes="Non-sensitive status note",
        actor="reviewer",
        reason=f"Rights are now {status.value}",
        expected_revision=0,
    )

    assert updated.eligible_for_training is False
    if status in {
        RightsStatus.RESTRICTED,
        RightsStatus.REJECTED,
        RightsStatus.REVOKED,
        RightsStatus.EXPIRED,
    }:
        assert updated.download_approval == ApprovalStatus.REJECTED


def test_private_registry_uses_owner_only_permissions(tmp_path: Path) -> None:
    registry = IntakeRegistry(tmp_path / "intake")
    source = SourceRecord.from_submission("https://example.com/private.mp4", "operator")

    def seed(state: RegistryState) -> None:
        state.sources[str(source.source_id)] = source

    registry.mutate(seed)

    assert stat.S_IMODE(registry.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(registry.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(registry.lock_path.stat().st_mode) == 0o600


def test_source_approval_download_and_eligibility_each_append_an_event(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    source, _ = service.add_source("https://example.com/lifecycle.mp4", submitted_by="operator")
    prepared = source.model_copy(
        update={
            "rights_status": RightsStatus.LICENSED,
            "inspection_status": InspectionStatus.SUPPORTED,
            "download_status": DownloadStatus.NORMALIZED,
        }
    )

    def seed(state: RegistryState) -> None:
        state.sources[str(source.source_id)] = prepared

    service.registry.mutate(seed)
    source_approved = service.set_source_approval(
        source.source_id,
        ApprovalStatus.APPROVED,
        actor="aven",
        reason="Source is relevant",
        expected_revision=0,
    )
    download_approved = service.set_download_approval(
        source.source_id,
        ApprovalStatus.APPROVED,
        actor="aven",
        reason="Acquisition is authorized",
        expected_revision=source_approved.revision,
    )
    eligible = service.set_training_eligibility(
        source.source_id,
        True,
        actor="aven",
        reason="All training gates are satisfied",
        expected_revision=download_approved.revision,
    )
    revoked = service.set_download_approval(
        source.source_id,
        ApprovalStatus.REJECTED,
        actor="aven",
        reason="Download authorization withdrawn",
        expected_revision=eligible.revision,
    )

    assert revoked.revision == 4
    assert revoked.eligible_for_training is False
    assert [event.action for event in service.list_audit_events()] == [
        "source_approval_updated",
        "download_approval_updated",
        "training_eligibility_updated",
        "download_approval_updated",
    ]


def test_review_queue_and_segment_audit_support_http_read_layer(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    source, _ = service.add_source("https://example.com/queue.mp4", submitted_by="operator")
    initial_queue = service.review_queue()
    assert [(item.entity_id, item.stage) for item in initial_queue] == [
        (source.source_id, "rights")
    ]

    reviewed_source = source.model_copy(
        update={
            "rights_status": RightsStatus.REJECTED,
            "source_approval": ApprovalStatus.REJECTED,
        }
    )
    segment = SegmentCandidate(
        source_id=source.source_id,
        start_seconds=1,
        end_seconds=5,
        duration_seconds=4,
        motion_score=0.8,
        audio_activity_score=0.7,
        scene_boundary=True,
        reason="Motion and audio boundary",
        extracted_path=tmp_path / "candidate.mp4",
        checksum_after_extraction="b" * 64,
    )

    def seed(state: RegistryState) -> None:
        state.sources[str(source.source_id)] = reviewed_source
        state.segments[str(segment.segment_id)] = segment

    service.registry.mutate(seed)
    queue = service.review_queue()
    assert len(queue) == 1
    assert queue[0].entity_id == segment.segment_id
    assert queue[0].stage == "segment_approval"

    approved = service.set_segment_approval(
        segment.segment_id,
        ApprovalStatus.APPROVED,
        actor="aven",
        reason="Useful synchronized event",
        expected_revision=0,
    )
    assert approved.revision == 1
    assert service.review_queue() == []
    events = service.list_audit_events(entity_type="segment")
    assert len(events) == 1
    assert events[0].action == "segment_approval_updated"


def test_old_cli_review_calls_default_to_cli_actor(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    source, _ = service.add_source("https://example.com/cli.mp4", submitted_by="operator")

    updated = service.set_source_approval(source.source_id, ApprovalStatus.REJECTED)

    assert updated.revision == 1
    assert service.list_audit_events()[0].actor == "cli"
