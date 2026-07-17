"""Dataset intake orchestration with explicit rights and storage gates."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from cadence.common.config import CadenceConfig
from cadence.dataset.downloaders import DownloaderChain
from cadence.dataset.legacy import load_legacy_pilot_sources
from cadence.dataset.media import MediaProcessor
from cadence.dataset.records import (
    PERMITTED_RIGHTS,
    ApprovalStatus,
    DatasetRecord,
    DownloadStatus,
    InspectionStatus,
    IntakeRegistry,
    ProcessingStatus,
    RegistryState,
    RightsStatus,
    SegmentCandidate,
    SourceRecord,
)
from cadence.ingestion.manifest import (
    ManifestEntry,
    deterministic_split,
    sha256_file,
    write_manifest,
)
from cadence.review.models import (
    AuditEvent,
    EvidenceReference,
    ReviewAction,
    ReviewEntityType,
    ReviewQueueItem,
    ReviewStage,
    ReviewStateValue,
    StaleRevisionError,
)
from cadence.storage.base import LocalFilesystemStorage

GIB = 1024**3


class DatasetIntakeService:
    def __init__(
        self,
        config: CadenceConfig,
        *,
        downloaders: DownloaderChain,
        media: MediaProcessor,
        registry: IntakeRegistry | None = None,
        storage: LocalFilesystemStorage | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or IntakeRegistry(config.paths.intake_root)
        self.storage = storage or LocalFilesystemStorage(
            config.paths.intake_root,
            maximum_working_bytes=round(config.dataset_intake.maximum_working_storage_gb * GIB),
            minimum_free_bytes=round(config.dataset_intake.minimum_free_disk_gb * GIB),
        )
        self.downloaders = downloaders
        self.media = media

    def add_source(
        self,
        url: str,
        *,
        submitted_by: str,
        collection_method: str = "user-submitted-url",
    ) -> tuple[SourceRecord, bool]:
        candidate = SourceRecord.from_submission(
            url,
            submitted_by,
            collection_method=collection_method,
        )
        existing = next(
            (
                source
                for source in self.registry.load().sources.values()
                if source.canonical_url == candidate.canonical_url
            ),
            None,
        )
        if existing is not None:
            return existing, False

        def add(state: RegistryState) -> None:
            state.sources[str(candidate.source_id)] = candidate

        self.registry.mutate(add)
        return candidate, True

    def import_legacy_pilot(
        self,
        pilot_dir: str | Path,
        *,
        submitted_by: str,
        execute: bool = False,
    ) -> dict[str, object]:
        """Import legacy source identities into quarantine; never trust old approvals or media."""

        legacy_sources, invalid_rows = load_legacy_pilot_sources(pilot_dir)
        added = 0
        would_add = 0
        duplicates = 0
        invalid = invalid_rows

        def import_sources(state: RegistryState) -> None:
            nonlocal added, duplicates, invalid, would_add
            existing_urls = {source.canonical_url for source in state.sources.values()}
            for legacy in legacy_sources:
                try:
                    candidate = SourceRecord.from_submission(
                        legacy.source_url,
                        f"{legacy.submitted_by} (migrated by {submitted_by})"[:100],
                        collection_method=(
                            f"legacy-pilot:{legacy.collection_method}"
                        )[:100],
                    )
                except ValueError:
                    invalid += 1
                    continue
                if (
                    candidate.canonical_url in existing_urls
                    or str(legacy.source_asset_id) in state.sources
                ):
                    duplicates += 1
                    continue
                publisher = legacy.creator or legacy.publisher
                updates: dict[str, object] = {
                    "source_id": legacy.source_asset_id,
                    "publisher_or_creator": publisher,
                    "license_notes": (
                        "Imported from the retired pilot registry; rights and approvals "
                        "must be reviewed again."
                    ),
                }
                if legacy.duration_s > 0:
                    updates["duration_seconds"] = legacy.duration_s
                if re.fullmatch(r"[0-9a-f]{64}", legacy.checksum_sha256):
                    updates["checksum_sha256"] = legacy.checksum_sha256
                candidate = candidate.model_copy(update=updates)
                existing_urls.add(candidate.canonical_url)
                if execute:
                    state.sources[str(candidate.source_id)] = candidate
                    added += 1
                else:
                    would_add += 1

        if execute:
            self.registry.mutate(import_sources)
        else:
            import_sources(self.registry.load())
        return {
            "executed": execute,
            "discovered": len(legacy_sources),
            "added": added,
            "would_add": would_add,
            "duplicates": duplicates,
            "invalid": invalid,
            "rights_status": RightsStatus.UNVERIFIED.value,
            "eligible_for_training": False,
        }

    def add_batch(self, path: str | Path, *, submitted_by: str) -> dict[str, int]:
        added = duplicates = invalid = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                _, created = self.add_source(value, submitted_by=submitted_by)
                added += int(created)
                duplicates += int(not created)
            except ValueError:
                invalid += 1
        return {"added": added, "duplicates": duplicates, "invalid": invalid}

    def list_sources(self) -> list[SourceRecord]:
        return sorted(self.registry.load().sources.values(), key=lambda source: source.submitted_at)

    def list_datasets(self) -> list[DatasetRecord]:
        return sorted(
            self.registry.load().datasets.values(),
            key=lambda dataset: (dataset.name, dataset.version),
            reverse=True,
        )

    def latest_dataset(self, name: str) -> DatasetRecord:
        matches = [dataset for dataset in self.list_datasets() if dataset.name == name]
        if not matches:
            raise KeyError(f"unknown dataset: {name}")
        return max(matches, key=lambda dataset: dataset.version)

    def review_queue(self) -> list[ReviewQueueItem]:
        """Return the next human decision required for each reviewable record."""

        state = self.registry.load()
        queue: list[ReviewQueueItem] = []
        for source in state.sources.values():
            stage: ReviewStage | None = None
            status: str | None = None
            if source.rights_status == RightsStatus.UNVERIFIED:
                stage = "rights"
                status = source.rights_status.value
            elif source.source_approval == ApprovalStatus.PENDING:
                stage = "source_approval"
                status = source.source_approval.value
            elif (
                source.source_approval == ApprovalStatus.APPROVED
                and source.inspection_status == InspectionStatus.SUPPORTED
                and source.download_approval == ApprovalStatus.PENDING
            ):
                stage = "download_approval"
                status = source.download_approval.value
            elif (
                source.download_status in {DownloadStatus.NORMALIZED, DownloadStatus.DUPLICATE}
                and source.rights_status in PERMITTED_RIGHTS
                and source.source_approval == ApprovalStatus.APPROVED
                and source.download_approval == ApprovalStatus.APPROVED
                and not source.eligible_for_training
            ):
                stage = "training_eligibility"
                status = "ineligible"
            if stage is not None and status is not None:
                queue.append(
                    ReviewQueueItem(
                        entity_type="source",
                        entity_id=source.source_id,
                        source_id=source.source_id,
                        revision=source.revision,
                        stage=stage,
                        status=status,
                        title=(source.title or str(source.url))[:500],
                        submitted_at=source.submitted_at,
                    )
                )
        for segment in state.segments.values():
            if segment.approval_status == ApprovalStatus.PENDING:
                queue.append(
                    ReviewQueueItem(
                        entity_type="segment",
                        entity_id=segment.segment_id,
                        source_id=segment.source_id,
                        revision=segment.revision,
                        stage="segment_approval",
                        status=segment.approval_status.value,
                        title=(
                            f"Segment {segment.start_seconds:.2f}-"
                            f"{segment.end_seconds:.2f} seconds"
                        ),
                        submitted_at=segment.created_at,
                    )
                )
        return sorted(queue, key=lambda item: (item.submitted_at, str(item.entity_id)))

    def list_audit_events(
        self,
        *,
        entity_type: ReviewEntityType | None = None,
        entity_id: UUID | str | None = None,
    ) -> list[AuditEvent]:
        """Read private audit history without exposing registry mutation access."""

        wanted_id = UUID(str(entity_id)) if entity_id is not None else None
        return [
            event
            for event in self.registry.load().audit_events
            if (entity_type is None or event.entity_type == entity_type)
            and (wanted_id is None or event.entity_id == wanted_id)
        ]

    def inspect_source(self, source_id: UUID | str) -> SourceRecord:
        source = self.registry.get_source(source_id)
        adapter, inspection = self.downloaders.inspect(str(source.url))
        if not inspection.supported or adapter is None:
            updated = source.model_copy(
                update={
                    "revision": source.revision + 1,
                    "inspection_status": InspectionStatus.UNSUPPORTED,
                    "download_status": DownloadStatus.UNSUPPORTED,
                    "processing_status": ProcessingStatus.FAILED,
                    "error_state": inspection.error or "unsupported source",
                }
            )
        else:
            estimated = (
                max(1, round(inspection.duration_seconds / 6))
                if inspection.duration_seconds
                else 0
            )
            updated = source.model_copy(
                update={
                    "revision": source.revision + 1,
                    "title": inspection.title,
                    "publisher_or_creator": inspection.publisher_or_creator,
                    "platform": inspection.platform,
                    "duration_seconds": inspection.duration_seconds,
                    "content_length_bytes": inspection.content_length_bytes,
                    "inspection_status": InspectionStatus.SUPPORTED,
                    "download_method": adapter.name,
                    "processing_status": ProcessingStatus.INSPECTED,
                    "error_state": None,
                    "estimated_useful_segment_count": estimated,
                }
            )
        self._save_source(updated)
        return updated

    def set_source_approval(
        self,
        source_id: UUID | str,
        status: ApprovalStatus,
        *,
        actor: str = "cli",
        reason: str = "CLI source approval update",
        evidence_reference: EvidenceReference | None = None,
        expected_revision: int | None = None,
    ) -> SourceRecord:
        def update(source: SourceRecord) -> SourceRecord:
            values: dict[str, object] = {"source_approval": status}
            if status == ApprovalStatus.REJECTED:
                values.update(
                    {
                        "download_approval": ApprovalStatus.REJECTED,
                        "eligible_for_training": False,
                    }
                )
            return source.model_copy(update=values)

        return self._review_source(
            source_id,
            action="source_approval_updated",
            actor=actor,
            reason=reason,
            evidence_reference=evidence_reference,
            expected_revision=expected_revision,
            state=lambda source: {
                "source_approval": source.source_approval.value,
                "download_approval": source.download_approval.value,
                "eligible_for_training": source.eligible_for_training,
            },
            update=update,
        )

    def set_download_approval(
        self,
        source_id: UUID | str,
        status: ApprovalStatus,
        *,
        actor: str = "cli",
        reason: str = "CLI download approval update",
        evidence_reference: EvidenceReference | None = None,
        expected_revision: int | None = None,
    ) -> SourceRecord:
        def update(source: SourceRecord) -> SourceRecord:
            if status == ApprovalStatus.APPROVED:
                if source.source_approval != ApprovalStatus.APPROVED:
                    raise ValueError("source approval is required before download approval")
                if source.inspection_status != InspectionStatus.SUPPORTED:
                    raise ValueError("a supported inspection is required before download approval")
                if source.rights_status in {
                    RightsStatus.RESTRICTED,
                    RightsStatus.REJECTED,
                    RightsStatus.REVOKED,
                    RightsStatus.EXPIRED,
                }:
                    raise ValueError(
                        "prohibited rights status cannot be approved for download"
                    )
            values: dict[str, object] = {"download_approval": status}
            if status != ApprovalStatus.APPROVED:
                values["eligible_for_training"] = False
            return source.model_copy(update=values)

        return self._review_source(
            source_id,
            action="download_approval_updated",
            actor=actor,
            reason=reason,
            evidence_reference=evidence_reference,
            expected_revision=expected_revision,
            state=lambda source: {
                "download_approval": source.download_approval.value,
                "eligible_for_training": source.eligible_for_training,
            },
            update=update,
        )

    def set_rights(
        self,
        source_id: UUID | str,
        status: RightsStatus,
        *,
        license_notes: str,
        actor: str = "cli",
        reason: str | None = None,
        evidence_reference: EvidenceReference | None = None,
        expected_revision: int | None = None,
    ) -> SourceRecord:
        def update(source: SourceRecord) -> SourceRecord:
            values: dict[str, object] = {
                "rights_status": status,
                "license_notes": license_notes,
                "eligible_for_training": False,
            }
            if status in {
                RightsStatus.RESTRICTED,
                RightsStatus.REJECTED,
                RightsStatus.REVOKED,
                RightsStatus.EXPIRED,
            }:
                values["download_approval"] = ApprovalStatus.REJECTED
            return source.model_copy(update=values)

        return self._review_source(
            source_id,
            action="rights_updated",
            actor=actor,
            reason=reason or license_notes or "CLI rights update",
            evidence_reference=evidence_reference,
            expected_revision=expected_revision,
            state=lambda source: {
                "rights_status": source.rights_status.value,
                "license_notes": source.license_notes,
                "download_approval": source.download_approval.value,
                "eligible_for_training": source.eligible_for_training,
            },
            update=update,
        )

    def set_training_eligibility(
        self,
        source_id: UUID | str,
        eligible: bool,
        *,
        actor: str = "cli",
        reason: str = "CLI training eligibility update",
        evidence_reference: EvidenceReference | None = None,
        expected_revision: int | None = None,
    ) -> SourceRecord:
        def update(source: SourceRecord) -> SourceRecord:
            if eligible:
                if source.rights_status not in PERMITTED_RIGHTS:
                    raise ValueError(
                        "training eligibility requires verified, owned, or licensed rights"
                    )
                if source.source_approval != ApprovalStatus.APPROVED:
                    raise ValueError("training eligibility requires source approval")
                if source.download_approval != ApprovalStatus.APPROVED:
                    raise ValueError("training eligibility requires download approval")
                if source.download_status not in {
                    DownloadStatus.NORMALIZED,
                    DownloadStatus.DUPLICATE,
                }:
                    raise ValueError("training eligibility requires a normalized download")
            return source.model_copy(update={"eligible_for_training": eligible})

        return self._review_source(
            source_id,
            action="training_eligibility_updated",
            actor=actor,
            reason=reason,
            evidence_reference=evidence_reference,
            expected_revision=expected_revision,
            state=lambda source: {"eligible_for_training": source.eligible_for_training},
            update=update,
        )

    def download_source(self, source_id: UUID | str) -> SourceRecord:
        source = self.registry.get_source(source_id)
        if source.source_approval != ApprovalStatus.APPROVED:
            raise ValueError("source is not approved")
        if source.download_approval != ApprovalStatus.APPROVED:
            raise ValueError("download is not approved")
        if source.download_method is None:
            raise ValueError("source has no supported download adapter")
        adapter = self.downloaders.by_name(source.download_method)
        reservation = source.content_length_bytes or round(
            self.config.dataset_intake.unknown_download_reservation_gb * GIB
        )
        self.storage.preflight(reservation)
        raw_path = self.storage.path_for("sources", "raw", f"{source.source_id}.mp4")
        normalized_path = self.storage.path_for(
            "sources", "normalized", f"{source.source_id}.mp4"
        )
        source = self._save_source(
            source.model_copy(
                update={
                    "revision": source.revision + 1,
                    "download_status": DownloadStatus.DOWNLOADING,
                    "error_state": None,
                }
            )
        )
        try:
            result = adapter.download(source, raw_path)
            result.path.chmod(0o600)
            self.storage.preflight(result.bytes_written)
            checksum = sha256_file(result.path)
            duplicate = next(
                (
                    item
                    for item in self.registry.load().sources.values()
                    if item.source_id != source.source_id and item.checksum_sha256 == checksum
                ),
                None,
            )
            if duplicate is not None:
                result.path.unlink(missing_ok=True)
                updated = source.model_copy(
                    update={
                        "revision": source.revision + 1,
                        "download_status": DownloadStatus.DUPLICATE,
                        "duplicate_of_source_id": duplicate.source_id,
                        "storage_path": duplicate.storage_path,
                        "normalized_path": duplicate.normalized_path,
                        "checksum_sha256": checksum,
                        "processing_status": duplicate.processing_status,
                        "duration_seconds": duplicate.duration_seconds,
                    }
                )
            else:
                self.storage.preflight(result.bytes_written)
                metadata = self.media.normalize(result.path, normalized_path)
                normalized_path.chmod(0o600)
                updated = source.model_copy(
                    update={
                        "revision": source.revision + 1,
                        "download_status": DownloadStatus.NORMALIZED,
                        "storage_path": result.path,
                        "normalized_path": normalized_path,
                        "checksum_sha256": checksum,
                        "processing_status": ProcessingStatus.NORMALIZED,
                        "duration_seconds": metadata.duration_seconds,
                        "estimated_useful_segment_count": max(
                            1, round(metadata.duration_seconds / 6)
                        ),
                        "error_state": None,
                    }
                )
            self._save_source(updated)
            return updated
        except Exception as exc:
            raw_path.with_suffix(raw_path.suffix + ".part").unlink(missing_ok=True)
            failed = source.model_copy(
                update={
                    "revision": source.revision + 1,
                    "download_status": DownloadStatus.FAILED,
                    "processing_status": ProcessingStatus.FAILED,
                    "error_state": str(exc),
                    "eligible_for_training": False,
                }
            )
            self._save_source(failed)
            return failed

    def suggest_source_segments(self, source_id: UUID | str) -> list[SegmentCandidate]:
        from cadence.dataset.signals import suggest_segments

        source = self.registry.get_source(source_id)
        if source.normalized_path is None or not source.normalized_path.is_file():
            raise ValueError("source must have a normalized local file")
        if source.duration_seconds is None:
            raise ValueError("source duration is unavailable")
        suggestions = suggest_segments(
            source.normalized_path,
            duration_seconds=source.duration_seconds,
            minimum_seconds=self.config.dataset_intake.segment_min_seconds,
            maximum_seconds=self.config.dataset_intake.segment_max_seconds,
            target_seconds=self.config.dataset_intake.segment_target_seconds,
            maximum_suggestions=self.config.dataset_intake.maximum_suggestions_per_source,
        )
        candidates: list[SegmentCandidate] = []
        existing_segments = self.registry.load().segments
        for suggestion in suggestions:
            segment_id = UUID(bytes=__import__("hashlib").sha256(
                f"{source.source_id}:{suggestion.start_seconds}:{suggestion.end_seconds}".encode()
            ).digest()[:16])
            existing = existing_segments.get(str(segment_id))
            if existing is not None:
                candidates.append(existing)
                continue
            output = self.storage.path_for("segments", "candidates", f"{segment_id}.mp4")
            estimated = max(1, round(source.normalized_path.stat().st_size * (
                suggestion.end_seconds - suggestion.start_seconds
            ) / source.duration_seconds))
            self.storage.preflight(estimated)
            self.media.extract_segment(
                source.normalized_path,
                output,
                suggestion.start_seconds,
                suggestion.end_seconds - suggestion.start_seconds,
            )
            output.chmod(0o600)
            candidate = SegmentCandidate(
                segment_id=segment_id,
                source_id=source.source_id,
                start_seconds=suggestion.start_seconds,
                end_seconds=suggestion.end_seconds,
                duration_seconds=suggestion.end_seconds - suggestion.start_seconds,
                motion_score=suggestion.motion_score,
                audio_activity_score=suggestion.audio_activity_score,
                scene_boundary=bool(suggestion.scene_boundary_seconds),
                scene_boundary_seconds=suggestion.scene_boundary_seconds,
                reason=suggestion.reason,
                categories=suggestion.categories,
                extracted_path=output,
                checksum_after_extraction=sha256_file(output),
            )
            candidates.append(candidate)

        def store(state: RegistryState) -> None:
            for candidate in candidates:
                state.segments[str(candidate.segment_id)] = candidate
            state.sources[str(source.source_id)] = source.model_copy(
                update={
                    "revision": state.sources[str(source.source_id)].revision + 1,
                    "processing_status": ProcessingStatus.SEGMENTS_SUGGESTED,
                    "estimated_useful_segment_count": len(candidates),
                }
            )

        self.registry.mutate(store)
        return candidates

    def list_segments(self, source_id: UUID | str) -> list[SegmentCandidate]:
        source_uuid = UUID(str(source_id))
        return sorted(
            [
                item
                for item in self.registry.load().segments.values()
                if item.source_id == source_uuid
            ],
            key=lambda item: item.start_seconds,
        )

    def set_segment_approval(
        self,
        segment_id: UUID | str,
        status: ApprovalStatus,
        *,
        actor: str = "cli",
        reason: str = "CLI segment approval update",
        evidence_reference: EvidenceReference | None = None,
        expected_revision: int | None = None,
    ) -> SegmentCandidate:
        key = str(segment_id)
        updated: SegmentCandidate | None = None

        def save(state: RegistryState) -> None:
            nonlocal updated
            segment = state.segments.get(key)
            if segment is None:
                raise KeyError(f"unknown segment ID: {segment_id}")
            self._check_revision(
                "segment", segment.segment_id, segment.revision, expected_revision
            )
            updated = segment.model_copy(
                update={
                    "revision": segment.revision + 1,
                    "approval_status": status,
                }
            )
            event = AuditEvent(
                entity_type="segment",
                entity_id=segment.segment_id,
                action="segment_approval_updated",
                actor=actor,
                reason=reason,
                evidence_reference=evidence_reference,
                revision=updated.revision,
                prior_state={"approval_status": segment.approval_status.value},
                new_state={"approval_status": updated.approval_status.value},
            )
            state.segments[key] = updated
            state.audit_events = (*state.audit_events, event)

        self.registry.mutate(save)
        if updated is None:
            raise RuntimeError("segment review did not produce an updated record")
        return updated

    def build_dataset(
        self,
        name: str,
        *,
        actor: str = "cli",
        reason: str = "CLI dataset build",
        evidence_reference: EvidenceReference | None = None,
    ) -> DatasetRecord:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", name) is None:
            raise ValueError("dataset name must be 2-64 lowercase letters, numbers, '_' or '-'")
        state = self.registry.load()
        included: list[tuple[SegmentCandidate, SourceRecord]] = []
        for segment in state.segments.values():
            source = state.sources[str(segment.source_id)]
            if segment.approval_status == ApprovalStatus.APPROVED and source.eligible_for_training:
                included.append((segment, source))
        if not included:
            raise ValueError("no approved segments from training-eligible sources")
        existing_versions = [item.version for item in state.datasets.values() if item.name == name]
        version = max(existing_versions, default=0) + 1
        dataset_dir = self.storage.path_for("datasets", name, f"v{version:04d}", ".keep").parent
        manifest_path = dataset_dir / "manifest.jsonl"
        report_path = dataset_dir / "report.json"
        entries: list[ManifestEntry] = []
        split_counts = {"train": 0, "validation": 0, "test": 0}
        total_bytes = 0
        for segment, source in included:
            metadata = self.media.probe(segment.extracted_path)
            split = deterministic_split(source.source_id, self.config.runtime.seed)
            split_counts[split] += 1
            total_bytes += segment.extracted_path.stat().st_size
            entries.append(
                ManifestEntry.model_validate(
                    {
                        "asset_id": segment.segment_id,
                        "source_asset_id": source.source_id,
                        "path": segment.extracted_path,
                        "duration_s": metadata.duration_seconds,
                        "fps": metadata.fps,
                        "width": metadata.width,
                        "height": metadata.height,
                        "audio_sample_rate": metadata.audio_sample_rate,
                        "has_video": metadata.has_video,
                        "has_audio": metadata.has_audio,
                        "checksum_sha256": segment.checksum_after_extraction,
                        "source_url": source.url,
                        "license_status": source.rights_status.value,
                        "collection_method": source.collection_method,
                        "split": split,
                        "eligible_for_contrastive": True,
                        "domain": "product-and-brand-launch-video-sound-design",
                    }
                )
            )
        write_manifest(entries, manifest_path)
        manifest_path.chmod(0o600)
        record = DatasetRecord(
            name=name,
            version=version,
            manifest_path=manifest_path,
            report_path=report_path,
            segment_ids=tuple(segment.segment_id for segment, _ in included),
            source_ids=tuple(dict.fromkeys(source.source_id for _, source in included)),
            total_bytes=total_bytes,
            split_counts=split_counts,
        )
        report = self._dataset_report_payload(record, state)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_path.chmod(0o600)

        def save(state_to_update: RegistryState) -> None:
            state_to_update.datasets[str(record.dataset_id)] = record
            event = AuditEvent(
                entity_type="dataset",
                entity_id=record.dataset_id,
                action="dataset_built",
                actor=actor,
                reason=reason,
                evidence_reference=evidence_reference,
                revision=record.version,
                prior_state={},
                new_state={
                    "name": record.name,
                    "version": record.version,
                    "segment_count": len(record.segment_ids),
                    "source_count": len(record.source_ids),
                },
            )
            state_to_update.audit_events = (*state_to_update.audit_events, event)

        self.registry.mutate(save)
        return record

    def dataset_report(self, name: str) -> dict[str, object]:
        latest = self.latest_dataset(name)
        if latest.report_path.is_file():
            payload = json.loads(latest.report_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        raise ValueError(f"dataset report is missing or invalid: {latest.report_path}")

    def _dataset_report_payload(
        self, record: DatasetRecord, state: RegistryState
    ) -> dict[str, object]:
        rights_counts: dict[str, int] = {}
        for source_id in record.source_ids:
            rights = state.sources[str(source_id)].rights_status.value
            rights_counts[rights] = rights_counts.get(rights, 0) + 1
        excluded_unverified = sum(
            1
            for source in state.sources.values()
            if source.rights_status == RightsStatus.UNVERIFIED and not source.eligible_for_training
        )
        segment_not_approved = sum(
            1
            for segment in state.segments.values()
            if segment.approval_status != ApprovalStatus.APPROVED
        )
        approved_but_source_ineligible = sum(
            1
            for segment in state.segments.values()
            if segment.approval_status == ApprovalStatus.APPROVED
            and not state.sources[str(segment.source_id)].eligible_for_training
        )
        return {
            **record.model_dump(mode="json"),
            "rights_counts": rights_counts,
            "excluded_unverified_sources": excluded_unverified,
            "registry_source_count": len(state.sources),
            "registry_segment_count": len(state.segments),
            "excluded_segments": {
                "not_approved": segment_not_approved,
                "source_ineligible": approved_but_source_ineligible,
            },
        }

    def _review_source(
        self,
        source_id: UUID | str,
        *,
        action: ReviewAction,
        actor: str,
        reason: str,
        evidence_reference: EvidenceReference | None,
        expected_revision: int | None,
        state: Callable[[SourceRecord], dict[str, ReviewStateValue]],
        update: Callable[[SourceRecord], SourceRecord],
    ) -> SourceRecord:
        key = str(source_id)
        updated: SourceRecord | None = None

        def save(registry_state: RegistryState) -> None:
            nonlocal updated
            source = registry_state.sources.get(key)
            if source is None:
                raise KeyError(f"unknown source ID: {source_id}")
            self._check_revision("source", source.source_id, source.revision, expected_revision)
            prior_state = state(source)
            updated = update(source).model_copy(update={"revision": source.revision + 1})
            event = AuditEvent(
                entity_type="source",
                entity_id=source.source_id,
                action=action,
                actor=actor,
                reason=reason,
                evidence_reference=evidence_reference,
                revision=updated.revision,
                prior_state=prior_state,
                new_state=state(updated),
            )
            registry_state.sources[key] = updated
            registry_state.audit_events = (*registry_state.audit_events, event)

        self.registry.mutate(save)
        if updated is None:
            raise RuntimeError("source review did not produce an updated record")
        return updated

    @staticmethod
    def _check_revision(
        entity_type: ReviewEntityType,
        entity_id: UUID,
        actual_revision: int,
        expected_revision: int | None,
    ) -> None:
        if expected_revision is not None and actual_revision != expected_revision:
            raise StaleRevisionError(
                entity_type,
                entity_id,
                expected_revision=expected_revision,
                actual_revision=actual_revision,
            )

    def _save_source(self, source: SourceRecord) -> SourceRecord:
        def save(state: RegistryState) -> None:
            state.sources[str(source.source_id)] = source

        self.registry.mutate(save)
        return source
