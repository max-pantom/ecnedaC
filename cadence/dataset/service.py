"""Dataset intake orchestration with explicit rights and storage gates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import UUID

from cadence.common.config import CadenceConfig
from cadence.dataset.downloaders import DownloaderChain
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

    def add_source(self, url: str, *, submitted_by: str) -> tuple[SourceRecord, bool]:
        candidate = SourceRecord.from_submission(url, submitted_by)
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

    def inspect_source(self, source_id: UUID | str) -> SourceRecord:
        source = self.registry.get_source(source_id)
        adapter, inspection = self.downloaders.inspect(str(source.url))
        if not inspection.supported or adapter is None:
            updated = source.model_copy(
                update={
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

    def set_source_approval(self, source_id: UUID | str, status: ApprovalStatus) -> SourceRecord:
        source = self.registry.get_source(source_id)
        updated = source.model_copy(update={"source_approval": status})
        if status == ApprovalStatus.REJECTED:
            updated = updated.model_copy(
                update={
                    "download_approval": ApprovalStatus.REJECTED,
                    "eligible_for_training": False,
                }
            )
        self._save_source(updated)
        return updated

    def set_download_approval(self, source_id: UUID | str, status: ApprovalStatus) -> SourceRecord:
        source = self.registry.get_source(source_id)
        if status == ApprovalStatus.APPROVED:
            if source.source_approval != ApprovalStatus.APPROVED:
                raise ValueError("source approval is required before download approval")
            if source.inspection_status != InspectionStatus.SUPPORTED:
                raise ValueError("a supported inspection is required before download approval")
            if source.rights_status in {RightsStatus.RESTRICTED, RightsStatus.REJECTED}:
                raise ValueError("restricted or rejected sources cannot be approved for download")
        updated = source.model_copy(update={"download_approval": status})
        self._save_source(updated)
        return updated

    def set_rights(
        self, source_id: UUID | str, status: RightsStatus, *, license_notes: str
    ) -> SourceRecord:
        source = self.registry.get_source(source_id)
        updated = source.model_copy(
            update={
                "rights_status": status,
                "license_notes": license_notes,
                "eligible_for_training": False,
            }
        )
        if status in {RightsStatus.RESTRICTED, RightsStatus.REJECTED}:
            updated = updated.model_copy(update={"download_approval": ApprovalStatus.REJECTED})
        self._save_source(updated)
        return updated

    def set_training_eligibility(self, source_id: UUID | str, eligible: bool) -> SourceRecord:
        source = self.registry.get_source(source_id)
        if eligible:
            if source.rights_status not in PERMITTED_RIGHTS:
                raise ValueError(
                    "training eligibility requires verified, owned, or licensed rights"
                )
            if source.source_approval != ApprovalStatus.APPROVED:
                raise ValueError("training eligibility requires source approval")
            if source.download_approval != ApprovalStatus.APPROVED:
                raise ValueError("training eligibility requires download approval")
            if source.download_status not in {DownloadStatus.NORMALIZED, DownloadStatus.DUPLICATE}:
                raise ValueError("training eligibility requires a normalized download")
        updated = source.model_copy(update={"eligible_for_training": eligible})
        self._save_source(updated)
        return updated

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
        self._save_source(
            source.model_copy(
                update={"download_status": DownloadStatus.DOWNLOADING, "error_state": None}
            )
        )
        try:
            result = adapter.download(source, raw_path)
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
                updated = source.model_copy(
                    update={
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
        self, segment_id: UUID | str, status: ApprovalStatus
    ) -> SegmentCandidate:
        segment = self.registry.get_segment(segment_id)
        updated = segment.model_copy(update={"approval_status": status})

        def save(state: RegistryState) -> None:
            state.segments[str(updated.segment_id)] = updated

        self.registry.mutate(save)
        return updated

    def build_dataset(self, name: str) -> DatasetRecord:
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
                        "collection_method": source.download_method or "unknown",
                        "split": split,
                        "eligible_for_contrastive": True,
                        "domain": "product-and-brand-launch-video-sound-design",
                    }
                )
            )
        write_manifest(entries, manifest_path)
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

        def save(state_to_update: RegistryState) -> None:
            state_to_update.datasets[str(record.dataset_id)] = record

        self.registry.mutate(save)
        return record

    def dataset_report(self, name: str) -> dict[str, object]:
        state = self.registry.load()
        matches = [item for item in state.datasets.values() if item.name == name]
        if not matches:
            raise KeyError(f"unknown dataset: {name}")
        latest = max(matches, key=lambda item: item.version)
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

    def _save_source(self, source: SourceRecord) -> None:
        def save(state: RegistryState) -> None:
            state.sources[str(source.source_id)] = source

        self.registry.mutate(save)
