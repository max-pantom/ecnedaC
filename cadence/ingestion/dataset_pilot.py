"""Launch-video dataset pilot coordination tools.

The pilot is deliberately CPU-cheap: metadata inspection, simple frame-difference
motion scoring, RMS/onset-style audio activity scoring, and reviewable segment
candidates. It does not run encoders, optical flow, or neural analysis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from uuid import NAMESPACE_URL, UUID, uuid5

import av
import numpy as np
from av.error import FFmpegError

from cadence.ingestion.manifest import (
    ManifestEntry,
    deterministic_split,
    sha256_file,
    write_manifest,
)

DOMAIN = "launch-video-sound-design"
MAX_CADENCE_STORAGE_BYTES = 20 * 1024**3
MIN_FREE_BYTES = 15 * 1024**3

SOURCE_FILE = "sources.jsonl"
SEGMENT_FILE = "segments.jsonl"

LAUNCH_PRIORITIES = (
    "product_reveal",
    "logo_resolution",
    "feature_montage",
    "kinetic_typography",
    "device_rotation",
    "interface_reveal",
    "camera_transition",
    "visual_buildup",
    "large_visual_arrival",
    "deliberate_silence",
    "final_brand_lockup",
)

REJECT_OR_FLAG = (
    "talking_head",
    "static_frames",
    "podcast_footage",
    "dialogue_only",
    "music_visualizer",
    "unrelated_cinematic",
    "corrupt_audio",
    "missing_audio",
    "duplicate_clip",
    "poor_synchronization",
)


@dataclass(frozen=True)
class SourceRecord:
    source_asset_id: UUID
    source_url: str
    media_path: str | None
    storage_uri: str | None
    creator: str | None
    publisher: str | None
    submitted_by: str
    collection_method: str
    license_status: str
    rights_status: str
    source_state: str
    download_status: str
    domain: str
    checksum_sha256: str
    duration_s: float
    fps: float
    width: int
    height: int
    audio_sample_rate: int
    has_video: bool
    has_audio: bool
    eligible_for_contrastive: bool
    eligible_for_training: bool
    added_at: str


@dataclass(frozen=True)
class SegmentCandidate:
    clip_asset_id: UUID
    source_asset_id: UUID
    source_url: str
    source_path: str | None
    start_s: float
    end_s: float
    duration_s: float
    motion_intensity: float
    audio_activity: float
    silence_ratio: float
    review_status: str
    rejection_reason: str | None
    license_status: str
    split: str
    eligible_for_contrastive: bool
    tags: tuple[str, ...]
    notes: str


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _json_default(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"unsupported JSON value: {value!r}")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(row), default=_json_default, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _coerce_source_row(row: dict[str, object]) -> SourceRecord:
    rights_status = str(row.get("rights_status") or (
        "unverified" if row.get("license_status") in {"unknown", "unverified", "unverified-research-quarantine"}
        else row.get("license_status") or "unverified"
    ))
    eligible = bool(row.get("eligible_for_training", row.get("eligible_for_contrastive", False)))
    if rights_status == "unverified":
        eligible = False
    return SourceRecord(
        source_asset_id=UUID(str(row["source_asset_id"])),
        source_url=str(row["source_url"]),
        media_path=str(row["media_path"]) if row.get("media_path") else None,
        storage_uri=str(row["storage_uri"]) if row.get("storage_uri") else None,
        creator=str(row["creator"]) if row.get("creator") else None,
        publisher=str(row["publisher"]) if row.get("publisher") else None,
        submitted_by=str(row.get("submitted_by") or "unknown"),
        collection_method=str(row["collection_method"]),
        license_status=str(row.get("license_status") or "unverified-research-quarantine"),
        rights_status=rights_status,
        source_state=str(row.get("source_state") or "candidate"),
        download_status=str(row.get("download_status") or ("downloaded" if row.get("media_path") else "not_downloaded")),
        domain=str(row.get("domain") or DOMAIN),
        checksum_sha256=str(row.get("checksum_sha256") or ""),
        duration_s=float(row.get("duration_s") or 0.0),
        fps=float(row.get("fps") or 0.0),
        width=int(row.get("width") or 0),
        height=int(row.get("height") or 0),
        audio_sample_rate=int(row.get("audio_sample_rate") or 0),
        has_video=bool(row.get("has_video") or False),
        has_audio=bool(row.get("has_audio") or False),
        eligible_for_contrastive=eligible,
        eligible_for_training=eligible,
        added_at=str(row.get("added_at") or _now()),
    )


def _read_sources(pilot_dir: str | Path) -> list[SourceRecord]:
    path = Path(pilot_dir) / SOURCE_FILE
    if not path.exists():
        return []
    return [
        _coerce_source_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_source_queue(pilot_dir: str | Path) -> list[SourceRecord]:
    return _read_sources(pilot_dir)


def _read_segments(pilot_dir: str | Path) -> list[SegmentCandidate]:
    path = Path(pilot_dir) / SEGMENT_FILE
    if not path.exists():
        return []
    rows = []
    for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line):
        rows.append(
            SegmentCandidate(
                **{
                    **row,
                    "clip_asset_id": UUID(row["clip_asset_id"]),
                    "source_asset_id": UUID(row["source_asset_id"]),
                    "tags": tuple(row.get("tags", ())),
                }
            )
        )
    return rows


def _pilot_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def enforce_storage_budget(pilot_dir: str | Path) -> None:
    root = Path(pilot_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    if usage.free < MIN_FREE_BYTES:
        raise RuntimeError("free VPS storage is below the 15 GB Cadence safety floor")
    if _pilot_size(root) > MAX_CADENCE_STORAGE_BYTES:
        raise RuntimeError("Cadence pilot storage exceeds the 20 GB VPS budget")


def _media_metadata(path: Path) -> dict[str, object]:
    try:
        with av.open(str(path)) as container:
            video = container.streams.video[0] if container.streams.video else None
            audio = container.streams.audio[0] if container.streams.audio else None
            duration_s = float(container.duration / av.time_base) if container.duration else 0.0
            fps = float(video.average_rate) if video and video.average_rate else 0.0
            return {
                "duration_s": duration_s,
                "fps": fps,
                "width": int(video.codec_context.width if video else 0),
                "height": int(video.codec_context.height if video else 0),
                "audio_sample_rate": int(audio.codec_context.sample_rate if audio else 0),
                "has_video": video is not None,
                "has_audio": audio is not None,
            }
    except (FFmpegError, OSError, ValueError) as exc:
        raise ValueError(f"failed to inspect media {path}: {exc}") from exc


def _is_training_eligible(license_status: str, has_video: bool, has_audio: bool) -> bool:
    quarantined = license_status in {"unverified-research-quarantine", "unknown", "unverified"}
    return has_video and has_audio and not quarantined



def write_candidate_sources(
    pilot_dir: str | Path,
    urls: list[str],
    *,
    submitted_by: str,
    collection_method: str = "user-submitted-url",
) -> list[SourceRecord]:
    enforce_storage_budget(pilot_dir)
    existing = _read_sources(pilot_dir)
    by_url = {source.source_url: source for source in existing}
    for url in urls:
        clean = url.strip()
        if not clean or clean in by_url:
            continue
        source_id = uuid5(NAMESPACE_URL, clean)
        by_url[clean] = SourceRecord(
            source_asset_id=source_id,
            source_url=clean,
            media_path=None,
            storage_uri=None,
            creator=None,
            publisher=None,
            submitted_by=submitted_by,
            collection_method=collection_method,
            license_status="unverified-research-quarantine",
            rights_status="unverified",
            source_state="candidate",
            download_status="not_downloaded",
            domain=DOMAIN,
            checksum_sha256="",
            duration_s=0.0,
            fps=0.0,
            width=0,
            height=0,
            audio_sample_rate=0,
            has_video=False,
            has_audio=False,
            eligible_for_contrastive=False,
            eligible_for_training=False,
            added_at=_now(),
        )
    rows = list(by_url.values())
    _write_jsonl(Path(pilot_dir) / SOURCE_FILE, rows)
    return rows


def approve_sources(pilot_dir: str | Path, source_asset_ids: list[UUID]) -> list[SourceRecord]:
    wanted = set(source_asset_ids)
    updated: list[SourceRecord] = []
    for source in _read_sources(pilot_dir):
        if source.source_asset_id in wanted:
            updated.append(SourceRecord(**{
                **asdict(source),
                "source_asset_id": source.source_asset_id,
                "source_state": "approved_source",
                "eligible_for_training": False if source.rights_status == "unverified" else source.eligible_for_training,
                "eligible_for_contrastive": False if source.rights_status == "unverified" else source.eligible_for_contrastive,
            }))
        else:
            updated.append(source)
    _write_jsonl(Path(pilot_dir) / SOURCE_FILE, updated)
    return updated

def write_source_record(
    pilot_dir: str | Path,
    *,
    media_path: str | Path,
    source_url: str,
    creator: str | None,
    collection_method: str,
    license_status: str = "unverified-research-quarantine",
    storage_uri: str | None = None,
) -> SourceRecord:
    enforce_storage_budget(pilot_dir)
    media = Path(media_path).resolve()
    if not media.is_file():
        raise FileNotFoundError(media)
    metadata = _media_metadata(media)
    checksum = sha256_file(media)
    source_id = uuid5(NAMESPACE_URL, f"{source_url}:{checksum}")
    record = SourceRecord(
        source_asset_id=source_id,
        source_url=source_url,
        media_path=str(media),
        storage_uri=storage_uri,
        creator=creator,
        publisher=creator,
        submitted_by="user",
        collection_method=collection_method,
        license_status=license_status,
        rights_status="unverified" if license_status in {"unknown", "unverified", "unverified-research-quarantine"} else license_status,
        source_state="approved_source",
        download_status="downloaded",
        domain=DOMAIN,
        checksum_sha256=checksum,
        duration_s=float(metadata["duration_s"]),
        fps=float(metadata["fps"]),
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        audio_sample_rate=int(metadata["audio_sample_rate"]),
        has_video=bool(metadata["has_video"]),
        has_audio=bool(metadata["has_audio"]),
        eligible_for_contrastive=_is_training_eligible(
            license_status, bool(metadata["has_video"]), bool(metadata["has_audio"])
        ),
        eligible_for_training=_is_training_eligible(
            license_status, bool(metadata["has_video"]), bool(metadata["has_audio"])
        ),
        added_at=_now(),
    )
    rows = [source for source in _read_sources(pilot_dir) if source.source_asset_id != source_id]
    rows.append(record)
    _write_jsonl(Path(pilot_dir) / SOURCE_FILE, rows)
    return record


def inspect_source(source: SourceRecord) -> SourceRecord:
    if source.media_path is None:
        return source
    metadata = _media_metadata(Path(source.media_path))
    return SourceRecord(
        **{
            **asdict(source),
            "source_asset_id": source.source_asset_id,
            "duration_s": float(metadata["duration_s"]),
            "fps": float(metadata["fps"]),
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "audio_sample_rate": int(metadata["audio_sample_rate"]),
            "has_video": bool(metadata["has_video"]),
            "has_audio": bool(metadata["has_audio"]),
            "eligible_for_contrastive": _is_training_eligible(
                source.license_status, bool(metadata["has_video"]), bool(metadata["has_audio"])
            ),
            "eligible_for_training": _is_training_eligible(
                source.license_status, bool(metadata["has_video"]), bool(metadata["has_audio"])
            ),
        }
    )


def _source_by_id(pilot_dir: str | Path, source_asset_id: UUID) -> SourceRecord:
    for source in _read_sources(pilot_dir):
        if source.source_asset_id == source_asset_id:
            return source
    raise ValueError(f"unknown source asset: {source_asset_id}")


def _window_scores(path: Path, start_s: float, end_s: float) -> tuple[float, float, float]:
    frame_values: list[np.ndarray] = []
    audio_chunks: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            if container.streams.video:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    if frame.pts is None or frame.time_base is None:
                        continue
                    timestamp = float(frame.pts * frame.time_base)
                    if timestamp < start_s:
                        continue
                    if timestamp >= end_s:
                        break
                    array = frame.to_ndarray(format="gray").astype(np.float32)
                    frame_values.append(array[:: max(1, array.shape[0] // 16), :: max(1, array.shape[1] // 16)])
            if container.streams.audio:
                stream = container.streams.audio[0]
                for frame in container.decode(stream):
                    if frame.pts is None or frame.time_base is None:
                        continue
                    timestamp = float(frame.pts * frame.time_base)
                    if timestamp < start_s:
                        continue
                    if timestamp >= end_s:
                        break
                    audio = frame.to_ndarray().astype(np.float32)
                    audio_chunks.append(audio.reshape(-1))
    except (FFmpegError, OSError, ValueError):
        return 0.0, 0.0, 1.0

    diffs = [float(np.mean(np.abs(b - a))) / 255.0 for a, b in zip(frame_values, frame_values[1:])]
    motion = float(mean(diffs)) if diffs else 0.0
    if audio_chunks:
        audio = np.concatenate(audio_chunks)
        if np.max(np.abs(audio)) > 1.5:
            audio = audio / np.iinfo(np.int16).max
        frame = max(1, len(audio) // 32)
        rms = np.array([np.sqrt(np.mean(audio[i : i + frame] ** 2)) for i in range(0, len(audio), frame)])
        audio_activity = float(np.mean(np.abs(np.diff(rms)))) if len(rms) > 1 else float(np.mean(rms))
        silence_ratio = float(np.mean(rms < 0.005)) if len(rms) else 1.0
    else:
        audio_activity = 0.0
        silence_ratio = 1.0
    return motion, audio_activity, silence_ratio


def suggest_segments(
    pilot_dir: str | Path,
    source_asset_id: UUID,
    *,
    min_duration_s: float = 4.0,
    max_duration_s: float = 10.0,
) -> list[SegmentCandidate]:
    enforce_storage_budget(pilot_dir)
    source = _source_by_id(pilot_dir, source_asset_id)
    if source.media_path is None:
        raise ValueError("local media_path is required for VPS-side segment suggestions")
    if not source.has_video or not source.has_audio:
        candidate = SegmentCandidate(
            clip_asset_id=uuid5(NAMESPACE_URL, f"{source.source_asset_id}:missing-modality"),
            source_asset_id=source.source_asset_id,
            source_url=source.source_url,
            source_path=source.media_path,
            start_s=0.0,
            end_s=min(source.duration_s, max_duration_s),
            duration_s=min(source.duration_s, max_duration_s),
            motion_intensity=0.0,
            audio_activity=0.0,
            silence_ratio=1.0,
            review_status="rejected",
            rejection_reason="missing_audio" if not source.has_audio else "missing_video",
            license_status=source.license_status,
            split=deterministic_split(source.source_asset_id),
            eligible_for_contrastive=False,
            tags=("missing_modality",),
            notes="Rejected automatically because source is missing required modality.",
        )
        rows = [segment for segment in _read_segments(pilot_dir) if segment.source_asset_id != source_asset_id]
        rows.append(candidate)
        _write_jsonl(Path(pilot_dir) / SEGMENT_FILE, rows)
        return [candidate]

    duration = max(0.0, source.duration_s)
    if duration <= 0:
        raise ValueError("source duration must be positive")
    window = min(max_duration_s, max(min_duration_s, duration))
    starts = [0.0]
    if duration > window:
        midpoint = max(0.0, (duration - window) / 2)
        starts.append(midpoint)
        starts.append(max(0.0, duration - window))
    unique_starts = sorted({round(start, 3) for start in starts})

    candidates: list[SegmentCandidate] = []
    existing = [segment for segment in _read_segments(pilot_dir) if segment.source_asset_id != source_asset_id]
    for start in unique_starts:
        end = min(duration, start + window)
        if end - start < min_duration_s:
            continue
        motion, audio, silence = _window_scores(Path(source.media_path), start, end)
        clip_id = uuid5(NAMESPACE_URL, f"{source.source_asset_id}:{start:.3f}:{end:.3f}")
        candidates.append(
            SegmentCandidate(
                clip_asset_id=clip_id,
                source_asset_id=source.source_asset_id,
                source_url=source.source_url,
                source_path=source.media_path,
                start_s=start,
                end_s=end,
                duration_s=round(end - start, 3),
                motion_intensity=motion,
                audio_activity=audio,
                silence_ratio=silence,
                review_status="candidate",
                rejection_reason=None,
                license_status=source.license_status,
                split=deterministic_split(source.source_asset_id),
                eligible_for_contrastive=False,
                tags=("launch_video_candidate",),
                notes="Review for launch-video sound-design content before approval.",
            )
        )
    _write_jsonl(Path(pilot_dir) / SEGMENT_FILE, existing + candidates)
    return candidates


def approve_segments(pilot_dir: str | Path, clip_asset_ids: list[UUID]) -> list[SegmentCandidate]:
    wanted = set(clip_asset_ids)
    updated: list[SegmentCandidate] = []
    for segment in _read_segments(pilot_dir):
        if segment.clip_asset_id in wanted and segment.rejection_reason is None:
            eligible = segment.license_status not in {"unverified-research-quarantine", "unknown", "unverified"}
            updated.append(
                SegmentCandidate(
                    **{
                        **asdict(segment),
                        "clip_asset_id": segment.clip_asset_id,
                        "source_asset_id": segment.source_asset_id,
                        "review_status": "approved",
                        "eligible_for_contrastive": eligible,
                        "tags": segment.tags,
                    }
                )
            )
        else:
            updated.append(segment)
    _write_jsonl(Path(pilot_dir) / SEGMENT_FILE, updated)
    return updated


def reject_segments(
    pilot_dir: str | Path, clip_asset_ids: list[UUID], *, reason: str
) -> list[SegmentCandidate]:
    wanted = set(clip_asset_ids)
    updated: list[SegmentCandidate] = []
    for segment in _read_segments(pilot_dir):
        if segment.clip_asset_id in wanted:
            updated.append(
                SegmentCandidate(
                    **{
                        **asdict(segment),
                        "clip_asset_id": segment.clip_asset_id,
                        "source_asset_id": segment.source_asset_id,
                        "review_status": "rejected",
                        "rejection_reason": reason,
                        "eligible_for_contrastive": False,
                        "tags": segment.tags,
                    }
                )
            )
        else:
            updated.append(segment)
    _write_jsonl(Path(pilot_dir) / SEGMENT_FILE, updated)
    return updated


def _extract_clip(segment: SegmentCandidate, output: Path) -> None:
    if segment.source_path is None:
        raise ValueError("source_path is required for local clip extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{segment.start_s:.3f}",
        "-t",
        f"{segment.duration_s:.3f}",
        "-i",
        segment.source_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c",
        "copy",
        str(output),
    ]
    subprocess.run(command, check=True)


def build_pilot_manifest(pilot_dir: str | Path, *, dataset_id: str) -> Path:
    enforce_storage_budget(pilot_dir)
    root = Path(pilot_dir)
    dataset_dir = root / "datasets" / dataset_id
    clips_dir = dataset_dir / "clips"
    entries: list[ManifestEntry] = []
    for segment in _read_segments(root):
        if segment.review_status != "approved":
            continue
        clip_path = clips_dir / f"{segment.clip_asset_id}.mp4"
        _extract_clip(segment, clip_path)
        metadata = _media_metadata(clip_path)
        entry = ManifestEntry.model_validate(
            {
                "asset_id": segment.clip_asset_id,
                "source_asset_id": segment.source_asset_id,
                "path": clip_path.resolve(),
                "duration_s": float(metadata["duration_s"]),
                "fps": float(metadata["fps"]),
                "width": int(metadata["width"]),
                "height": int(metadata["height"]),
                "audio_sample_rate": int(metadata["audio_sample_rate"]),
                "has_video": bool(metadata["has_video"]),
                "has_audio": bool(metadata["has_audio"]),
                "checksum_sha256": sha256_file(clip_path),
                "source_url": segment.source_url,
                "license_status": segment.license_status,
                "collection_method": "cadence-dataset-pilot-local-filesystem",
                "split": segment.split,
                "eligible_for_contrastive": segment.eligible_for_contrastive
                and bool(metadata["has_video"])
                and bool(metadata["has_audio"]),
                "domain": DOMAIN,
                "clip_start_s": segment.start_s,
                "clip_end_s": segment.end_s,
                "review_status": segment.review_status,
                "rejection_reason": segment.rejection_reason,
            }
        )
        entries.append(entry)
    manifest = dataset_dir / "manifest.jsonl"
    write_manifest(entries, manifest)
    write_report(root, dataset_id=dataset_id)
    return manifest


def _bucket(values: list[float], step: float) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        lower = int(value // step) * step
        upper = lower + step
        counts[f"{lower:.1f}-{upper:.1f}"] += 1
    return dict(sorted(counts.items()))


def build_report(pilot_dir: str | Path, *, dataset_id: str) -> dict[str, object]:
    root = Path(pilot_dir)
    sources = _read_sources(root)
    segments = _read_segments(root)
    approved = [segment for segment in segments if segment.review_status == "approved"]
    rejected = [segment for segment in segments if segment.review_status == "rejected"]
    checksums = Counter(source.checksum_sha256 for source in sources if source.checksum_sha256)
    license_counts = Counter(source.license_status for source in sources)
    missing_modality = sum(1 for source in sources if not source.has_video or not source.has_audio)
    dataset_dir = root / "datasets" / dataset_id
    disk_usage = _pilot_size(dataset_dir)
    manifest = dataset_dir / "manifest.jsonl"
    manifest_durations = []
    if manifest.exists():
        manifest_durations = [
            ManifestEntry.model_validate_json(line).duration_s
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    total_duration = sum(manifest_durations) if manifest_durations else sum(
        segment.duration_s for segment in approved
    )
    return {
        "dataset_id": dataset_id,
        "source_videos": len(sources),
        "candidate_segments": len(segments),
        "approved_segments": len(approved),
        "rejected_segments": len(rejected),
        "total_duration_s": round(total_duration, 3),
        "estimated_disk_usage_bytes": disk_usage,
        "license_status_breakdown": dict(sorted(license_counts.items())),
        "duplicate_count": sum(count - 1 for count in checksums.values() if count > 1),
        "missing_modality_count": missing_modality,
        "duration_distribution": _bucket([segment.duration_s for segment in segments], 2.0),
        "motion_intensity_distribution": _bucket([segment.motion_intensity for segment in segments], 0.05),
        "audio_activity_distribution": _bucket([segment.audio_activity for segment in segments], 0.01),
        "created_at": _now(),
    }


def write_report(pilot_dir: str | Path, *, dataset_id: str) -> Path:
    report = build_report(pilot_dir, dataset_id=dataset_id)
    output = Path(pilot_dir) / "datasets" / dataset_id / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_report(pilot_dir: str | Path, *, dataset_id: str) -> dict[str, object]:
    path = Path(pilot_dir) / "datasets" / dataset_id / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))
