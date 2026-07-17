"""CPU-efficient media signals for candidate segment suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt

from cadence.dataset.records import SegmentCategory


@dataclass(frozen=True)
class SegmentSuggestion:
    start_seconds: float
    end_seconds: float
    motion_score: float
    audio_activity_score: float
    scene_boundary_seconds: tuple[float, ...]
    reason: str
    categories: tuple[SegmentCategory, ...]


FloatArray = npt.NDArray[np.float32]


def _normalize(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-8:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _video_signals(path: Path) -> tuple[FloatArray, FloatArray]:
    times: list[float] = []
    differences: list[float] = []
    previous: FloatArray | None = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 4)
        stride = max(1, round(fps / 4.0))
        for index, frame in enumerate(container.decode(stream)):
            if index % stride:
                continue
            time_base = frame.time_base
            if time_base is None:
                continue
            gray = frame.to_ndarray(format="gray")[::4, ::4].astype(np.float32) / 255.0
            difference = 0.0 if previous is None else float(np.mean(np.abs(gray - previous)))
            previous = gray
            times.append(float(frame.pts * time_base))
            differences.append(difference)
    return np.asarray(times, dtype=np.float32), np.asarray(differences, dtype=np.float32)


def _audio_signals(path: Path) -> tuple[FloatArray, FloatArray]:
    times: list[float] = []
    rms_values: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            native = frame.to_ndarray()
            array = native.astype(np.float32)
            if array.size == 0:
                continue
            if np.issubdtype(native.dtype, np.integer):
                array /= float(2 ** (8 * native.dtype.itemsize - 1))
            times.append(float(frame.pts * frame.time_base))
            rms_values.append(float(np.sqrt(np.mean(np.square(array)) + 1e-12)))
    return np.asarray(times, dtype=np.float32), np.asarray(rms_values, dtype=np.float32)


def _window_mean(times: FloatArray, values: FloatArray, start: float, end: float) -> float:
    selected = values[(times >= start) & (times < end)]
    return float(np.mean(selected)) if selected.size else 0.0


def suggest_segments(
    path: Path,
    *,
    duration_seconds: float,
    minimum_seconds: float,
    maximum_seconds: float,
    target_seconds: float,
    maximum_suggestions: int,
) -> list[SegmentSuggestion]:
    if duration_seconds < minimum_seconds:
        return []
    video_times, raw_motion = _video_signals(path)
    audio_times, raw_audio = _audio_signals(path)
    motion = _normalize(raw_motion)
    audio = _normalize(raw_audio)
    scene_threshold = max(0.35, float(np.mean(motion) + 2 * np.std(motion))) if motion.size else 1.0
    scene_times = video_times[motion >= scene_threshold]

    anchor_scores: list[tuple[float, float]] = []
    for time, score in zip(video_times, motion, strict=True):
        anchor_scores.append((float(score), float(time)))
    if audio.size > 1:
        changes = np.abs(np.diff(audio, prepend=audio[0]))
        for time, score in zip(audio_times, changes, strict=True):
            anchor_scores.append((float(score), float(time)))
    for time in scene_times:
        anchor_scores.append((2.0, float(time)))
    spacing = target_seconds * 0.75
    cursor = target_seconds / 2
    while cursor < duration_seconds:
        anchor_scores.append((0.2, cursor))
        cursor += spacing
    anchor_scores.sort(reverse=True)

    suggestions: list[SegmentSuggestion] = []
    for _, anchor in anchor_scores:
        length = min(maximum_seconds, max(minimum_seconds, target_seconds))
        start = min(max(0.0, anchor - length / 2), max(0.0, duration_seconds - length))
        end = min(duration_seconds, start + length)
        if end - start < minimum_seconds:
            continue
        overlaps_existing = any(
            min(end, item.end_seconds) - max(start, item.start_seconds) > 0.7 * length
            for item in suggestions
        )
        if overlaps_existing:
            continue
        motion_score = _window_mean(video_times, motion, start, end)
        audio_score = _window_mean(audio_times, audio, start, end)
        boundary_mask = (scene_times >= start) & (scene_times < end)
        boundaries = tuple(float(value) for value in scene_times[boundary_mask])
        categories: list[SegmentCategory] = []
        reasons: list[str] = []
        if start <= 0.1 * duration_seconds:
            categories.append(SegmentCategory.OPENING_BUILDUP)
            reasons.append("opening structure")
        if boundaries:
            categories.extend((SegmentCategory.TRANSITION, SegmentCategory.PRODUCT_REVEAL))
            reasons.append("scene boundary with visual change")
        if motion_score >= 0.5:
            categories.append(SegmentCategory.DEVICE_MOVEMENT)
            reasons.append("high frame-difference motion")
        if audio_score <= 0.15:
            categories.append(SegmentCategory.DELIBERATE_SILENCE)
            reasons.append("low audio activity or silence boundary")
        if end >= 0.9 * duration_seconds:
            categories.extend((SegmentCategory.LOGO_RESOLUTION, SegmentCategory.BRAND_LOCKUP))
            reasons.append("closing brand-resolution region")
        if not reasons:
            categories.append(SegmentCategory.CINEMATIC_PRODUCT_SHOT)
            reasons.append("balanced motion and native-audio activity")
        suggestions.append(
            SegmentSuggestion(
                start_seconds=round(start, 6),
                end_seconds=round(end, 6),
                motion_score=max(0.0, min(1.0, motion_score)),
                audio_activity_score=max(0.0, min(1.0, audio_score)),
                scene_boundary_seconds=boundaries,
                reason="; ".join(reasons),
                categories=tuple(dict.fromkeys(categories)),
            )
        )
        if len(suggestions) >= maximum_suggestions:
            break
    return sorted(suggestions, key=lambda item: item.start_seconds)
