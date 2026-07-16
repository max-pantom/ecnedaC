"""Generate tiny deterministic media fixtures; generated files are never committed."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import av
import numpy as np

from cadence.ingestion.manifest import ManifestEntry, sha256_file, write_manifest


def _write_mp4(
    path: Path,
    *,
    duration_s: float,
    fps: int = 8,
    sample_rate: int = 8000,
    event_s: float | None = 0.5,
    include_audio: bool = True,
    silent: bool = False,
) -> None:
    width = height = 32
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        audio_stream = None
        if include_audio:
            audio_stream = container.add_stream("aac", rate=sample_rate)
            audio_stream.layout = "mono"
        frame_count = max(1, round(duration_s * fps))
        for index in range(frame_count):
            timestamp = index / fps
            value = 255 if event_s is not None and abs(timestamp - event_s) < 0.5 / fps else 16
            pixels = np.full((height, width, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            sample_count = max(1, round(duration_s * sample_rate))
            waveform = np.zeros((1, sample_count), dtype=np.float32)
            if not silent and event_s is not None:
                start = min(sample_count, round(event_s * sample_rate))
                stop = min(sample_count, start + max(1, sample_rate // 20))
                timeline = np.arange(stop - start, dtype=np.float32) / sample_rate
                waveform[0, start:stop] = 0.8 * np.sin(2 * np.pi * 880 * timeline)
            audio_frame = av.AudioFrame.from_ndarray(waveform, format="flt", layout="mono")
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, sample_rate)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)


def _entry(path: Path, duration_s: float, *, has_audio: bool) -> ManifestEntry:
    asset_id = uuid5(NAMESPACE_URL, path.name)
    return ManifestEntry.model_validate(
        {
            "asset_id": asset_id,
            "source_asset_id": asset_id,
            "path": path.resolve(),
            "duration_s": duration_s,
            "fps": 8.0,
            "width": 32,
            "height": 32,
            "audio_sample_rate": 8000,
            "has_video": True,
            "has_audio": has_audio,
            "checksum_sha256": sha256_file(path),
            "source_url": f"https://fixtures.cadence.invalid/{path.name}",
            "license_status": "synthetic-generated",
            "collection_method": "cadence-fixture-generator",
            "split": "train",
            "eligible_for_contrastive": has_audio,
        }
    )


def generate_fixtures(output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("aligned.mp4", 1.0, True, False, 0.5),
        ("silence.mp4", 1.0, True, True, None),
        ("short.mp4", 0.5, True, False, 0.25),
        ("video-only.mp4", 1.0, False, False, 0.5),
    ]
    entries: list[ManifestEntry] = []
    for name, duration, has_audio, silent, event in specifications:
        path = output / name
        _write_mp4(
            path,
            duration_s=duration,
            event_s=event,
            include_audio=has_audio,
            silent=silent,
        )
        entries.append(_entry(path, duration, has_audio=has_audio))
    (output / "corrupt.mp4").write_bytes(b"not a media container")
    manifest = output / "manifest.jsonl"
    write_manifest(entries, manifest)
    return manifest
