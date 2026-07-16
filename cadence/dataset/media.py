"""FFmpeg/FFprobe media metadata, normalization, and aligned clip extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class MediaMetadata:
    duration_seconds: float
    fps: float
    width: int
    height: int
    audio_sample_rate: int
    has_video: bool
    has_audio: bool


class MediaProcessor(Protocol):
    def probe(self, path: Path) -> MediaMetadata: ...

    def normalize(self, source: Path, destination: Path) -> MediaMetadata: ...

    def extract_segment(
        self, source: Path, destination: Path, start_seconds: float, duration_seconds: float
    ) -> MediaMetadata: ...


class FFmpegMediaProcessor:
    def __init__(self, ffmpeg_binary: str = "ffmpeg", ffprobe_binary: str = "ffprobe") -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary

    def _require_binary(self, binary: str) -> None:
        if shutil.which(binary) is None:
            raise RuntimeError(f"required media binary is unavailable: {binary}")

    def probe(self, path: Path) -> MediaMetadata:
        self._require_binary(self.ffprobe_binary)
        result = subprocess.run(
            [
                self.ffprobe_binary,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
        if video is None or audio is None:
            raise ValueError("source must contain both video and native audio")
        rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
        numerator, denominator = (float(part) for part in rate_text.split("/"))
        duration = float(payload.get("format", {}).get("duration") or video.get("duration") or 0)
        if duration <= 0:
            raise ValueError("media duration is unavailable or invalid")
        return MediaMetadata(
            duration_seconds=duration,
            fps=numerator / denominator if denominator else 0,
            width=int(video.get("width") or 0),
            height=int(video.get("height") or 0),
            audio_sample_rate=int(audio.get("sample_rate") or 0),
            has_video=True,
            has_audio=True,
        )

    def normalize(self, source: Path, destination: Path) -> MediaMetadata:
        self._require_binary(self.ffmpeg_binary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                self.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-vf",
                "fps=30,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            check=True,
        )
        return self.probe(destination)

    def extract_segment(
        self, source: Path, destination: Path, start_seconds: float, duration_seconds: float
    ) -> MediaMetadata:
        self._require_binary(self.ffmpeg_binary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                self.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_seconds:.6f}",
                "-i",
                str(source),
                "-t",
                f"{duration_seconds:.6f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(destination),
            ],
            check=True,
        )
        return self.probe(destination)

