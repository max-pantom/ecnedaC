"""Deterministic, timestamp-aligned contrastive video/audio dataset."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from av.error import FFmpegError
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info

from cadence.common.config import DataConfig
from cadence.encoders.types import AudioBatch, VideoBatch
from cadence.ingestion.manifest import ManifestEntry, load_manifest

VIDEO_MEAN = (0.45, 0.45, 0.45)
VIDEO_STD = (0.225, 0.225, 0.225)


@dataclass(frozen=True)
class ContrastiveSample:
    video: Tensor
    video_timestamps: Tensor
    video_mask: Tensor
    audio: Tensor
    audio_timestamps: Tensor
    audio_mask: Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ContrastiveBatch:
    video: VideoBatch
    audio: AudioBatch
    metadata: tuple[dict[str, Any], ...]


class ContrastiveClipDataset(Dataset[ContrastiveSample]):
    def __init__(
        self,
        manifest_path: str | Path,
        config: DataConfig,
        *,
        seed: int = 1337,
        max_samples: int | None = None,
    ) -> None:
        entries = [entry for entry in load_manifest(manifest_path) if entry.split == config.split]
        if max_samples is not None:
            entries = entries[:max_samples]
        if not entries:
            raise ValueError(f"manifest has no entries for split {config.split}")
        for entry in entries:
            self._validate_entry(entry)
        self.entries = entries
        self.config = config
        self.seed = seed
        self.epoch = 0
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    @staticmethod
    def _validate_entry(entry: ManifestEntry) -> None:
        if not entry.eligible_for_contrastive:
            raise ValueError(f"asset {entry.asset_id} is not eligible for contrastive training")
        if not entry.has_video or not entry.has_audio:
            raise ValueError(f"asset {entry.asset_id} must contain video and native audio")
        if entry.path is None:
            raise ValueError(f"asset {entry.asset_id} is non-local; remote media is disabled")
        if not entry.path.is_file():
            raise ValueError(f"asset {entry.asset_id} path does not exist: {entry.path}")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.entries)

    def _sample_start(self, entry: ManifestEntry) -> float:
        maximum = max(entry.duration_s - self.config.clip_seconds, 0.0)
        if maximum == 0:
            return 0.0
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        material = f"{self.seed}:{self.epoch}:{worker_id}:{entry.asset_id}".encode()
        random_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(random_seed).uniform(0.0, maximum)

    @staticmethod
    def _decode_video(path: Path, start: float, end: float) -> tuple[Tensor, Tensor]:
        frames: list[Tensor] = []
        timestamps: list[float] = []
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    raise ValueError("no video stream")
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    time_base = frame.time_base
                    if time_base is None:
                        continue
                    timestamp = float(frame.pts * time_base)
                    if start <= timestamp < end:
                        array = frame.to_ndarray(format="rgb24")
                        frames.append(torch.from_numpy(array.copy()))
                        timestamps.append(timestamp)
        except (FFmpegError, OSError, ValueError) as exc:
            raise ValueError(f"failed to decode video {path}: {exc}") from exc
        if not frames:
            raise ValueError(f"decoded no video frames from {path} in [{start}, {end})")
        return torch.stack(frames), torch.tensor(timestamps, dtype=torch.float32)

    @staticmethod
    def _decode_audio(path: Path, start: float, end: float) -> tuple[Tensor, int]:
        selected: list[Tensor] = []
        native_rate = 0
        try:
            with av.open(str(path)) as container:
                if not container.streams.audio:
                    raise ValueError("no audio stream")
                stream = container.streams.audio[0]
                native_rate = int(stream.codec_context.sample_rate or stream.rate or 0)
                if native_rate <= 0:
                    raise ValueError("invalid audio sample rate")
                for frame in container.decode(stream):
                    pts = frame.pts
                    time_base = frame.time_base
                    if pts is None or time_base is None:
                        continue
                    frame_start = float(pts * time_base)
                    array = frame.to_ndarray()
                    if array.ndim == 1:
                        array = array[None, :]
                    tensor = torch.from_numpy(np.asarray(array).copy()).float()
                    if np.issubdtype(array.dtype, np.integer):
                        scale = float(2 ** (8 * array.dtype.itemsize - 1))
                        tensor = tensor / scale
                    tensor = tensor.mean(dim=0, keepdim=True)
                    frame_end = frame_start + tensor.shape[1] / native_rate
                    overlap_start = max(start, frame_start)
                    overlap_end = min(end, frame_end)
                    if overlap_end <= overlap_start:
                        continue
                    left = round((overlap_start - frame_start) * native_rate)
                    right = round((overlap_end - frame_start) * native_rate)
                    selected.append(tensor[:, left:right])
        except (FFmpegError, OSError, ValueError) as exc:
            raise ValueError(f"failed to decode audio {path}: {exc}") from exc
        if not selected:
            raise ValueError(f"decoded no audio samples from {path} in [{start}, {end})")
        return torch.cat(selected, dim=1), native_rate

    def _process_video(
        self, frames: Tensor, timestamps: Tensor, start: float, valid_duration: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        video = frames.float().permute(0, 3, 1, 2) / 255.0
        video = F.interpolate(
            video,
            size=(self.config.frame_size, self.config.frame_size),
            mode="bilinear",
            align_corners=False,
        )
        source_timestamps = timestamps - start
        target_timestamps = (
            torch.arange(self.config.num_frames, dtype=torch.float32)
            * self.config.clip_seconds
            / self.config.num_frames
        )
        right = torch.searchsorted(source_timestamps.contiguous(), target_timestamps).clamp(
            max=source_timestamps.numel() - 1
        )
        left = (right - 1).clamp_min(0)
        use_left = (
            (target_timestamps - source_timestamps[left]).abs()
            <= (source_timestamps[right] - target_timestamps).abs()
        )
        indices = torch.where(use_left, left, right)
        video = video[indices]
        mask = target_timestamps < valid_duration
        mean = video.new_tensor(VIDEO_MEAN).view(1, 3, 1, 1)
        std = video.new_tensor(VIDEO_STD).view(1, 3, 1, 1)
        video = ((video - mean) / std).permute(1, 0, 2, 3)
        return video, target_timestamps, mask

    def _process_audio(
        self, waveform: Tensor, native_rate: int, start: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        if native_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, native_rate, self.config.sample_rate
            )
        target = round(self.config.clip_seconds * self.config.sample_rate)
        valid_samples = min(waveform.shape[1], target)
        waveform = waveform[:, :target]
        if waveform.shape[1] < target:
            waveform = F.pad(waveform, (0, target - waveform.shape[1]))
        mel = self.amplitude_to_db(self.mel_spectrogram(waveform))
        mean = mel.mean()
        std = mel.std(unbiased=False)
        mel = (mel - mean) / std.clamp_min(1e-6)
        timestamps = (
            torch.arange(mel.shape[-1], dtype=torch.float32)
            * self.config.hop_length
            / self.config.sample_rate
        )
        mask = timestamps < (valid_samples / self.config.sample_rate)
        return mel, timestamps + (start - start), mask

    def __getitem__(self, index: int) -> ContrastiveSample:
        entry = self.entries[index]
        assert entry.path is not None
        start = self._sample_start(entry)
        end = start + self.config.clip_seconds
        frames, frame_timestamps = self._decode_video(entry.path, start, end)
        waveform, native_rate = self._decode_audio(entry.path, start, end)
        valid_duration = min(entry.duration_s - start, self.config.clip_seconds)
        video, video_timestamps, video_mask = self._process_video(
            frames, frame_timestamps, start, valid_duration
        )
        audio, audio_timestamps, audio_mask = self._process_audio(waveform, native_rate, start)
        return ContrastiveSample(
            video,
            video_timestamps,
            video_mask,
            audio,
            audio_timestamps,
            audio_mask,
            {
                "asset_id": str(entry.asset_id),
                "source_asset_id": str(entry.source_asset_id),
                "start_s": start,
                "end_s": end,
                "source_url": str(entry.source_url),
                "license_status": entry.license_status,
                "collection_method": entry.collection_method,
            },
        )


def collate_contrastive(samples: list[ContrastiveSample]) -> ContrastiveBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    return ContrastiveBatch(
        video=VideoBatch(
            torch.stack([sample.video for sample in samples]),
            torch.stack([sample.video_timestamps for sample in samples]),
            torch.stack([sample.video_mask for sample in samples]),
        ),
        audio=AudioBatch(
            torch.stack([sample.audio for sample in samples]),
            torch.stack([sample.audio_timestamps for sample in samples]),
            torch.stack([sample.audio_mask for sample in samples]),
        ),
        metadata=tuple(sample.metadata for sample in samples),
    )
