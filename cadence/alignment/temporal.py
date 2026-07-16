"""Resample masked modality tokens onto an explicit shared time grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cadence.encoders.types import TemporalEncoding


@dataclass(frozen=True)
class AlignedEncodings:
    video_tokens: Tensor
    audio_tokens: Tensor
    timestamps: Tensor
    video_mask: Tensor
    audio_mask: Tensor


class SharedTemporalAdapter:
    @staticmethod
    def _interpolate(encoding: TemporalEncoding, grid: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, grid_length = grid.shape
        output = encoding.tokens.new_zeros(
            (batch_size, grid_length, encoding.tokens.shape[-1])
        )
        output_mask = torch.zeros((batch_size, grid_length), dtype=torch.bool, device=grid.device)
        for batch_index in range(batch_size):
            valid = encoding.mask[batch_index]
            source_times = encoding.timestamps[batch_index, valid]
            source_tokens = encoding.tokens[batch_index, valid]
            if source_times.numel() == 0:
                continue
            targets = grid[batch_index]
            inside = (targets >= source_times[0]) & (targets <= source_times[-1])
            if source_times.numel() == 1:
                output[batch_index, inside] = source_tokens[0]
                output_mask[batch_index] = inside
                continue
            right = torch.searchsorted(source_times.contiguous(), targets.contiguous()).clamp(
                1, source_times.numel() - 1
            )
            left = right - 1
            left_time = source_times[left]
            right_time = source_times[right]
            weight = (
                (targets - left_time) / (right_time - left_time).clamp_min(1e-8)
            ).unsqueeze(-1)
            values = source_tokens[left] + weight * (source_tokens[right] - source_tokens[left])
            output[batch_index, inside] = values[inside]
            output_mask[batch_index] = inside
        return output, output_mask

    def __call__(
        self, video: TemporalEncoding, audio: TemporalEncoding, grid: Tensor
    ) -> AlignedEncodings:
        if grid.ndim != 2 or grid.shape[0] != video.tokens.shape[0]:
            raise ValueError("grid must have shape (B,G) matching the encoding batch")
        if audio.tokens.shape[0] != video.tokens.shape[0]:
            raise ValueError("audio and video batches must match")
        video_tokens, video_mask = self._interpolate(video, grid)
        audio_tokens, audio_mask = self._interpolate(audio, grid)
        return AlignedEncodings(video_tokens, audio_tokens, grid, video_mask, audio_mask)
