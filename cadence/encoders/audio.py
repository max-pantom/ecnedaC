"""Compact residual audio encoder over log-mel spectrograms."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor

from cadence.encoders.common import (
    ProjectionHead,
    masked_temporal_mean,
    resample_temporal_metadata,
    zero_masked_tokens,
)
from cadence.encoders.types import AudioBatch, TemporalEncoding


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, freq_stride: int = 1, time_stride: int = 1
    ) -> None:
        super().__init__()
        stride = (freq_stride, time_stride)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, value: Tensor) -> Tensor:
        identity = value
        output = self.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return cast(Tensor, self.relu(output + identity))


class AudioEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        projection_dim: int = 128,
        sequence_length: int = 8,
        base_channels: int = 96,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.base_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        channels = [base_channels * multiplier for multiplier in (1, 2, 4, 8)]
        self.stage1 = ConvBlock(channels[0], channels[0], freq_stride=2)
        self.stage2 = ConvBlock(channels[0], channels[1], freq_stride=2, time_stride=2)
        self.stage3 = ConvBlock(channels[1], channels[2], freq_stride=2)
        self.stage4 = ConvBlock(channels[2], channels[3], freq_stride=2, time_stride=2)
        self.frequency_pool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_pool = nn.AdaptiveAvgPool1d(sequence_length)
        self.to_embed = nn.Linear(channels[3], embed_dim)
        self.projection = ProjectionHead(embed_dim, embed_dim, projection_dim)

    def forward(self, spectrograms: Tensor) -> Tensor:
        value = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(spectrograms)))))
        value = self.frequency_pool(value).squeeze(2)
        return cast(Tensor, self.to_embed(self.temporal_pool(value).transpose(1, 2)))

    def encode(self, batch: AudioBatch) -> TemporalEncoding:
        tokens = self.forward(batch.spectrograms)
        timestamps, mask = resample_temporal_metadata(
            batch.timestamps, batch.mask, self.sequence_length
        )
        tokens = zero_masked_tokens(tokens, mask)
        global_embedding = self.projection(masked_temporal_mean(tokens, mask))
        return TemporalEncoding(
            tokens=tokens,
            timestamps=timestamps,
            mask=mask,
            global_embedding=global_embedding,
            metadata={
                "architecture": "spectrogram-resnet",
                "base_channels": self.base_channels,
                "sequence_length": self.sequence_length,
            },
        )
