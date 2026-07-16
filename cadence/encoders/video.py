"""Compact R(2+1)D video encoder."""

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
from cadence.encoders.types import TemporalEncoding, VideoBatch


def conv2plus1d(
    in_channels: int,
    out_channels: int,
    spatial_stride: int = 1,
    temporal_stride: int = 1,
) -> nn.Sequential:
    temporal_kernel = spatial_kernel = 3
    mid_channels = int(
        (temporal_kernel * spatial_kernel**2 * in_channels * out_channels)
        / (spatial_kernel**2 * in_channels + temporal_kernel * out_channels)
    )
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            max(mid_channels, 1),
            kernel_size=(1, 3, 3),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, 1, 1),
            bias=False,
        ),
        nn.BatchNorm3d(max(mid_channels, 1)),
        nn.ReLU(inplace=True),
        nn.Conv3d(
            max(mid_channels, 1),
            out_channels,
            kernel_size=(3, 1, 1),
            stride=(temporal_stride, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        ),
    )


class R2Plus1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv2plus1d(in_channels, out_channels, spatial_stride, temporal_stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2plus1d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample: nn.Module | None = None
        if spatial_stride != 1 or temporal_stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(temporal_stride, spatial_stride, spatial_stride),
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, value: Tensor) -> Tensor:
        identity = value
        output = self.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return cast(Tensor, self.relu(output + identity))


class VideoEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        projection_dim: int = 128,
        sequence_length: int = 8,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.base_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(3, base_channels, (1, 7, 7), (1, 2, 2), (0, 3, 3), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        channels = [base_channels * multiplier for multiplier in (1, 2, 4, 8)]
        self.stage1 = R2Plus1DBlock(channels[0], channels[0])
        self.stage2 = R2Plus1DBlock(channels[0], channels[1], spatial_stride=2)
        self.stage3 = R2Plus1DBlock(channels[1], channels[2], spatial_stride=2)
        self.stage4 = R2Plus1DBlock(
            channels[2], channels[3], spatial_stride=2, temporal_stride=2
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool1d(sequence_length)
        self.to_embed = nn.Linear(channels[3], embed_dim)
        self.projection = ProjectionHead(embed_dim, embed_dim, projection_dim)

    def forward(self, frames: Tensor) -> Tensor:
        value = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(frames)))))
        value = self.spatial_pool(value).squeeze(-1).squeeze(-1)
        return cast(Tensor, self.to_embed(self.temporal_pool(value).transpose(1, 2)))

    def encode(self, batch: VideoBatch) -> TemporalEncoding:
        tokens = self.forward(batch.frames)
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
                "architecture": "r2plus1d",
                "base_channels": self.base_channels,
                "sequence_length": self.sequence_length,
            },
        )
