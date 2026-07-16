"""Shared encoder building blocks and reporting."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(self.net(value), dim=-1)


def resample_temporal_metadata(
    timestamps: Tensor, mask: Tensor, sequence_length: int
) -> tuple[Tensor, Tensor]:
    output_timestamps = F.interpolate(
        timestamps.unsqueeze(1), size=sequence_length, mode="linear", align_corners=True
    ).squeeze(1)
    output_mask = F.interpolate(
        mask.float().unsqueeze(1), size=sequence_length, mode="nearest"
    ).squeeze(1).bool()
    return output_timestamps, output_mask


def masked_temporal_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True)
class MemoryEstimate:
    parameters_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    input_bytes: int
    estimated_total_bytes: int


def estimate_training_memory(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    bytes_per_value: int = 4,
    optimizer_multiplier: int = 2,
    activation_multiplier: int = 6,
) -> MemoryEstimate:
    parameters = count_parameters(model) * bytes_per_value
    gradients = parameters
    optimizer = parameters * optimizer_multiplier
    input_values = 1
    for dimension in input_shape:
        input_values *= dimension
    input_bytes = input_values * bytes_per_value
    total = parameters + gradients + optimizer + input_bytes * activation_multiplier
    return MemoryEstimate(parameters, gradients, optimizer, input_bytes, total)


def zero_masked_tokens(tokens: Tensor, mask: Tensor) -> Tensor:
    return tokens * mask.unsqueeze(-1).to(tokens.dtype)
