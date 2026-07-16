"""Typed input and output contracts for Cadence encoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class VideoBatch:
    frames: Tensor
    timestamps: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        _require(self.frames.ndim == 5, "video frames must have shape (B,C,T,H,W)")
        _require(self.frames.shape[1] == 3, "video frames must have three RGB channels")
        _require(self.timestamps.shape == self.frames.shape[:1] + self.frames.shape[2:3],
                 "video timestamps must have shape (B,T)")
        _require(self.mask.shape == self.timestamps.shape, "video mask must have shape (B,T)")
        _require(self.mask.dtype == torch.bool, "video mask must be boolean")


@dataclass(frozen=True)
class AudioBatch:
    spectrograms: Tensor
    timestamps: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        _require(self.spectrograms.ndim == 4, "spectrograms must have shape (B,1,F,T)")
        _require(self.spectrograms.shape[1] == 1, "spectrograms must have one channel")
        expected = self.spectrograms.shape[:1] + self.spectrograms.shape[3:4]
        _require(self.timestamps.shape == expected, "audio timestamps must have shape (B,T)")
        _require(self.mask.shape == expected, "audio mask must have shape (B,T)")
        _require(self.mask.dtype == torch.bool, "audio mask must be boolean")


@dataclass(frozen=True)
class TemporalEncoding:
    tokens: Tensor
    timestamps: Tensor
    mask: Tensor
    global_embedding: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.tokens.ndim == 3, "tokens must have shape (B,S,D)")
        _require(self.timestamps.shape == self.tokens.shape[:2], "timestamps must have shape (B,S)")
        _require(self.mask.shape == self.tokens.shape[:2], "mask must have shape (B,S)")
        _require(self.global_embedding.shape[0] == self.tokens.shape[0],
                 "global embedding batch must match tokens")
        _require(self.mask.dtype == torch.bool, "encoding mask must be boolean")

