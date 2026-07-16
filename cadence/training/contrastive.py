"""Symmetric InfoNCE and full-set retrieval metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce_loss(
    video_embedding: Tensor, audio_embedding: Tensor, temperature: float = 0.07
) -> tuple[Tensor, Tensor]:
    if video_embedding.ndim != 2 or video_embedding.shape != audio_embedding.shape:
        raise ValueError("video and audio embeddings must have the same (B,D) shape")
    if video_embedding.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two paired embeddings")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = video_embedding @ audio_embedding.t() / temperature
    target = torch.arange(logits.shape[0], device=logits.device)
    loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2
    return loss, logits


@dataclass(frozen=True)
class DirectionMetrics:
    recall_at_1: float
    recall_at_5: float
    mean_rank: float
    median_rank: float


@dataclass(frozen=True)
class RetrievalMetrics:
    loss: float
    chance: float
    sample_count: int
    video_to_audio: DirectionMetrics
    audio_to_video: DirectionMetrics

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _direction_metrics(logits: Tensor) -> DirectionMetrics:
    count = logits.shape[0]
    target = torch.arange(count, device=logits.device)
    order = logits.argsort(dim=1, descending=True)
    ranks = (order == target.unsqueeze(1)).nonzero(as_tuple=False)[:, 1].float() + 1
    return DirectionMetrics(
        recall_at_1=float((ranks <= 1).float().mean().item()),
        recall_at_5=float((ranks <= min(5, count)).float().mean().item()),
        mean_rank=float(ranks.mean().item()),
        median_rank=float(ranks.median().item()),
    )


@torch.no_grad()
def evaluate_retrieval(
    video_embedding: Tensor, audio_embedding: Tensor, temperature: float = 0.07
) -> RetrievalMetrics:
    loss, logits = info_nce_loss(video_embedding, audio_embedding, temperature)
    return RetrievalMetrics(
        loss=float(loss.item()),
        chance=1.0 / logits.shape[0],
        sample_count=logits.shape[0],
        video_to_audio=_direction_metrics(logits),
        audio_to_video=_direction_metrics(logits.t()),
    )

