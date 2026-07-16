"""CPU-safe one-step synthetic readiness training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cadence.common.config import CadenceConfig
from cadence.common.repro import file_hash, git_commit, stable_hash
from cadence.encoders.audio import AudioEncoder
from cadence.encoders.types import AudioBatch, VideoBatch
from cadence.encoders.video import VideoEncoder
from cadence.training.checkpoint import ResumePosition, load_checkpoint, save_checkpoint
from cadence.training.contrastive import evaluate_retrieval, info_nce_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_models(config: CadenceConfig) -> tuple[VideoEncoder, AudioEncoder]:
    model = config.encoders
    return (
        VideoEncoder(
            model.embed_dim,
            model.projection_dim,
            model.sequence_length,
            model.video_base_channels,
        ),
        AudioEncoder(
            model.embed_dim,
            model.projection_dim,
            model.sequence_length,
            model.audio_base_channels,
        ),
    )


def _synthetic_microbatch(config: CadenceConfig, index: int) -> tuple[VideoBatch, AudioBatch]:
    data = config.data
    generator = torch.Generator().manual_seed(config.runtime.seed + index)
    frames = torch.randn(
        1, 3, data.num_frames, data.frame_size, data.frame_size, generator=generator
    )
    audio_steps = max(8, round(data.clip_seconds * data.sample_rate / data.hop_length) + 1)
    spectrogram = torch.randn(1, 1, data.n_mels, audio_steps, generator=generator)
    video_time = torch.linspace(0, data.clip_seconds, data.num_frames).unsqueeze(0)
    audio_time = torch.linspace(0, data.clip_seconds, audio_steps).unsqueeze(0)
    return (
        VideoBatch(frames, video_time, torch.ones_like(video_time, dtype=torch.bool)),
        AudioBatch(spectrogram, audio_time, torch.ones_like(audio_time, dtype=torch.bool)),
    )


def run_synthetic_training(
    config: CadenceConfig,
    *,
    checkpoint_path: str | Path,
    resume_from: str | Path | None = None,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    if config.runtime.device != "cpu":
        raise ValueError("synthetic local readiness training must use CPU")
    seed_everything(config.runtime.seed)
    video_encoder, audio_encoder = build_models(config)
    optimizer = torch.optim.AdamW(
        list(video_encoder.parameters()) + list(audio_encoder.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    config_digest = stable_hash(config.model_dump(mode="json"))
    manifest_digest = (
        file_hash(config.paths.manifest_path)
        if config.paths.manifest_path
        else stable_hash("synthetic")
    )
    lock_digest = file_hash(Path(repo_root) / "uv.lock")
    position = ResumePosition(epoch=0, next_sample_offset=0, global_step=0)
    if resume_from:
        position, _ = load_checkpoint(
            resume_from,
            video_encoder=video_encoder,
            audio_encoder=audio_encoder,
            optimizer=optimizer,
            expected_config_hash=config_digest,
            expected_manifest_hash=manifest_digest,
            expected_lock_hash=lock_digest,
        )
    before = next(video_encoder.parameters()).detach().clone()
    video_embeddings = []
    audio_embeddings = []
    group_size = max(2, config.runtime.contrastive_group_size)
    for index in range(group_size):
        synthetic_index = position.global_step * group_size + index
        video_batch, audio_batch = _synthetic_microbatch(config, synthetic_index)
        video_embeddings.append(video_encoder.encode(video_batch).global_embedding)
        audio_embeddings.append(audio_encoder.encode(audio_batch).global_embedding)
    video_global = torch.cat(video_embeddings)
    audio_global = torch.cat(audio_embeddings)
    loss, _ = info_nce_loss(video_global, audio_global, config.training.temperature)
    if not torch.isfinite(loss):
        raise RuntimeError("synthetic contrastive loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    updated = not torch.equal(before, next(video_encoder.parameters()).detach())
    if not updated:
        raise RuntimeError("optimizer did not update video encoder parameters")
    metrics = evaluate_retrieval(
        video_global.detach(), audio_global.detach(), config.training.temperature
    ).to_dict()
    metrics.update({"optimizer_updated": updated, "microbatch_size": 1, "group_size": group_size})
    next_position = ResumePosition(
        epoch=position.epoch,
        next_sample_offset=position.next_sample_offset + group_size,
        global_step=position.global_step + 1,
    )
    save_checkpoint(
        checkpoint_path,
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
        optimizer=optimizer,
        position=next_position,
        config_hash=config_digest,
        manifest_hash=manifest_digest,
        lock_hash=lock_digest,
        git_commit=git_commit(repo_root),
        metrics=metrics,
    )
    return {"loss": float(loss.item()), "position": next_position.__dict__, **metrics}
