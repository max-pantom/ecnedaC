"""Configuration-driven manifest contrastive training entry point."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from cadence.common.config import CadenceConfig
from cadence.common.repro import file_hash, git_commit, stable_hash
from cadence.data.contrastive import (
    ContrastiveBatch,
    ContrastiveClipDataset,
    collate_contrastive,
)
from cadence.training.checkpoint import ResumePosition, load_checkpoint, save_checkpoint
from cadence.training.contrastive import evaluate_retrieval, info_nce_loss
from cadence.training.synthetic import build_models, seed_everything


def _epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def run_contrastive_training(
    config: CadenceConfig,
    *,
    resume_from: str | Path | None = None,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    if config.paths.manifest_path is None:
        raise ValueError("paths.manifest_path is required for contrastive training")
    if config.runtime.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("configuration requests CUDA but no CUDA device is available")
    seed_everything(config.runtime.seed)
    device = torch.device(config.runtime.device)
    dataset = ContrastiveClipDataset(
        config.paths.manifest_path,
        config.data,
        seed=config.runtime.seed,
        max_samples=config.runtime.max_samples,
    )
    video_encoder, audio_encoder = build_models(config)
    video_encoder.to(device)
    audio_encoder.to(device)
    optimizer = torch.optim.AdamW(
        list(video_encoder.parameters()) + list(audio_encoder.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    config_hash = stable_hash(config.model_dump(mode="json"))
    manifest_hash = file_hash(config.paths.manifest_path)
    lock_hash = file_hash(Path(repo_root) / "uv.lock")
    position = ResumePosition(0, 0, 0)
    if resume_from:
        position, payload = load_checkpoint(
            resume_from,
            video_encoder=video_encoder,
            audio_encoder=audio_encoder,
            optimizer=optimizer,
            expected_config_hash=config_hash,
            expected_manifest_hash=manifest_hash,
            expected_lock_hash=lock_hash,
        )
        if payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])

    checkpoint_path = config.paths.checkpoint_dir / "latest.pt"
    last_checkpoint = time.monotonic()
    last_metrics: dict[str, Any] = {}
    video_encoder.train()
    audio_encoder.train()

    for epoch in range(position.epoch, config.runtime.epochs):
        dataset.set_epoch(epoch)
        order = _epoch_order(len(dataset), config.runtime.seed, epoch)
        offset = position.next_sample_offset if epoch == position.epoch else 0
        subset = Subset(dataset, order[offset:])
        loader = DataLoader(
            subset,
            batch_size=config.runtime.microbatch_size,
            shuffle=False,
            num_workers=config.runtime.num_workers,
            collate_fn=collate_contrastive,
            pin_memory=device.type == "cuda",
        )
        buffered_video: list[torch.Tensor] = []
        buffered_audio: list[torch.Tensor] = []
        buffered_samples = 0
        for batch in loader:
            assert isinstance(batch, ContrastiveBatch)
            video_batch = type(batch.video)(
                batch.video.frames.to(device),
                batch.video.timestamps.to(device),
                batch.video.mask.to(device),
            )
            audio_batch = type(batch.audio)(
                batch.audio.spectrograms.to(device),
                batch.audio.timestamps.to(device),
                batch.audio.mask.to(device),
            )
            autocast_context = (
                torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
            )
            with autocast_context:
                buffered_video.append(video_encoder.encode(video_batch).global_embedding)
                buffered_audio.append(audio_encoder.encode(audio_batch).global_embedding)
            batch_size = video_batch.frames.shape[0]
            buffered_samples += batch_size
            if buffered_samples < config.runtime.contrastive_group_size:
                continue
            video_global = torch.cat(buffered_video)
            audio_global = torch.cat(buffered_audio)
            loss, _ = info_nce_loss(
                video_global, audio_global, config.training.temperature
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.step(optimizer)
            scaler.update()
            position = ResumePosition(
                epoch=epoch,
                next_sample_offset=offset + buffered_samples,
                global_step=position.global_step + 1,
            )
            last_metrics = evaluate_retrieval(
                video_global.detach(), audio_global.detach(), config.training.temperature
            ).to_dict()
            buffered_video.clear()
            buffered_audio.clear()
            buffered_samples = 0
            should_checkpoint = (
                time.monotonic() - last_checkpoint >= config.training.checkpoint_interval_seconds
            )
            if should_checkpoint or position.global_step >= config.training.max_steps:
                save_checkpoint(
                    checkpoint_path,
                    video_encoder=video_encoder,
                    audio_encoder=audio_encoder,
                    optimizer=optimizer,
                    position=position,
                    config_hash=config_hash,
                    manifest_hash=manifest_hash,
                    lock_hash=lock_hash,
                    git_commit=git_commit(repo_root),
                    metrics=last_metrics,
                    scaler_state=scaler.state_dict(),
                )
                last_checkpoint = time.monotonic()
            if position.global_step >= config.training.max_steps:
                return {
                    "position": position.__dict__,
                    "checkpoint": str(checkpoint_path),
                    **last_metrics,
                }
        position = ResumePosition(epoch + 1, 0, position.global_step)

    save_checkpoint(
        checkpoint_path,
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
        optimizer=optimizer,
        position=position,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        lock_hash=lock_hash,
        git_commit=git_commit(repo_root),
        metrics=last_metrics,
        scaler_state=scaler.state_dict(),
    )
    return {"position": position.__dict__, "checkpoint": str(checkpoint_path), **last_metrics}
