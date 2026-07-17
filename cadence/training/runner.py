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
from cadence.encoders.audio import AudioEncoder
from cadence.encoders.video import VideoEncoder
from cadence.ingestion.manifest import load_manifest, sha256_file
from cadence.training.checkpoint import ResumePosition, load_checkpoint, save_checkpoint
from cadence.training.contrastive import evaluate_retrieval, info_nce_loss
from cadence.training.synthetic import build_models, seed_everything


def _epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def _autocast_context(config: CadenceConfig, device: torch.device) -> Any:
    if device.type == "cuda" and config.training.precision == "amp-fp16":
        return torch.cuda.amp.autocast()
    return nullcontext()


@torch.no_grad()
def _evaluate_validation(
    config: CadenceConfig,
    *,
    manifest_path: Path,
    video_encoder: VideoEncoder,
    audio_encoder: AudioEncoder,
    device: torch.device,
) -> dict[str, object]:
    validation_data = config.data.model_copy(update={"split": "validation"})
    dataset = ContrastiveClipDataset(
        manifest_path,
        validation_data,
        seed=config.runtime.seed,
        max_samples=None,
    )
    dataset.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.microbatch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        collate_fn=collate_contrastive,
        pin_memory=device.type == "cuda",
    )
    video_embeddings: list[torch.Tensor] = []
    audio_embeddings: list[torch.Tensor] = []
    video_encoder.eval()
    audio_encoder.eval()
    try:
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
            with _autocast_context(config, device):
                video_embeddings.append(
                    video_encoder.encode(video_batch).global_embedding.detach().cpu()
                )
                audio_embeddings.append(
                    audio_encoder.encode(audio_batch).global_embedding.detach().cpu()
                )
    finally:
        video_encoder.train()
        audio_encoder.train()
    return evaluate_retrieval(
        torch.cat(video_embeddings),
        torch.cat(audio_embeddings),
        config.training.temperature,
    ).to_dict()


def _first_run_success_metrics(
    config: CadenceConfig,
    metrics: dict[str, Any],
) -> dict[str, object]:
    if not config.first_run.enabled:
        return {}
    success = config.first_run.success
    validation = metrics.get("validation")
    if success is None or not isinstance(validation, dict):
        raise RuntimeError("first-run completion requires full validation retrieval")
    chance = float(validation["chance"])
    minimum = chance * success.minimum_recall_at_1_over_chance
    for direction in ("video_to_audio", "audio_to_video"):
        direction_metrics = validation.get(direction)
        if not isinstance(direction_metrics, dict):
            raise RuntimeError(f"first-run validation is missing {direction}")
        if float(direction_metrics["recall_at_1"]) < minimum:
            raise RuntimeError(
                f"first-run {direction} Recall@1 did not meet the frozen chance threshold"
            )
    return {
        "success_criteria_met": True,
        "minimum_recall_at_1": minimum,
        "full_validation_sample_count": validation["sample_count"],
    }


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
    manifest_entries = load_manifest(config.paths.manifest_path)
    for entry in manifest_entries:
        if entry.path is None or sha256_file(entry.path) != entry.checksum_sha256:
            raise ValueError(f"manifest media checksum mismatch for asset {entry.asset_id}")
    if config.first_run.enabled:
        split_counts = {
            split: sum(entry.split == split for entry in manifest_entries)
            for split in ("train", "validation", "test")
        }
        expected_counts = {
            "train": config.first_run.expected_train_rows,
            "validation": config.first_run.expected_validation_rows,
            "test": config.first_run.expected_test_rows,
        }
        if split_counts != expected_counts:
            raise ValueError(
                f"first-run manifest split counts changed: {split_counts} != {expected_counts}"
            )
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
    amp_enabled = device.type == "cuda" and config.training.precision == "amp-fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    config_hash = stable_hash(config.model_dump(mode="json"))
    manifest_hash = file_hash(config.paths.manifest_path)
    lock_hash = file_hash(Path(repo_root) / "uv.lock")
    position = ResumePosition(0, 0, 0)
    last_metrics: dict[str, Any] = {}
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
        restored_metrics = payload.get("metrics")
        if isinstance(restored_metrics, dict):
            last_metrics = restored_metrics

    checkpoint_path = config.paths.checkpoint_dir / "latest.pt"
    if position.global_step >= config.training.max_steps:
        return {
            "position": position.__dict__,
            "checkpoint": str(resume_from),
            "stop_reason": "already-complete",
            "resume_verified": True,
            **last_metrics,
            **_first_run_success_metrics(config, last_metrics),
        }
    run_started = time.monotonic()
    hard_runtime_seconds = config.remote.maximum_runtime_minutes * 60
    soft_runtime_seconds = (
        config.first_run.abort.soft_stop_runtime_minutes * 60
        if config.first_run.enabled and config.first_run.abort is not None
        else None
    )
    last_checkpoint = time.monotonic()
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
        consumed_samples = offset
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
            with _autocast_context(config, device):
                buffered_video.append(video_encoder.encode(video_batch).global_embedding)
                buffered_audio.append(audio_encoder.encode(audio_batch).global_embedding)
            batch_size = video_batch.frames.shape[0]
            buffered_samples += batch_size
            consumed_samples += batch_size
            if buffered_samples < config.runtime.contrastive_group_size:
                continue
            video_global = torch.cat(buffered_video)
            audio_global = torch.cat(buffered_audio)
            loss, _ = info_nce_loss(
                video_global, audio_global, config.training.temperature
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite contrastive loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            gradients = [
                parameter.grad
                for parameter in list(video_encoder.parameters())
                + list(audio_encoder.parameters())
                if parameter.grad is not None
            ]
            if any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError("non-finite gradient")
            scaler.step(optimizer)
            scaler.update()
            position = ResumePosition(
                epoch=epoch,
                next_sample_offset=consumed_samples,
                global_step=position.global_step + 1,
            )
            last_metrics = {
                "training_batch": evaluate_retrieval(
                    video_global.detach(),
                    audio_global.detach(),
                    config.training.temperature,
                ).to_dict()
            }
            evaluation_interval = config.training.evaluation_interval_steps
            if (
                evaluation_interval is not None
                and position.global_step % evaluation_interval == 0
            ):
                last_metrics["validation"] = _evaluate_validation(
                    config,
                    manifest_path=config.paths.manifest_path,
                    video_encoder=video_encoder,
                    audio_encoder=audio_encoder,
                    device=device,
                )
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
                    "stop_reason": "maximum-steps",
                    **last_metrics,
                    **_first_run_success_metrics(config, last_metrics),
                }
            if (
                soft_runtime_seconds is not None
                and time.monotonic() - run_started >= soft_runtime_seconds
            ):
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
                return {
                    "position": position.__dict__,
                    "checkpoint": str(checkpoint_path),
                    "stop_reason": "soft-runtime-boundary",
                    **last_metrics,
                }
            if time.monotonic() - run_started >= hard_runtime_seconds:
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
                return {
                    "position": position.__dict__,
                    "checkpoint": str(checkpoint_path),
                    "stop_reason": "maximum-runtime",
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
