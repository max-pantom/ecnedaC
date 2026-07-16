"""Atomic, compatibility-checked Cadence checkpoints."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

CHECKPOINT_VERSION = "0.1.0"


@dataclass(frozen=True)
class ResumePosition:
    epoch: int
    next_sample_offset: int
    global_step: int


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    video_encoder: nn.Module,
    audio_encoder: nn.Module,
    optimizer: Optimizer,
    position: ResumePosition,
    config_hash: str,
    manifest_hash: str,
    lock_hash: str,
    git_commit: str,
    metrics: dict[str, Any],
    scaler_state: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "position": {
            "epoch": position.epoch,
            "next_sample_offset": position.next_sample_offset,
            "global_step": position.global_step,
        },
        "video_encoder": video_encoder.state_dict(),
        "audio_encoder": audio_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler_state or {},
        "rng": capture_rng_state(),
        "compatibility": {
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "lock_hash": lock_hash,
        },
        "git_commit": git_commit,
        "metrics": metrics,
        "model_metadata": {
            "video": type(video_encoder).__name__,
            "audio": type(audio_encoder).__name__,
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    video_encoder: nn.Module,
    audio_encoder: nn.Module,
    optimizer: Optimizer,
    expected_config_hash: str,
    expected_manifest_hash: str,
    expected_lock_hash: str,
    allow_incompatible: bool = False,
    restore_rng: bool = True,
) -> tuple[ResumePosition, dict[str, Any]]:
    payload: dict[str, Any] = torch.load(Path(path), map_location="cpu")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    expected = {
        "config_hash": expected_config_hash,
        "manifest_hash": expected_manifest_hash,
        "lock_hash": expected_lock_hash,
    }
    actual = payload.get("compatibility", {})
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches and not allow_incompatible:
        raise ValueError("incompatible checkpoint fields: " + ", ".join(mismatches))
    video_encoder.load_state_dict(payload["video_encoder"])
    audio_encoder.load_state_dict(payload["audio_encoder"])
    optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    position = ResumePosition(**payload["position"])
    return position, payload


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(Path(path), map_location="cpu")
    return {
        "checkpoint_version": payload.get("checkpoint_version"),
        "position": payload.get("position"),
        "compatibility": payload.get("compatibility"),
        "git_commit": payload.get("git_commit"),
        "metrics": payload.get("metrics"),
        "model_metadata": payload.get("model_metadata"),
    }

