import random

import pytest
import torch

from cadence.encoders.audio import AudioEncoder
from cadence.encoders.video import VideoEncoder
from cadence.training.checkpoint import ResumePosition, load_checkpoint, save_checkpoint


def models_and_optimizer() -> tuple[VideoEncoder, AudioEncoder, torch.optim.Optimizer]:
    video = VideoEncoder(embed_dim=8, projection_dim=4, sequence_length=2, base_channels=2)
    audio = AudioEncoder(embed_dim=8, projection_dim=4, sequence_length=2, base_channels=2)
    optimizer = torch.optim.AdamW(list(video.parameters()) + list(audio.parameters()))
    return video, audio, optimizer


def test_atomic_checkpoint_resume_and_compatibility(tmp_path: object) -> None:
    checkpoint = tmp_path / "checkpoint.pt"  # type: ignore[operator]
    video, audio, optimizer = models_and_optimizer()
    random.seed(5)
    save_checkpoint(
        checkpoint,
        video_encoder=video,
        audio_encoder=audio,
        optimizer=optimizer,
        position=ResumePosition(2, 17, 9),
        config_hash="c",
        manifest_hash="m",
        lock_hash="l",
        git_commit="a" * 40,
        metrics={"loss": 1.0},
    )
    expected_random = random.random()
    restored_video, restored_audio, restored_optimizer = models_and_optimizer()
    position, payload = load_checkpoint(
        checkpoint,
        video_encoder=restored_video,
        audio_encoder=restored_audio,
        optimizer=restored_optimizer,
        expected_config_hash="c",
        expected_manifest_hash="m",
        expected_lock_hash="l",
    )
    assert position == ResumePosition(2, 17, 9)
    assert payload["metrics"]["loss"] == 1.0
    assert random.random() == expected_random
    with pytest.raises(ValueError, match="config_hash"):
        load_checkpoint(
            checkpoint,
            video_encoder=restored_video,
            audio_encoder=restored_audio,
            optimizer=restored_optimizer,
            expected_config_hash="wrong",
            expected_manifest_hash="m",
            expected_lock_hash="l",
        )
    assert not checkpoint.with_suffix(".pt.tmp").exists()

