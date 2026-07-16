from pathlib import Path

import pytest
import torch

from cadence.common.config import load_config
from cadence.data.contrastive import ContrastiveClipDataset, collate_contrastive
from cadence.ingestion.fixtures import generate_fixtures
from cadence.ingestion.manifest import load_manifest, write_manifest


@pytest.mark.media
def test_generated_fixture_decodes_to_aligned_masked_batch(tmp_path: Path) -> None:
    manifest = generate_fixtures(tmp_path)
    config = load_config("configs/test.yaml")
    dataset = ContrastiveClipDataset(manifest, config.data, seed=7, max_samples=3)
    first = dataset[0]
    repeat = dataset[0]
    assert torch.equal(first.video, repeat.video)
    assert torch.equal(first.audio, repeat.audio)
    assert first.video.shape == (3, 4, 32, 32)
    assert first.audio.shape[0:2] == (1, 16)
    assert first.video_timestamps[0] == pytest.approx(0.0, abs=0.15)
    assert torch.isfinite(first.audio).all()
    batch = collate_contrastive([first])
    assert batch.video.frames.shape == (1, 3, 4, 32, 32)
    assert batch.audio.spectrograms.shape[0] == 1


@pytest.mark.media
def test_short_clip_pads_and_silence_is_finite(tmp_path: Path) -> None:
    manifest = generate_fixtures(tmp_path)
    config = load_config("configs/test.yaml")
    dataset = ContrastiveClipDataset(manifest, config.data, max_samples=3)
    silence = dataset[1]
    short = dataset[2]
    assert torch.isfinite(silence.audio).all()
    assert short.video_mask.sum() < config.data.num_frames
    assert short.audio_mask.sum() < short.audio_mask.numel()


@pytest.mark.media
def test_ineligible_missing_audio_is_rejected(tmp_path: Path) -> None:
    manifest = generate_fixtures(tmp_path)
    entries = load_manifest(manifest)
    missing_audio_manifest = tmp_path / "missing-audio.jsonl"
    write_manifest([entries[3]], missing_audio_manifest)
    with pytest.raises(ValueError, match="not eligible"):
        ContrastiveClipDataset(missing_audio_manifest, load_config("configs/test.yaml").data)

