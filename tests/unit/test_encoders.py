import torch

from cadence.encoders.audio import AudioEncoder
from cadence.encoders.common import count_parameters, estimate_training_memory
from cadence.encoders.types import AudioBatch, VideoBatch
from cadence.encoders.video import VideoEncoder


def test_small_encoders_shape_mask_normalization_and_backward() -> None:
    video = VideoEncoder(embed_dim=16, projection_dim=8, sequence_length=2, base_channels=4)
    audio = AudioEncoder(embed_dim=16, projection_dim=8, sequence_length=2, base_channels=4)
    frame_mask = torch.tensor([[True, True, True, False]])
    audio_mask = torch.tensor([[True] * 7 + [False]])
    video_output = video.encode(VideoBatch(
        torch.randn(1, 3, 4, 32, 32), torch.linspace(0, 1, 4).unsqueeze(0), frame_mask
    ))
    audio_output = audio.encode(AudioBatch(
        torch.randn(1, 1, 16, 8), torch.linspace(0, 1, 8).unsqueeze(0), audio_mask
    ))
    assert video_output.tokens.shape == (1, 2, 16)
    assert audio_output.tokens.shape == (1, 2, 16)
    assert torch.allclose(video_output.global_embedding.norm(dim=-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(audio_output.global_embedding.norm(dim=-1), torch.ones(1), atol=1e-5)
    loss = (
        video_output.global_embedding.square().mean()
        + audio_output.global_embedding.square().mean()
    )
    loss.backward()
    assert next(video.parameters()).grad is not None
    assert next(audio.parameters()).grad is not None


def test_full_research_parameter_counts() -> None:
    assert count_parameters(VideoEncoder(base_channels=64)) == 14_589_014
    assert count_parameters(AudioEncoder(base_channels=96)) == 11_318_368


def test_memory_estimate_is_explicit() -> None:
    model = VideoEncoder(embed_dim=16, projection_dim=8, sequence_length=2, base_channels=4)
    estimate = estimate_training_memory(model, (1, 3, 4, 32, 32))
    assert estimate.parameters_bytes > 0
    assert estimate.estimated_total_bytes > estimate.parameters_bytes
