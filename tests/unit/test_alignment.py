import torch

from cadence.alignment.temporal import SharedTemporalAdapter
from cadence.encoders.types import TemporalEncoding


def encoding(times: list[float], mask: list[bool], offset: float = 0.0) -> TemporalEncoding:
    tokens = torch.tensor([[[time + offset, 2 * time] for time in times]])
    boolean_mask = torch.tensor([mask])
    return TemporalEncoding(
        tokens=tokens,
        timestamps=torch.tensor([times]),
        mask=boolean_mask,
        global_embedding=torch.ones(1, 2),
    )


def test_timestamp_interpolation_and_masked_range() -> None:
    adapter = SharedTemporalAdapter()
    video = encoding([0.0, 0.5, 1.0], [True, True, True])
    audio = encoding([0.25, 0.75, 1.25], [True, True, False], offset=1.0)
    grid = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    aligned = adapter(video, audio, grid)
    assert aligned.video_mask.tolist() == [[True, True, True, True, True]]
    assert aligned.audio_mask.tolist() == [[False, True, True, True, False]]
    assert torch.allclose(aligned.video_tokens[0, 1], torch.tensor([0.25, 0.5]))
    assert torch.equal(aligned.audio_tokens[0, 0], torch.zeros(2))


def test_fully_invalid_encoding_stays_zero() -> None:
    adapter = SharedTemporalAdapter()
    invalid = encoding([0.0, 1.0], [False, False])
    result = adapter(invalid, invalid, torch.tensor([[0.0, 0.5]]))
    assert not result.video_mask.any()
    assert not result.video_tokens.any()

