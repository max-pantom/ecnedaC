import pytest
import torch
import torch.nn.functional as F

from cadence.training.contrastive import evaluate_retrieval, info_nce_loss


def test_info_nce_is_symmetric_and_matched_retrieval_is_perfect() -> None:
    embeddings = F.normalize(torch.eye(6), dim=-1)
    loss, logits = info_nce_loss(embeddings, embeddings)
    reverse_loss, reverse_logits = info_nce_loss(embeddings, embeddings)
    assert torch.allclose(loss, reverse_loss)
    assert torch.allclose(logits, reverse_logits.t())
    metrics = evaluate_retrieval(embeddings, embeddings)
    assert metrics.video_to_audio.recall_at_1 == 1.0
    assert metrics.audio_to_video.recall_at_5 == 1.0
    assert metrics.chance == pytest.approx(1 / 6)


def test_info_nce_requires_negatives() -> None:
    with pytest.raises(ValueError, match="at least two"):
        info_nce_loss(torch.ones(1, 2), torch.ones(1, 2))

