import pytest
import torch

from cellkit.data.truncation import TRUNCATION_MODES
from cellkit.data.truncation import select_positions


def test_select_positions_topk_returns_descending_score_positions():
    scores = torch.tensor([3.0, 7.0, 2.0], dtype=torch.float32)

    positions = select_positions(scores, k=2, mode="topk")

    assert torch.equal(positions, torch.tensor([1, 0]))


def test_select_positions_random_returns_k_positions():
    torch.manual_seed(0)
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    positions = select_positions(scores, k=2, mode="random")

    assert torch.equal(positions, torch.tensor([0, 1]))


def test_select_positions_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        select_positions(torch.tensor([1.0]), k=1, mode="unsupported")


def test_select_positions_rejects_non_vector_scores():
    with pytest.raises(ValueError, match="scores must be a 1D tensor"):
        select_positions(torch.tensor([[1.0, 2.0]]), k=1, mode="topk")


def test_select_positions_rejects_invalid_k():
    scores = torch.tensor([1.0, 2.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="k must be > 0"):
        select_positions(scores, k=0, mode="topk")

    with pytest.raises(ValueError, match="k must be <= len\\(scores\\)"):
        select_positions(scores, k=3, mode="topk")


def test_truncation_modes_constant_matches_supported_modes():
    assert TRUNCATION_MODES == {"topk", "random"}
