import torch

from cellkit.masking import build_prediction_mask


def test_uniform_masking_respects_padding_and_minimum():
    torch.manual_seed(0)
    values = torch.tensor([[1.0, 2.0, 0.0]])
    padding_mask = torch.tensor([[False, False, True]])

    prediction_mask = build_prediction_mask(
        values,
        mask_prob=0.0,
        padding_mask=padding_mask,
        strategy="uniform",
        min_masked_positions=1,
    )

    assert torch.equal(prediction_mask, torch.tensor([[True, False, False]]))


def test_nonzero_only_masks_only_nonzero_when_available():
    torch.manual_seed(0)
    values = torch.tensor([[0.0, 2.0, 3.0]])
    padding_mask = torch.zeros_like(values, dtype=torch.bool)

    prediction_mask = build_prediction_mask(
        values,
        mask_prob=1.0,
        padding_mask=padding_mask,
        strategy="nonzero_only",
        min_masked_positions=1,
    )

    assert torch.equal(prediction_mask, torch.tensor([[False, True, True]]))


def test_nonzero_only_falls_back_to_valid_positions_when_no_nonzero_exist():
    torch.manual_seed(0)
    values = torch.tensor([[0.0, 0.0, 0.0]])
    padding_mask = torch.zeros_like(values, dtype=torch.bool)

    prediction_mask = build_prediction_mask(
        values,
        mask_prob=0.0,
        padding_mask=padding_mask,
        strategy="nonzero_only",
        min_masked_positions=1,
    )

    assert prediction_mask.sum().item() == 1


def test_nonzero_preferred_prefers_nonzero_positions_for_minimum_fill():
    torch.manual_seed(0)
    values = torch.tensor([[0.0, 4.0, 0.0, 5.0]])
    padding_mask = torch.zeros_like(values, dtype=torch.bool)

    prediction_mask = build_prediction_mask(
        values,
        mask_prob=0.0,
        padding_mask=padding_mask,
        strategy="nonzero_preferred",
        min_masked_positions=2,
    )

    assert torch.equal(prediction_mask, torch.tensor([[False, True, False, True]]))


def test_masking_rejects_invalid_strategy():
    values = torch.tensor([[1.0, 2.0]])
    padding_mask = torch.zeros_like(values, dtype=torch.bool)

    try:
        build_prediction_mask(
            values,
            mask_prob=0.1,
            padding_mask=padding_mask,
            strategy="unsupported",
        )
    except ValueError as exc:
        assert "strategy must be one of" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid strategy")
