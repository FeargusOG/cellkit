from __future__ import annotations

import torch

MASK_STRATEGIES = {"uniform", "nonzero_only", "nonzero_preferred"}


def build_prediction_mask(
    values: torch.Tensor,
    *,
    mask_prob: float,
    padding_mask: torch.Tensor,
    strategy: str = "uniform",
    min_masked_positions: int = 1,
) -> torch.Tensor:
    if strategy not in MASK_STRATEGIES:
        raise ValueError(
            "strategy must be one of: uniform, nonzero_only, nonzero_preferred"
        )
    if min_masked_positions < 0:
        raise ValueError("min_masked_positions must be >= 0")

    valid_mask = ~padding_mask
    nonzero_mask = valid_mask & (values != 0)
    sample_probs = _sample_probs(
        valid_mask=valid_mask,
        nonzero_mask=nonzero_mask,
        mask_prob=mask_prob,
        strategy=strategy,
    )
    prediction_mask = torch.rand_like(values, dtype=torch.float32) < sample_probs
    prediction_mask = prediction_mask & valid_mask

    if min_masked_positions > 0:
        prediction_mask = _ensure_min_masked_positions(
            prediction_mask=prediction_mask,
            valid_mask=valid_mask,
            nonzero_mask=nonzero_mask,
            strategy=strategy,
            min_masked_positions=min_masked_positions,
        )

    return prediction_mask


def _sample_probs(
    *,
    valid_mask: torch.Tensor,
    nonzero_mask: torch.Tensor,
    mask_prob: float,
    strategy: str,
) -> torch.Tensor:
    sample_probs = torch.zeros_like(valid_mask, dtype=torch.float32)

    if strategy == "uniform":
        sample_probs[valid_mask] = mask_prob
        return sample_probs

    if strategy == "nonzero_only":
        sample_probs[nonzero_mask] = mask_prob
        return sample_probs

    valid_counts = valid_mask.sum(dim=1).to(torch.float32)
    weight_sum = valid_mask.to(torch.float32) + nonzero_mask.to(torch.float32)
    weight_sum = weight_sum.sum(dim=1)
    # Match the uniform strategy's expected masked count per sample while preferring nonzero values.
    row_targets = mask_prob * valid_counts
    scaled_probs = (
        row_targets.unsqueeze(1)
        * (valid_mask.to(torch.float32) + nonzero_mask.to(torch.float32))
        / weight_sum.unsqueeze(1).clamp_min(1.0)
    )
    sample_probs = torch.where(valid_mask, scaled_probs.clamp(max=1.0), sample_probs)
    return sample_probs


def _ensure_min_masked_positions(
    *,
    prediction_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    nonzero_mask: torch.Tensor,
    strategy: str,
    min_masked_positions: int,
) -> torch.Tensor:
    for row in range(prediction_mask.shape[0]):
        masked_count = int(prediction_mask[row].sum().item())
        if masked_count >= min_masked_positions:
            continue

        needed = min_masked_positions - masked_count
        eligible_mask = _eligible_fill_mask(
            valid_mask=valid_mask[row],
            nonzero_mask=nonzero_mask[row],
            current_mask=prediction_mask[row],
            strategy=strategy,
        )
        if not torch.any(eligible_mask):
            continue

        eligible_positions = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        chosen = eligible_positions[
            torch.randperm(eligible_positions.numel(), device=eligible_positions.device)[
                :needed
            ]
        ]
        prediction_mask[row, chosen] = True

    return prediction_mask


def _eligible_fill_mask(
    *,
    valid_mask: torch.Tensor,
    nonzero_mask: torch.Tensor,
    current_mask: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    if strategy == "uniform":
        return valid_mask & (~current_mask)

    if strategy == "nonzero_only":
        eligible_nonzero = nonzero_mask & (~current_mask)
        if torch.any(eligible_nonzero):
            return eligible_nonzero
        return valid_mask & (~current_mask)

    preferred = nonzero_mask & (~current_mask)
    if torch.any(preferred):
        return preferred
    return valid_mask & (~current_mask)
