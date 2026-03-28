from __future__ import annotations

import torch

TRUNCATION_MODES = {"topk", "random"}


def select_positions(
    scores: torch.Tensor,
    *,
    k: int,
    mode: str,
) -> torch.Tensor:
    if mode not in TRUNCATION_MODES:
        raise ValueError(
            "mode must be one of: " + ", ".join(sorted(TRUNCATION_MODES))
        )
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D tensor")
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > scores.shape[0]:
        raise ValueError("k must be <= len(scores)")

    if mode == "topk":
        return torch.topk(scores, k=k, largest=True, sorted=True).indices

    return torch.randperm(scores.shape[0], device=scores.device)[:k]
