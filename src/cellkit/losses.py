from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked_pred = pred[mask]
    masked_target = target[mask]
    return F.mse_loss(masked_pred, masked_target)


def compute_masked_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked_pred = pred[mask]
    masked_target = target[mask]
    return F.l1_loss(masked_pred, masked_target)
