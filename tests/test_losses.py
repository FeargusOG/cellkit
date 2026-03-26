import torch

from cellkit.losses import compute_masked_mae_loss, compute_masked_mse_loss


def test_compute_masked_mse_loss_uses_only_masked_positions():
    pred = torch.tensor([[1.0, 10.0, 3.0]])
    target = torch.tensor([[2.0, 0.0, 1.0]])
    mask = torch.tensor([[True, False, True]])

    loss = compute_masked_mse_loss(pred, target, mask)

    assert torch.isclose(loss, torch.tensor(2.5))


def test_compute_masked_mae_loss_uses_only_masked_positions():
    pred = torch.tensor([[1.0, 10.0, 3.0]])
    target = torch.tensor([[2.0, 0.0, 1.0]])
    mask = torch.tensor([[True, False, True]])

    loss = compute_masked_mae_loss(pred, target, mask)

    assert torch.isclose(loss, torch.tensor(1.5))
