"""Prediction heads built on top of sequence model hidden states."""

import torch


class ScalarRegressionHead(torch.nn.Module):
    """Predict one scalar value per sequence position."""

    def __init__(
        self,
        d_model: int
    ):
        """Initialize the regression head.

        Args:
            d_model: Width of the input hidden states consumed by the head.
        """
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return regression outputs for each sequence position.

        Args:
            x: Hidden states with shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of predictions with shape ``(batch, seq_len)``.
        """
        return self.fc(x).squeeze(-1)
