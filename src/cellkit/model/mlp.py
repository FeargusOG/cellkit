"""Feed-forward blocks used inside transformer layers."""

import torch


class GatedMLP(torch.nn.Module):
    """SwiGLU-style gated MLP projection."""

    def __init__(self, d_model, hidden_dim):
        """Initialize the gated MLP.

        Args:
            d_model: Width of the input and output hidden states.
            hidden_dim: Width of the intermediate gated projection.
        """
        super().__init__()
        self.gate_proj = torch.nn.Linear(d_model, hidden_dim)
        self.up_proj = torch.nn.Linear(d_model, hidden_dim)
        self.down_proj = torch.nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        """Expand, gate, and project hidden states back to ``d_model``.

        Args:
            x: Hidden states with shape ``(..., d_model)``.

        Returns:
            Tensor with the same leading dimensions as ``x`` and width ``d_model``.
        """
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )
