"""Normalization layers used by the model stack."""

import torch


class RMSNorm(torch.nn.Module):
    """Root-mean-square normalization with a learned scale."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        """Initialize RMS normalization parameters.

        Args:
            d_model: Width of the final tensor dimension to normalize.
            eps: Small constant added for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def _norm(self, x):
        """Normalize activations by their per-token RMS.

        Args:
            x: Tensor whose last dimension matches ``d_model``.

        Returns:
            RMS-normalized tensor with the same shape as ``x``.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMS normalization while preserving the input dtype.

        Args:
            x: Tensor of activations to normalize.

        Returns:
            Tensor with the same shape and dtype as ``x``.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.to(dtype=output.dtype)
