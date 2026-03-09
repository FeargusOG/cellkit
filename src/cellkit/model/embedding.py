"""Embedding layers used to encode token and scalar model inputs."""

import torch


class TokenEmbedding(torch.nn.Module):
    """Embed token ids into the model hidden space."""

    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        dropout: float,
        padding_idx: int,
    ):
        """Initialize the token embedding stack.

        Args:
            num_embeddings: Number of entries in the token vocabulary.
            d_model: Width of the embedding vectors and output hidden states.
            dropout: Dropout probability applied after the embedding lookup.
            padding_idx: Token id reserved for padding, kept fixed in the embedding table.
        """
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings, d_model, padding_idx),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map token ids to hidden states.

        Args:
            x: Tensor of token ids with shape ``(batch, seq_len)``.

        Returns:
            Tensor of embedded token representations with shape ``(batch, seq_len, d_model)``.
        """
        return self.layers(x)


class ScalarEmbedding(torch.nn.Module):
    """Project scalar features into the model hidden space."""

    def __init__(self, d_model: int, dropout: float, max_value: float | None = None):
        """Initialize the scalar embedding MLP.

        Args:
            d_model: Width of the hidden representation produced for each scalar input.
            dropout: Dropout probability applied after projection.
            max_value: Optional upper bound applied to inputs before projection.
        """
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model),
            torch.nn.Dropout(p=dropout),
        )
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed scalar inputs with optional clipping before projection.

        Args:
            x: Scalar features with shape ``(batch, seq_len)``.

        Returns:
            Tensor of scalar embeddings with shape ``(batch, seq_len, d_model)``.
        """
        # Expand the last dimension to ``(batch, seq_len, 1)`` for the MLP.
        x = x.unsqueeze(-1)
        # Cap large values before projection when a maximum is configured.
        if self.max_value is not None:
            x = torch.clamp(x, max=self.max_value)
        return self.layers(x)
