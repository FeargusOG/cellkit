"""Transformer building blocks used throughout the model package."""

import torch
from cellkit.model import mlp
from cellkit.model import norm


class Attention(torch.nn.Module):
    """Multi-head self-attention implemented with PyTorch SDPA."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
    ):
        """Initialize the self-attention projections.

        Args:
            d_model: Width of the model hidden states.
            num_heads: Number of attention heads.
            attn_dropout: Dropout probability applied to attention weights during training.
        """
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = d_model // num_heads
        self.query = torch.nn.Linear(d_model, d_model, bias=False)
        self.key = torch.nn.Linear(d_model, d_model, bias=False)
        self.value = torch.nn.Linear(d_model, d_model, bias=False)
        self.out = torch.nn.Linear(d_model, d_model, bias=False)
        for w in [
            self.query.weight,
            self.key.weight,
            self.value.weight,
            self.out.weight,
        ]:
            torch.nn.init.xavier_uniform_(w)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Apply self-attention to ``x`` with an optional boolean mask.

        Args:
            x: Input tensor with shape ``(batch, seq_len, d_model)``.
            mask: Optional tensor with shape ``(batch, seq_len)`` where ``True`` marks
                positions that should be masked out.
            causal: Whether to apply causal masking in addition to the provided mask.

        Returns:
            Tensor with shape ``(batch, seq_len, d_model)``.
        """
        batch_size, seq_len, _ = x.shape
        # Compute Q, K, V
        q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, H, T, D)
        k = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Flash Attention (scaled_dot_product_attention)
        # Flash Attention v2 is automatically used by PyTorch if supported by hardware.
        # Otherwise, it falls back to a fused or standard implementation.
        if mask is not None:
            # mask is expected as (B, T) with True/1 = masked and False/0 = visible.
            if mask.ndim != 2:
                raise ValueError("mask must have shape (batch, seq_len)")
            if mask.shape != (batch_size, seq_len):
                raise ValueError("mask shape must match x as (batch, seq_len)")
            mask = (~mask.to(dtype=torch.bool))[:, None, None, :]  # (B, 1, 1, T)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,  # shape: (B, 1, T, T) or broadcastable
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=causal,
        )
        # Merge heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.out(attn_output)


class TransformerLayer(torch.nn.Module):
    """Pre-norm transformer block with attention and a gated MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        """Initialize one transformer layer.

        Args:
            d_model: Width of the input and output hidden states.
            num_heads: Number of attention heads.
            attn_dropout: Dropout probability applied inside attention.
            resid_dropout: Dropout probability applied to residual branch outputs.
        """
        super().__init__()
        self.mlp = mlp.GatedMLP(d_model, 4 * d_model)
        self.norm_mlp = norm.RMSNorm(d_model)
        self.attn = Attention(d_model, num_heads, attn_dropout=attn_dropout)
        self.resid_dropout = torch.nn.Dropout(resid_dropout)
        self.norm_att = norm.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Apply attention and feed-forward sublayers with residual connections.

        Args:
            x: Hidden states with shape ``(batch, seq_len, d_model)``.
            mask: Optional tensor with shape ``(batch, seq_len)`` where ``True`` marks
                positions that should be masked out.
            causal: Whether to apply causal masking in the attention block.

        Returns:
            Tensor with shape ``(batch, seq_len, d_model)``.
        """
        x_att = self.attn(self.norm_att(x), mask, causal)
        h = x + self.resid_dropout(x_att)
        out = h + self.resid_dropout(self.mlp(self.norm_mlp(h)))
        return out


class Transformer(torch.nn.Module):
    """Stack of transformer layers sharing a common mask and causal setting."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        """Initialize a transformer encoder stack.

        Args:
            d_model: Width of the hidden states passed between layers.
            num_layers: Number of transformer layers to apply.
            num_heads: Number of attention heads in each layer.
            attn_dropout: Dropout probability applied inside attention.
            resid_dropout: Dropout probability applied to residual branch outputs.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    d_model,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Run the input sequence through each transformer layer in order.

        Args:
            x: Hidden states with shape ``(batch, seq_len, d_model)``.
            mask: Optional tensor with shape ``(batch, seq_len)`` where ``True`` marks
                positions that should be masked out.
            causal: Whether to apply causal masking in every layer.

        Returns:
            Tensor with shape ``(batch, seq_len, d_model)`` after all layers.
        """
        for layer in self.layers:
            x = layer(x, mask=mask, causal=causal)
        return x
