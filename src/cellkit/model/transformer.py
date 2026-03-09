import torch
from cellkit.model import mlp
from cellkit.model import norm


class Attention(torch.nn.Module):
    def __init__(self, d_model: int = 512, heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        if heads <= 0:
            raise ValueError("heads must be > 0")
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.head_dim = d_model // heads
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
        B, T, _ = x.shape
        # Compute Q, K, V
        q = (
            self.query(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        )  # (B, H, T, D)
        k = self.key(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        # Flash Attention (scaled_dot_product_attention)
        # Flash Attention v2 is automatically used by PyTorch if supported by hardware.
        # Otherwise, it falls back to a fused or standard implementation.
        if mask is not None:
            # mask is expected as (B, T) with True/1 = keep and False/0 = block.
            if mask.ndim != 2:
                raise ValueError("mask must have shape (batch, seq_len)")
            if mask.shape != (B, T):
                raise ValueError("mask shape must match x as (batch, seq_len)")
            mask = mask.to(dtype=torch.bool)[:, None, None, :]  # (B, 1, 1, T)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,  # shape: (B, 1, T, T) or broadcastable
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=causal,
        )
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(attn_output)


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int = 8,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = mlp.GatedMLP(d_model, 4 * d_model)
        self.norm_mlp = norm.RMSNorm(d_model)
        self.attn = Attention(d_model, heads, attn_dropout=attn_dropout)
        self.resid_dropout = torch.nn.Dropout(resid_dropout)
        self.norm_att = norm.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        x_att = self.attn(self.norm_att(x), mask, causal)
        h = x + self.resid_dropout(x_att)
        out = h + self.resid_dropout(self.mlp(self.norm_mlp(h)))
        return out


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        heads: int = 8,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    d_model,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, causal=causal)
        return x
