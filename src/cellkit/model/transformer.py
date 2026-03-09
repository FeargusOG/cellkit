import torch
from cellkit.model import mlp
from cellkit.model import norm


class Attention(torch.nn.Module):
    def __init__(self, d_model=512, heads=8):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
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

    def forward(self, x, mask=None, causal=False):
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
            # mask is (B, T) where 1 = keep and 0 = masked/blocked
            mask = mask.bool()[:, None, None, :]  # (B, 1, 1, T)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,  # shape: (B, 1, T, T) or broadcastable
            dropout_p=0.0,
            is_causal=causal,
        )
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(attn_output)


class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model, heads=8):
        super().__init__()
        self.mlp = mlp.GatedMLP(d_model, 4 * d_model)
        self.norm_mlp = norm.RMSNorm(d_model)
        self.attn = Attention(d_model, heads)
        self.norm_att = norm.RMSNorm(d_model)

    def forward(self, x, mask=None, causal=False):
        x_att = self.attn(self.norm_att(x), mask, causal)
        h = x + x_att
        out = h + self.mlp(self.norm_mlp(h))
        return out


class Transformer(torch.nn.Module):
    def __init__(self, d_model, layers, heads=8):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [TransformerLayer(d_model, heads=heads) for _ in range(layers)]
        )

    def forward(self, x, mask=None, causal=False):
        for layer in self.layers:
            x = layer(x, mask=mask, causal=causal)
        return x
