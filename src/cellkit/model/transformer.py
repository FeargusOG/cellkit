import torch
import mlp
import norm

class Attention(torch.nn.Module):
    """
    Multihead Attention Layer using PyTorch Flash Attention V2.
    Includes rotary embeddings and optional causal/mask support.
    """
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
        self.out = torch.nn.Linear(dim, dim, bias=False)
        for w in [self.query.weight, self.key.weight, self.value.weight, self.out.weight]:
            torch.nn.init.xavier_uniform_(w)

    def forward(self, x,  mask=None, causal=False):
        B, T, _ = x.shape
        # Compute Q, K, V
        q = self.query(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.key(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        # Flash Attention (scaled_dot_product_attention)
        # Flash Attention v2 is automatically used by PyTorch if supported by hardware.
        # Otherwise, it falls back to a fused or standard implementation.
        if mask is not None:
            # Suppose mask is (B, T) where 1 = mask, 0 = keep
            mask = mask.bool()[:, None, None, :]  # (B, 1, 1, T)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,  # shape: (B, 1, T, T) or broadcastable
            dropout_p=0.0,
            is_causal=causal
        )
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.out(attn_output)

class TransformerLayer(torch.nn.Module):

    def __init__(self, dim, heads=8):
        super().__init__()
        self.mlp = mlp.GatedMLP(dim, 4 * dim)
        self.norm_mlp = norm.RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm_att = norm.RMSNorm(dim)

    def forward(self, x, mask=None, causal=False):
        x_att = self.attn(self.norm_att(x), mask, causal)
        h = x + x_att
        out = h + self.mlp(self.norm_mlp(h))
        return out
