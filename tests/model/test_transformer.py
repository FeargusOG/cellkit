import torch
import pytest

from cellkit.model.transformer import Attention
from cellkit.model.transformer import Transformer
from cellkit.model.transformer import TransformerLayer


def test_attention_forward_shape_dtype_and_finite_values():
    model = Attention(d_model=16, heads=4)
    model.eval()
    x = torch.randn(2, 6, 16, dtype=torch.float32)

    out = model(x)

    assert out.shape == (2, 6, 16)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_attention_raises_when_d_model_not_divisible_by_heads():
    with pytest.raises(ValueError, match="d_model must be divisible by heads"):
        Attention(d_model=10, heads=3)


def test_attention_raises_when_heads_is_non_positive():
    with pytest.raises(ValueError, match="heads must be > 0"):
        Attention(d_model=16, heads=0)


def test_attention_all_zeros_mask_matches_no_mask():
    torch.manual_seed(0)
    model = Attention(d_model=16, heads=4)
    model.eval()
    x = torch.randn(2, 5, 16, dtype=torch.float32)
    zero_mask = torch.zeros(2, 5, dtype=torch.long)

    out_no_mask = model(x, mask=None, causal=False)
    out_zero_mask = model(x, mask=zero_mask, causal=False)

    assert torch.allclose(out_no_mask, out_zero_mask, atol=1e-6, rtol=1e-6)


def test_attention_all_ones_mask_blocks_attention():
    torch.manual_seed(0)
    model = Attention(d_model=16, heads=4)
    model.eval()
    x = torch.randn(2, 5, 16, dtype=torch.float32)
    ones_mask = torch.ones(2, 5, dtype=torch.long)

    out = model(x, mask=ones_mask, causal=False)

    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6, rtol=1e-6)


def test_attention_bool_and_integer_masks_are_equivalent():
    torch.manual_seed(0)
    model = Attention(d_model=16, heads=4)
    model.eval()
    x = torch.randn(2, 5, 16, dtype=torch.float32)
    int_mask = torch.tensor([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]], dtype=torch.long)
    bool_mask = int_mask.bool()

    out_int = model(x, mask=int_mask, causal=False)
    out_bool = model(x, mask=bool_mask, causal=False)

    assert torch.allclose(out_int, out_bool, atol=1e-6, rtol=1e-6)


def test_attention_invalid_mask_shape_raises():
    model = Attention(d_model=16, heads=4)
    x = torch.randn(2, 5, 16, dtype=torch.float32)
    wrong_rank = torch.ones(2, 1, 5, dtype=torch.bool)
    wrong_width = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask must have shape"):
        model(x, mask=wrong_rank)
    with pytest.raises(ValueError, match="mask shape must match x"):
        model(x, mask=wrong_width)


def test_transformer_layer_with_zeroed_weights_is_identity():
    model = TransformerLayer(d_model=8, heads=2)
    model.eval()
    for param in model.parameters():
        torch.nn.init.zeros_(param)

    x = torch.randn(2, 4, 8, dtype=torch.float32)
    out = model(x)

    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_transformer_forward_shape_dtype_and_finite_values():
    model = Transformer(d_model=12, layers=3, heads=3)
    model.eval()
    x = torch.randn(2, 7, 12, dtype=torch.float32)
    mask = torch.ones(2, 7, dtype=torch.long)

    out = model(x, mask=mask, causal=True)

    assert out.shape == (2, 7, 12)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_transformer_backward_pass():
    model = Transformer(d_model=8, layers=2, heads=2)
    x = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)

    out = model(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
