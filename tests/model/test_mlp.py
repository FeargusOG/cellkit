import torch

from cellkit.model.mlp import GatedMLP


def test_gated_mlp_forward_shape_dtype_and_finite_values():
    model = GatedMLP(d_model=8, hidden_dim=16)
    model.eval()
    x = torch.randn(2, 5, 8, dtype=torch.float32)

    out = model(x)

    assert out.shape == (2, 5, 8)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_gated_mlp_matches_explicit_formula():
    model = GatedMLP(d_model=8, hidden_dim=16)
    model.eval()
    x = torch.randn(3, 8, dtype=torch.float32)

    out = model(x)
    expected = model.down_proj(
        torch.nn.functional.silu(model.gate_proj(x)) * model.up_proj(x)
    )

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_gated_mlp_backward_pass():
    model = GatedMLP(d_model=8, hidden_dim=16)
    x = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)

    out = model(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
