import torch

from cellkit.model.norm import RMSNorm


def test_rmsnorm_forward_shape_dtype_and_finite_values():
    model = RMSNorm(d_model=8)
    model.eval()
    x = torch.randn(2, 5, 8, dtype=torch.float32)

    out = model(x)

    assert out.shape == (2, 5, 8)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_rmsnorm_matches_formula_with_default_weight():
    d_model = 8
    eps = 1e-6
    model = RMSNorm(d_model=d_model, eps=eps)
    model.eval()
    x = torch.randn(3, 8, dtype=torch.float32)

    out = model(x)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_rmsnorm_applies_learnable_weight():
    model = RMSNorm(d_model=4, eps=1e-6)
    model.eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + model.eps)
    out = model(x)

    assert torch.allclose(out, normalized * model.weight, atol=1e-6, rtol=1e-6)


def test_rmsnorm_backward_pass():
    model = RMSNorm(d_model=8)
    x = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)

    out = model(x)
    out.sum().backward()

    assert x.grad is not None
    assert model.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.weight.grad).all()
