import torch

from cellkit.model.heads import ScalarRegressionHead


def test_scalar_regression_head_forward_3d_shape_dtype_and_finite_values():
    model = ScalarRegressionHead(d_model=8)
    model.eval()
    x = torch.randn(2, 5, 8, dtype=torch.float32)

    out = model(x)

    assert out.shape == (2, 5)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_scalar_regression_head_forward_2d_shape():
    model = ScalarRegressionHead(d_model=8)
    model.eval()
    x = torch.randn(4, 8, dtype=torch.float32)

    out = model(x)

    assert out.shape == (4,)


def test_scalar_regression_head_backward_pass():
    model = ScalarRegressionHead(d_model=8)
    x = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)

    out = model(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
