import torch
from typing import cast

from cellkit.model.embedding import ScalarEmbedding
from cellkit.model.embedding import TokenEmbedding


def test_token_embedding_forward_shape_dtype_and_finite_values():
    model = TokenEmbedding(num_embeddings=16, d_model=8, dropout=0.0, padding_idx=0)
    model.eval()
    x = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)

    out = model(x)

    assert out.shape == (2, 3, 8)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_token_embedding_padding_idx_outputs_zero_vector():
    model = TokenEmbedding(num_embeddings=10, d_model=6, dropout=0.0, padding_idx=0)
    model.eval()
    x = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)

    out = model(x)

    assert torch.allclose(out[0], torch.zeros_like(out[0]))


def test_token_embedding_backward_skips_padding_idx_gradient():
    model = TokenEmbedding(num_embeddings=12, d_model=4, dropout=0.0, padding_idx=0)
    x = torch.tensor([[0, 1, 2], [3, 0, 4]], dtype=torch.long)

    out = model(x)
    out.sum().backward()

    embedding = cast(torch.nn.Embedding, model.layers[0])
    grad = embedding.weight.grad
    assert grad is not None
    assert torch.allclose(grad[0], torch.zeros_like(grad[0]))


def test_scalar_embedding_forward_shape_dtype_and_finite_values():
    model = ScalarEmbedding(d_model=8, dropout=0.0)
    model.eval()
    x = torch.tensor([[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float32)

    out = model(x)

    assert out.shape == (2, 3, 8)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_scalar_embedding_optional_max_value_none_behaves_without_clamp():
    torch.manual_seed(0)
    model_no_clamp = ScalarEmbedding(d_model=8, dropout=0.0, max_value=None)
    model_no_clamp.eval()
    model_with_clamp = ScalarEmbedding(d_model=8, dropout=0.0, max_value=1.5)
    model_with_clamp.load_state_dict(model_no_clamp.state_dict())
    model_with_clamp.eval()

    x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float32)

    expected = model_no_clamp(torch.clamp(x, max=1.5))
    actual = model_with_clamp(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_scalar_embedding_backward_pass():
    model = ScalarEmbedding(d_model=8, dropout=0.0, max_value=None)
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32, requires_grad=True)

    out = model(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
