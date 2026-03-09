import torch

class TokenEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        dropout: float,
        padding_idx: int,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings, d_model, padding_idx),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class ScalarEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float, max_value: int | None = None):
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
        # expand last dimension i.e. => [batch_size, seq_len, 1]
        x = x.unsqueeze(-1)
        # optionally clip x to [-inf, max_value]
        if self.max_value is not None:
            x = torch.clamp(x, max=self.max_value)
        return self.layers(x)
