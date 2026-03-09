import torch

class TokenEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embsize: int,
        dropout: float,
        padding_idx: int,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings, embsize, padding_idx),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(embsize),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class ScalarEmbedding(torch.nn.Module):
    def __init__(self, embsize: int, dropout: float, max_value: int | None = None):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, embsize),
            torch.nn.ReLU(),
            torch.nn.Linear(embsize, embsize),
            torch.nn.LayerNorm(embsize),
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
