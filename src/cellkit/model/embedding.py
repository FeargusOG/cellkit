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
        self.embedding = torch.nn.Embedding(num_embeddings, embsize, padding_idx)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.enc_norm = torch.nn.LayerNorm(embsize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.dropout(x)
        x = self.enc_norm(x)
        return x
    
class ScalarEmbedding(torch.nn.Module):
    def __init__(self, embsize: int, dropout: float, max_value: int | None = None):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear1 = torch.nn.Linear(1, embsize)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(embsize, embsize)
        self.norm = torch.nn.LayerNorm(embsize)
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expand last dimension i.e. => [batch_size, seq_len, 1]
        x = x.unsqueeze(-1)
        # optionally clip x to [-inf, max_value]
        if self.max_value is not None:
            x = torch.clamp(x, max=self.max_value)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)
