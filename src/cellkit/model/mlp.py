import torch

class GatedMLP(torch.nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate_proj = torch.nn.Linear(d_model, hidden_dim)
        self.up_proj = torch.nn.Linear(d_model, hidden_dim)
        self.down_proj = torch.nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )