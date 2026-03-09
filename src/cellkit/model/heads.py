import torch

class ScalarRegressionHead(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_model),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (batch, seq_len)