import torch


class ZeroPE(torch.nn.Module):
    def __init__(self, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.device = device

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (position_ids.shape[0], position_ids.shape[1], self.embed_dim)
        ).to(self.device)
