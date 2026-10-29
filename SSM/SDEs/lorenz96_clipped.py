import torch

from SSM.SDEs.lorenz96 import Lorenz96


class Lorenz96Clipped(Lorenz96):
    """Lorenz-96 with norm-clipped drift inputs, not a hard state constraint."""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.max_norm = float(config.get("max_norm", 20.0))
        if not 0 < self.max_norm < float("inf"):
            raise ValueError("max_norm must be finite and positive")

    def drift(self, t, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        x = x * (self.max_norm / norm.clamp_min(self.max_norm))
        return super().drift(t, x)
