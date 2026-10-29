import torch
from SSM.SDEs.SDE import SDE


class OrnsteinUhlenbeck(SDE):
    def __init__(self, config, device):
        dimension_state = len(config["A"])
        dimension_noise = len(config["Sigma"][0])
        super().__init__(dimension_state, dimension_noise)
        self.A = torch.tensor(config["A"], device=device, dtype=torch.float32)            # (d × d)
        self.mu = torch.tensor(config["mu"], device=device, dtype=torch.float32)          # (d,)
        self.Sigma = torch.tensor(config["Sigma"], device=device, dtype=torch.float32)    # (d × m)

        self.d = dimension_state
        self.m = dimension_noise

    def drift(self, t, x):
        return torch.einsum("ij,bj->bi", self.A, self.mu - x)

    def diffusion(self, t, x):
        temp = self.Sigma.unsqueeze(0)
        temp = temp.repeat_interleave(x.shape[0], 0)
        return temp
