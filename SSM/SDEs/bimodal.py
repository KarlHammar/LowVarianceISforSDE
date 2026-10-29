import torch
from SSM.SDEs.SDE import SDE


class Bimodal(SDE):
    def __init__(self, config, device):
        super().__init__(dimension_state=1, dimension_noise=1)
        self.a = torch.tensor(config['a'], device=device, dtype=torch.float32)
        self.b = torch.tensor(config['b'], device=device, dtype=torch.float32)
        self.Sigma = torch.tensor(config['Sigma'], device=device, dtype=torch.float32)
        self.Sigma = self.Sigma.reshape(1,1)  # make sure it's (1, m)

    def drift(self, t, x):
        return self.a*x**3 + self.b*x

    def diffusion(self, t, x):
        temp = self.Sigma.unsqueeze(0)
        temp = temp.repeat_interleave(x.shape[0], 0)
        # print( temp.shape)
        return temp
