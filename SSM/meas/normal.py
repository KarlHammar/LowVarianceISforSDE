import torch


class NormalObservation:
    """
    Class for normal observation model
    """

    def __init__(self, config, device):
        self.R          = torch.tensor(config['R']).to(device).float()
        self.R_det      = torch.linalg.det(self.R).to(device).float()
        self.R_inv      = torch.linalg.inv(self.R).to(device).float()
        self.R_chol     = torch.linalg.cholesky(self.R).to(device).float()
        self.H          = torch.tensor(config['H']).to(device).float()
        self.dimension  = len(config['H'])

    def measure(self, x):
        """
        Sample observations for batched x as: y = Hx + noise.

        Parameters
        ----------
        x : torch.Tensor
            State vectors of shape (Batch-size, state dimension)
        """
        N_batch = x.shape[0]
        noise = torch.randn(
            (N_batch, self.dimension), device=x.device, dtype=x.dtype
        ) @ self.R_chol.T
        return torch.einsum('ij,Bj->Bi', self.H, x) + noise

    def log_g(self, x, y):
        """
        Log-likelihood function: log p(y|x)

        Parameters
        ----------
        x : torch.Tensor
            State vector of shape (Batch-size, state dimension)
        y : torch.Tensor
            Observation vector of shape (Batch-size, observation dimension)
        """
        # Measurement model is y = H*x + noise
        ypred = torch.einsum('od,bd->bo', self.H, x)

        return -0.5 * torch.einsum('bi,ij,bj->b', ypred - y, self.R_inv, ypred - y) \
                - 0.5 * torch.log((2 * torch.pi)**(self.dimension)*self.R_det)
