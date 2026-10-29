import torch


class AbsoluteNormalObservation:
    """Gaussian observation model with mean ``abs(Hx)``.

    Unlike a folded-normal random variable, the absolute value is applied to
    the noise-free projection.  Thus the model is

        y = abs(Hx) + epsilon,  epsilon ~ N(0, R),

    which deliberately makes opposite projected states observationally
    indistinguishable.
    """

    def __init__(self, config, device):
        self.R = torch.as_tensor(config["R"], device=device, dtype=torch.float32)
        self.R_det = torch.linalg.det(self.R)
        self.R_inv = torch.linalg.inv(self.R)
        self.R_chol = torch.linalg.cholesky(self.R)
        self.H = torch.as_tensor(config["H"], device=device, dtype=torch.float32)
        self.dimension = len(config["H"])

    def observation_mean(self, x):
        """Return the ambiguous observation mean ``abs(Hx)``."""
        return torch.abs(
            torch.einsum("od,bd->bo", self.H.to(dtype=x.dtype), x)
        )

    def measure(self, x):
        """Sample ``y = abs(Hx) + noise`` for a batch of states."""
        noise = torch.randn(
            (x.shape[0], self.dimension), device=x.device, dtype=x.dtype
        ) @ self.R_chol.to(dtype=x.dtype).T
        return self.observation_mean(x) + noise

    def log_g(self, x, y):
        """Evaluate the normalized Gaussian log likelihood ``log p(y | x)``."""
        residual = self.observation_mean(x) - y
        quadratic = torch.einsum(
            "bi,ij,bj->b", residual, self.R_inv.to(dtype=x.dtype), residual
        )
        normalizer = torch.log(
            (2 * torch.pi) ** self.dimension * self.R_det.to(dtype=x.dtype)
        )
        return -0.5 * (quadratic + normalizer)
