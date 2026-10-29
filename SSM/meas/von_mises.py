import torch


class VonMisesObservation:
    """A periodic latent phase observed with von Mises noise.

    The sampled angle is returned as ``(cos(angle), sin(angle))`` so the
    controller input has no artificial discontinuity at the angular boundary.
    """

    def __init__(self, config, device):
        self.H = torch.as_tensor(config["H"], device=device, dtype=torch.float32)
        if self.H.shape[0] != 1:
            raise ValueError("VonMisesObservation currently supports one latent phase")
        self.period = torch.as_tensor(config["period"], device=device, dtype=torch.float32)
        self.concentration = torch.as_tensor(
            config["concentration"], device=device, dtype=torch.float32
        )
        if self.period.item() <= 0 or self.concentration.item() <= 0:
            raise ValueError("period and concentration must be positive")
        self.dimension = 2

    def phase(self, x):
        projection = torch.einsum("od,bd->bo", self.H.to(dtype=x.dtype), x)
        return 2 * torch.pi * projection[:, 0] / self.period.to(dtype=x.dtype)

    def measure(self, x):
        phase = self.phase(x)
        concentration = self.concentration.to(dtype=x.dtype).expand_as(phase)
        angle = torch.distributions.VonMises(phase, concentration).sample()
        return torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)

    def log_g(self, x, y):
        if y.shape[1] != 2:
            raise ValueError("von Mises observations must be (cos(angle), sin(angle))")
        angle = torch.atan2(y[:, 1], y[:, 0])
        phase = self.phase(x)
        concentration = self.concentration.to(dtype=x.dtype).expand_as(phase)
        return torch.distributions.VonMises(phase, concentration).log_prob(angle)
