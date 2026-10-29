import math

import torch

from SSM.SDEs.ou import OrnsteinUhlenbeck
from SSM.SDEs.bimodal import Bimodal
from SSM.SDEs.lorenz63 import Lorenz63
from SSM.SDEs.lorenz96 import Lorenz96
from SSM.SDEs.lorenz96_clipped import Lorenz96Clipped
from SSM.meas.normal import NormalObservation
from SSM.meas.absolute_normal import AbsoluteNormalObservation
from SSM.meas.von_mises import VonMisesObservation


class SSM:

    def __init__(
        self,
        ssm_config,
        device
    ):
        self.device = device
        self.dt = ssm_config["dt"]
        self.deltaT_meas = ssm_config["deltaT_meas"]
        self.num_meas = ssm_config["num_meas"]
        self.init_config = ssm_config["init_config"]
        self.T = self.num_meas * self.deltaT_meas
        self.sde = initialize_sde(ssm_config["sde_config"], device)
        self.meas = initialize_observation_model(ssm_config["meas_config"], device)
        if self.init_config["name"] == "stationary_empirical":
            self.stationary_bank = self._get_stationary_bank()

    def sample_initial_states(self, batch_size, device, dtype=torch.float32):
        mean = self.get_initial_mean(batch_size, device, dtype=dtype)
        name = self.init_config["name"]

        if name == "stationary_empirical":
            indices = torch.randint(self.stationary_bank.shape[0], (batch_size,), device=device)
            return self.stationary_bank[indices]

        if name == "fixed":
            return mean
        if name == "gaussian":
            cov = torch.as_tensor(self.init_config["cov"], device=device, dtype=dtype)
            noise = torch.randn(batch_size, self.sde.dimension_state, device=device, dtype=dtype)
            return mean + noise @ torch.linalg.cholesky(cov).T

        raise ValueError(f"Unknown initial-state sampler: {name}")

    @torch.no_grad()
    def _get_stationary_bank(self):
        """Approximate the simulator's invariant law with burn-in endpoints."""
        print("Generating stationary bank...")
        size = self.init_config["bank_size"]
        burn_in = self.init_config["burn_in"]
        steps = max(1, math.ceil(burn_in / self.dt))
        x = self.get_initial_mean(size, self.device)
        for ii in range(steps):
            noise = torch.randn(
                size, self.sde.dimension_noise, device=self.device
            ) * self.dt ** 0.5
            x = x + self.sde.drift(ii * self.dt, x) * self.dt + torch.einsum(
                "Bdm,Bm->Bd", self.sde.diffusion(ii * self.dt, x), noise,
            )
        print(f"Stationary bank generated with {steps} steps of burn-in.")
        return x

    def get_initial_mean(self, batch_size, device, dtype=torch.float32):
        mean = torch.as_tensor(self.init_config["mean"], device=device, dtype=dtype)
        return mean.unsqueeze(0).repeat(batch_size, 1)

    def sample_observed_trajectories(
        self,
        num_trajectories,
        dtype=torch.float32,
        return_noise=False,
    ):
        x0 = self.sample_initial_states(num_trajectories, device=self.device, dtype=dtype)
        simulated = self.simulate_paths(x0, self.T, return_noise=return_noise)
        if return_noise:
            x, W = simulated
        else:
            x = simulated
        y, yt = self.simulate_observations(x)
        if return_noise:
            return x, y, yt, W
        return x, y, yt

    def simulate_paths(self, x0, T, return_noise=False):
        """
        Simulate latent paths using Euler-Maruyama.

        Parameters
        ----------
        x0 : torch.Tensor
            Initial states of shape (N_paths, dimension_state).
        T : float
            Final time.

        Returns
        -------
        torch.Tensor
            Simulated paths of shape (N_paths, N_steps + 1, dimension_state).
        """
        N_step = int(round(T / self.dt))
        N_paths = x0.shape[0]
        dim_state = self.sde.dimension_state
        dim_noise = self.sde.dimension_noise
        device = x0.device
        dtype = x0.dtype

        x = torch.zeros(N_paths, N_step + 1, dim_state, device=device, dtype=dtype)
        x[:, 0, :] = x0

        W = torch.randn(N_paths, N_step, dim_noise, device=device, dtype=dtype) * self.dt ** 0.5

        for ii in range(N_step):
            t = ii * self.dt
            w = W[:, ii, :]
            xi = x[:, ii, :]

            base_drift = self.sde.drift(t, xi)
            sigma = self.sde.diffusion(t, xi)
            diffusion = torch.einsum('Bdm,Bm->Bd', sigma, w)
            x[:, ii + 1, :] = xi + base_drift * self.dt + diffusion

        if return_noise:
            return x, W
        return x

    def simulate_observations(self, x):
        N_obs = int(round((x.shape[1] - 1) * self.dt / self.deltaT_meas))
        N_batch = x.shape[0]
        obs_t_vec = torch.linspace(
            self.deltaT_meas,
            N_obs * self.deltaT_meas,
            N_obs,
            device=x.device,
        )
        step = int(self.deltaT_meas / self.dt)

        obs_vec = torch.zeros(
            (N_batch, N_obs, self.meas.dimension),
            device=x.device,
            dtype=x.dtype,
        )
        for ii in range(N_obs):
            obs_vec[:, ii, :] = self.meas.measure(x[:, ii * step + step, :])

        return obs_vec, obs_t_vec


def initialize_sde(sde_config, device):
    sde_classes = {
        "ou": OrnsteinUhlenbeck,
        "bimodal": Bimodal,
        "lorenz63": Lorenz63,
        "lorenz96": Lorenz96,
        "lorenz96_clipped": Lorenz96Clipped,
    }
    sde_class = sde_classes.get(sde_config["name"])
    if sde_class is None:
        raise ValueError(f"Unknown SDE name: {sde_config['name']}")
    return sde_class(sde_config, device)


def initialize_observation_model(meas_config, device):
    meas_classes = {
        "normal": NormalObservation,
        "absolute_normal": AbsoluteNormalObservation,
        "von_mises": VonMisesObservation,
    }
    meas_class = meas_classes.get(meas_config["name"])
    if meas_class is None:
        raise ValueError(f"Unknown observation name: {meas_config['name']}")
    return meas_class(meas_config, device)
