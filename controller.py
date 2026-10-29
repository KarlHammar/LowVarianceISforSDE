import numpy as np

from SSM.SSM import SSM
import torch
from torch.utils.checkpoint import checkpoint
from model import Z_Network_multi, V0_Network_multi
import os


class controller():
    """Controller class, can be different things
    Moment (single/multi)
    Analytical OU
    
    """

    def __init__(self, ssm_config: dict, device):
        self.ssm = SSM(ssm_config, device)
        self.device = device

    def simulate_controlled_paths(self, x0, t0, t1, Y, yt, W_vec, use_checkpointing=False, detached=False, return_full=False):

        X = x0
        RC = torch.zeros((x0.shape[0],), device=x0.device) # running cost
        CC = torch.zeros((x0.shape[0],), device=x0.device) # cost compensation

        N_steps = int(np.round((t1-t0) / self.ssm.dt))

        if return_full:
            x_vec = torch.zeros(
                (x0.shape[0], N_steps + 1, x0.shape[1]),
                device=x0.device,
                dtype=x0.dtype,
            )
            x_vec[:,0,:] = X

        t = t0
        for ii in range(N_steps):
            if use_checkpointing:
                ui = checkpoint(lambda X, t: self.control(t, X, yt, Y), X, t, use_reentrant=False)
            else:
                ui = self.control(t, X, yt, Y)

            vi = ui.detach() if detached else ui

            # NOTE: Do NOT use += here, it will mess up the checkpointing autograd graph
            RC = RC + (((vi - 0.5 * ui)*ui).sum(dim=1)) * self.ssm.dt
            CC = CC + (ui * W_vec[:,ii,:]).sum(dim=1)

            drift = self.ssm.sde.drift(t, X) # (B, d)
            sigma = self.ssm.sde.diffusion(t, X)
            control = torch.einsum('Bdm,Bm->Bd', sigma, vi) # (B, d)
            diffusion = torch.einsum('Bdm,Bm->Bd', sigma, W_vec[:,ii,:]) # (B, d)
            
            X = X + (drift + control) * self.ssm.dt  + diffusion
            
            if return_full:
                x_vec[:,ii+1,:] = X

            t += self.ssm.dt
        
        if return_full:
            return X, RC, CC, x_vec

        return X, RC, CC

    def _next_obs_indices(self, t, yt, n_max_future=1):
        """Return indices of up to n_future future observations after current time t."""
        if torch.is_tensor(t):
            t = float(t.detach().cpu())
        if torch.compiler.is_compiling():
            start_idx = int((t + 1e-10) / self.ssm.deltaT_meas)
            stop_idx = min(start_idx + n_max_future, yt.shape[0])
            return start_idx, stop_idx
        t_query = torch.as_tensor(t, device=yt.device, dtype=yt.dtype)
        start_idx = int(torch.searchsorted(yt, t_query, right=True).detach().cpu().item())
        stop_idx = min(start_idx + n_max_future, yt.shape[0])
        return start_idx, stop_idx

    def extract_future_measurements(self, t, yt_extracted, y, n_max_future=1):
        """Return up to n_future future observations and time-until-next for each."""
        start_idx, stop_idx = self._next_obs_indices(t, yt_extracted, n_max_future)
        if start_idx >= stop_idx:
            return None, None, 0
        
        y_extracted = y[:, start_idx:stop_idx, :]       # (batch, n_future, obs_dim)
        yt_future = yt_extracted[start_idx:stop_idx]    # (n_future,)
        yt_impact = yt_future - t                       # (n_future,)
        
        n_future_actual = stop_idx - start_idx

        return y_extracted, yt_impact, n_future_actual

    def supports_twisting_potential(self):
        return False

    def log_twisting_potential(self, t, x, y, yt):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a twisting potential interface."
        )

    def log_future_twisting_potential(self, t, x, y, yt):
        if y.shape[1] <= 1:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return self.log_twisting_potential(t, x, y[:, 1:, :], yt[1:])


class emptyController(controller):
    def __init__(self, ssm_config, device):
        super().__init__(ssm_config, device)
        self.name = "noCont"

    def control(self, t, x, zt, z):
        return torch.zeros_like(x)

    def supports_twisting_potential(self):
        return True

    def log_twisting_potential(self, t, x, y, yt):
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class optimalOUcontroller(controller):
    def __init__(self, ssm_config, device, lookahead_limit=10):
        super().__init__(ssm_config, device)
        if lookahead_limit > 10:
            print(f"Warning: lookahead_limit {lookahead_limit} is large, performance may suffer.")
        self.lookahead_limit = lookahead_limit
        self.name = "optOUcont"
        if self.lookahead_limit is not None:
            self.name += f"({lookahead_limit})"
    
    def supports_twisting_potential(self):
        return True
    
    def log_twisting_potential(self, t, x, y, yt):
        y, yt, num_meas_extracted = self.extract_future_measurements(t, yt, y, n_max_future=self.lookahead_limit)

        if num_meas_extracted == 0:
            return torch.zeros_like(x)

        dim_z = self.ssm.meas.dimension
        dim_x = self.ssm.sde.dimension_state

        N_batch = x.shape[0]
        N_future_z = yt.shape[0]

        A = self.ssm.sde.A # (dim_x, dim_x)
        mu = self.ssm.sde.mu # (dim_x)
        Sigma = self.ssm.sde.Sigma # (dim_x, dim_z)

        H = self.ssm.meas.H # (dim_z, dim_x)
        sigma_R = self.ssm.meas.R # (dim_z, dim_z)
        device = x.device
        dtype = x.dtype

        Q_vec = torch.empty((N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        exp_At_vec = torch.empty((N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        exp_mAt_vec = torch.empty((N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        exp_mATt_vec = torch.empty((N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        mu_z_vec = torch.zeros((N_batch, N_future_z, dim_z), device=device, dtype=dtype)
        for ii in range(N_future_z):
            exp_mAt, Q, exp_ATt = self.van_loan(A, Sigma, yt[ii])
            exp_mATt = exp_mAt.T
            exp_At = exp_ATt.T

            pred_state = mu + torch.einsum("ij,Bj->Bi", exp_mAt, (x - mu))       # (N_batch, dim_x)
            mu_z = torch.einsum("ij,Bj->Bi" , H, pred_state)                     # (N_batch, dim_z)

            Q_vec[ii,:,:] = Q
            exp_At_vec[ii,:,:] = exp_At
            exp_mAt_vec[ii,:,:] = exp_mAt
            exp_mATt_vec[ii,:,:] = exp_mATt
            mu_z_vec[:, ii, :] = mu_z

        # Compute this matrix in a "blocked" sense, to be flattened later
        Cov_ij = torch.zeros((N_future_z, N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        for ii in range(N_future_z):
            for jj in range(N_future_z):
                left = exp_mAt_vec[ii,:,:]
                right = exp_mATt_vec[jj,:,:]

                Cov_ij[ii, jj] = left @ Q_vec[min(ii,jj),:,:] @ right

        Sigma_zz = torch.einsum('pa,ijab,bq->ijpq', H, Cov_ij, H.T)

        for ii in range(N_future_z):
            Sigma_zz[ii, ii, :, :] += sigma_R
        
        # (1) form big Sigma_zz
        Sigma_zz_big = Sigma_zz.permute(0,2,1,3)\
                         .reshape(N_future_z*dim_z,
                                  N_future_z*dim_z)

        # Innovation:  (z - mu_z_vec) has shape (N_batch, N_future_z, dim_z), here we stack them...
        innovation = (y - mu_z_vec).reshape(N_batch, N_future_z * dim_z)

        # Up to an additive constant, the twisting potential is the Gaussian
        # log-likelihood of the future observations conditional on x_t.
        solved = torch.linalg.solve(Sigma_zz_big, innovation.T).T
        log_L = -0.5 * torch.einsum("Bi,Bi->B", innovation, solved)

        return log_L

    def control(self, t, x, yt, y):
        y, yt, num_meas_extracted = self.extract_future_measurements(t, yt, y, n_max_future=self.lookahead_limit)

        if num_meas_extracted == 0:
            return torch.zeros_like(x)

        dim_z = self.ssm.meas.dimension
        dim_x = self.ssm.sde.dimension_state

        N_batch = x.shape[0]
        N_future_z = yt.shape[0]

        A = self.ssm.sde.A # (dim_x, dim_x)
        mu = self.ssm.sde.mu # (dim_x)
        Sigma = self.ssm.sde.Sigma # (dim_x, dim_z)

        H = self.ssm.meas.H # (dim_z, dim_x)
        sigma_R = self.ssm.meas.R # (dim_z, dim_z)
        device = x.device
        dtype = x.dtype

        Q_vec = torch.empty((N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        exp_At_vec = torch.empty((N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        exp_mAt_vec = torch.empty((N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        exp_mATt_vec = torch.empty((N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        mu_z_vec = torch.zeros((N_batch, N_future_z, dim_z), device=device, dtype=dtype)
        for ii in range(N_future_z):
            exp_mAt, Q, exp_ATt = self.van_loan(A, Sigma, yt[ii])
            exp_mATt = exp_mAt.T
            exp_At = exp_ATt.T

            pred_state = mu + torch.einsum("ij,Bj->Bi", exp_mAt, (x - mu))       # (N_batch, dim_x)
            mu_z = torch.einsum("ij,Bj->Bi" , H, pred_state)                     # (N_batch, dim_z)

            Q_vec[ii,:,:] = Q
            exp_At_vec[ii,:,:] = exp_At
            exp_mAt_vec[ii,:,:] = exp_mAt
            exp_mATt_vec[ii,:,:] = exp_mATt
            mu_z_vec[:, ii, :] = mu_z

        # Compute this matrix in a "blocked" sense, to be flattened later
        Cov_ij = torch.zeros((N_future_z, N_future_z, dim_x, dim_x), device=device, dtype=dtype)
        for ii in range(N_future_z):
            for jj in range(N_future_z):
                left = exp_mAt_vec[ii,:,:]
                right = exp_mATt_vec[jj,:,:]

                Cov_ij[ii, jj] = left @ Q_vec[min(ii,jj),:,:] @ right

        Sigma_zz = torch.einsum('pa,ijab,bq->ijpq', H, Cov_ij, H.T)

        for ii in range(N_future_z):
            Sigma_zz[ii, ii, :, :] += sigma_R
        
        # (1) form big Sigma_zz
        Sigma_zz_big = Sigma_zz.permute(0,2,1,3)\
                         .reshape(N_future_z*dim_z,
                                  N_future_z*dim_z)

        # (2) invert
        SigmaZZ_inv = torch.linalg.inv(Sigma_zz_big)           # (N*dz, N*dz)

        # Innovation:  (z - mu_z_vec) has shape (N_batch, N_future_z, dim_z), here we stack them...
        innovation = (y - mu_z_vec).reshape(N_batch, N_future_z * dim_z)

        # Compute w = Σ_ZZ^{-1} (Z - μ_Z)
        w = torch.einsum("ij,Bj->Bi", SigmaZZ_inv, innovation)  # (B, N_future_z*dim_z)

        J_k = torch.einsum("za,kab->kzb", H, exp_mAt_vec)       # (N, dz, dx)

        J_T = J_k.permute(2, 0, 1).reshape(dim_x, N_future_z * dim_z)

        grad_x = w @ J_T.T                                      # (B, dx)

        u = grad_x @ Sigma                                      # (B, dim_w)

        return u
    
    def van_loan(self, A, Sigma, T):
        """
        https://personales.upv.es/asala/DocenciaOnline/material/DiscretizRuidoVanLoanEN.pdf

        Returns:
            F1       = exp(-A T)
            Q        = ∫_0^T exp(A s) (Sigma Sigmaᵀ) exp(Aᵀ s) ds
            F2       = exp(Aᵀ T)
        """

        d = A.shape[0]

        # Build Van Loan block matrix:
        #
        #   M = [ -A    G  ]
        #       [ 0  Aᵀ  ]
        #
        M = torch.zeros((2*d, 2*d), device=A.device, dtype=A.dtype)
        M[:d, :d] = -A
        M[:d, d:] =  Sigma @ Sigma.T
        M[d:, d:] =  A.T

        # Block matrix exponential
        expM = torch.linalg.matrix_exp(M * T)

        # Extract blocks
        F1 = expM[:d, :d]      # exp(-A T)
        G  = expM[:d, d:]      # 
        F2 = expM[d:, d:]      # exp(Aᵀ T)

        # Convert into integral we seek:
        integral = F2.T @ G

        return F1, integral, F2


class multiNnController(controller):
    def __init__(self, ssm_config, nn_config, device, name="abstract_controller", path=None, compile_rollouts=True):
        super().__init__(ssm_config, device)
        self.device = device
        self.nn_config = nn_config
        self.name = name
        self.path = path if path is not None else os.path.join("models", self.name)
        self.compile_rollouts = compile_rollouts

        self.nn_models = {}
        self.init_multi_models()
        self._compiled_rollouts = {}

    def load_model(self):
        for model_name, model in self.nn_models.items():
            path = f"{self.path}/state_dict_{model_name}.pt"
            try:
                model.load_state_dict(torch.load(path, map_location=self.device))
            except FileNotFoundError:
                if "V0" in model_name:
                    print(f"Warning: missing V0 weights at {path}. Using random initialization for {model_name}.")
                    continue
                raise

    def eval(self):
        for _, model in self.nn_models.items():
            model.eval()

    def save_model(self, nn_models=None):
        if nn_models is not None:
            models_to_save = nn_models
        else:
            models_to_save = self.nn_models

        for name, model in models_to_save.items():
            torch.save(model.state_dict(), f"{self.path}/state_dict_{name}.pt")

    def control(self, t, x, yt, y):
        Y_future, t_impact, num_meas_left = self.extract_future_measurements(
            t, yt, y, n_max_future=self.ssm.num_meas
        )
        if num_meas_left == 0:
            return torch.zeros_like(x)

        return -self.nn_models[f"Z_model_{num_meas_left - 1}"](t_impact[0], x, Y_future)

    def _make_compiled_controlled_paths(self, t0, t1, use_checkpointing, detached):
        def rollout(x0, y, yt, w_vec):
            return self.simulate_controlled_paths(
                x0,
                t0,
                t1,
                y,
                yt,
                W_vec=w_vec,
                use_checkpointing=use_checkpointing,
                detached=detached,
            )

        return rollout

    def _get_compiled_controlled_paths(self, t0, t1, use_checkpointing, detached):
        cache_key = (
            self.__class__,
            round(float(t0), 12),
            round(float(t1), 12),
            use_checkpointing,
            detached,
        )
        compiled_rollout = self._compiled_rollouts.get(cache_key)
        if compiled_rollout is None:
            compiled_rollout = self._make_compiled_controlled_paths(t0, t1, use_checkpointing, detached)
            if self.compile_rollouts:
                compiled_rollout = torch.compile(compiled_rollout)
            self._compiled_rollouts[cache_key] = compiled_rollout
        return compiled_rollout

    def simulate_training_interval_paths(self, X0, t0, t1, y, yt, use_checkpointing=False, detached=False):
        N_steps = int(round((t1 - t0) / self.ssm.dt))
        W_vec = torch.randn(
            X0.shape[0],
            N_steps,
            self.ssm.sde.dimension_noise,
            device=X0.device,
            dtype=X0.dtype,
        ) * np.sqrt(self.ssm.dt)

        rollout = self._get_compiled_controlled_paths(t0, t1, use_checkpointing, detached)
        return rollout(X0, y, yt, W_vec)

    def supports_twisting_potential(self):
        return any(name.startswith("V0_model_") for name in self.nn_models)

    def log_twisting_potential(self, t, x, y, yt):
        del t, yt
        num_meas_left = y.shape[1]
        model_name = f"V0_model_{num_meas_left - 1}"
        model = self.nn_models.get(model_name)
        if model is None:
            raise ValueError(
                f"Missing twisting potential '{model_name}' for lookahead {num_meas_left}."
            )
        return -model(x, y)

    def init_multi_models(self):
        if self.nn_config["Z_model"] == "Z_Network_multi":
            for network_num in range(self.ssm.num_meas):
                self.nn_models[f"Z_model_{network_num}"] = Z_Network_multi(
                    dimension_state=self.ssm.sde.dimension_state,
                    dimension_noise=self.ssm.sde.dimension_noise,
                    dimension_obs=self.ssm.meas.dimension,
                    number_obs=network_num + 1,
                    config=self.nn_config["Z_model_config"],
                ).to(self.device)
        else:
            raise ValueError(f"Unknown Z model: {self.nn_config['Z_model']}")

        if self.nn_config["V0_model"] == "V0_Network_multi":
            for network_num in range(self.ssm.num_meas):
                self.nn_models[f"V0_model_{network_num}"] = V0_Network_multi(
                    dimension_state=self.ssm.sde.dimension_state,
                    dimension_obs=self.ssm.meas.dimension,
                    number_obs=network_num + 1,
                    config=self.nn_config["V0_model_config"],
                ).to(self.device)
        else:
            raise ValueError(f"Unknown V0 model: {self.nn_config['V0_model']}")
        
