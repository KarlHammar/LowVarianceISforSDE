import numpy as np

import torch
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from training import TrainMethod, compute_logw_std, compute_ress


class RECV(TrainMethod):
    def compute_controller_loss_from_terms(self, terms):
        logw_grouped = -self.grouped_total_cost(terms)
        loss = torch.mean(-logw_grouped)
        metrics = dict(
            loss=loss.item(),
            ress=compute_ress(logw_grouped, self.B_minibatch_per_obs),
            logw_std=compute_logw_std(logw_grouped),
        )
        return loss, metrics

    def simulate_training_interval_paths(self, X0, t0, t1, y, yt):
        ctrl = self.ctrl
        N_steps = int(round((t1 - t0) / ctrl.ssm.dt))
        W_vec = torch.randn(
            X0.shape[0],
            N_steps,
            ctrl.ssm.sde.dimension_noise,
            device=X0.device,
            dtype=X0.dtype,
        ) * np.sqrt(ctrl.ssm.dt)
        return self.simulate_controlled_paths(X0, t0, t1, y, yt, W_vec)

    def simulate_controlled_paths(self, x0, t0, t1, Y, yt, W_vec):
        ctrl = self.ctrl
        X = x0
        RC = torch.zeros((x0.shape[0],), device=x0.device)
        CC = torch.zeros((x0.shape[0],), device=x0.device)

        N_steps = int(np.round((t1 - t0) / ctrl.ssm.dt))
        t = t0
        for ii in range(N_steps):
            if self.use_checkpointing:
                Zi = checkpoint(lambda X, t: -ctrl.control(t, X, yt, Y), X, t, use_reentrant=False)
                Zi_f = checkpoint(lambda X, t: -self.control_f(t, X, yt, Y), X, t, use_reentrant=False)
            else:
                Zi = -ctrl.control(t, X, yt, Y)
                Zi_f = -self.control_f(t, X, yt, Y)

            ci = -Zi.detach() if self.detached else -Zi

            RC = RC - ((0.5 * Zi**2 + ci * Zi).sum(dim=1)) * ctrl.ssm.dt
            CC = CC - (Zi_f * W_vec[:, ii, :]).sum(dim=1)
            base_drift = ctrl.ssm.sde.drift(t, X)
            sigma = ctrl.ssm.sde.diffusion(t, X)
            control = torch.einsum("Bdm,Bm->Bd", sigma, ci)
            diffusion = torch.einsum("Bdm,Bm->Bd", sigma, W_vec[:, ii, :])
            X = X + (base_drift + control) * ctrl.ssm.dt + diffusion
            t += ctrl.ssm.dt

        return X, RC, CC

    def control_f(self, t, x, yt, y):
        ctrl = self.ctrl
        Y_future, t_impact, num_meas_left = ctrl.extract_future_measurements(
            t,
            yt,
            y,
            n_max_future=ctrl.ssm.num_meas,
        )
        if num_meas_left == 0:
            return torch.zeros_like(x)

        model = ctrl.nn_models[f"Z_model_{num_meas_left - 1}"]
        params = {
            name: value.detach()
            for name, value in model.named_parameters()
        }
        buffers = {
            name: value
            for name, value in model.named_buffers()
        }
        return -functional_call(model, (params, buffers), (t_impact[0], x, Y_future))
