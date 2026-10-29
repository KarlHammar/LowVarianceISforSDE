import torch

from training import TrainMethod


class JointCE(TrainMethod):
    """Reverse-KL cross entropy on the joint latent/observation law."""

    def __init__(self, ctrl, params):
        super().__init__(ctrl, params)
        if self.B_minibatch_per_obs != 1:
            raise ValueError(
                "JointCE samples one true latent realization per observation path; "
                "set B_minibatch_per_obs to 1."
            )

    def train_recursive(self):
        for network_num in range(self.ctrl.ssm.num_meas):
            self.network_num = network_num
            optimizers, schedulers, training_targets = self.init_opt_sched(
                network_num,
                "controller",
            )
            self.train_step(
                self.compute_controller_losses,
                optimizers,
                schedulers,
                training_targets,
            )

    def compute_controller_losses(self):
        x, y, yt, W = self.ctrl.ssm.sample_observed_trajectories(
            self.M_obs_per_batch,
            return_noise=True,
        )
        loss, metrics = self.joint_reverse_kl(x, y, yt, W)
        return {"controller": loss}, metrics

    def compute_controller_loss_from_terms(self, terms):
        raise RuntimeError(
            "JointCE evaluates true joint trajectories, not importance-weight terms."
        )

    def joint_reverse_kl(self, x, y, yt, W):
        ssm = self.ctrl.ssm
        steps_per_observation = int(round(ssm.deltaT_meas / ssm.dt))

        if self.network_num is None:
            start_step = 0
            stop_step = W.shape[1]
        else:
            observation_index = ssm.num_meas - self.network_num - 1
            start_step = observation_index * steps_per_observation
            stop_step = start_step + steps_per_observation

        energy = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        stochastic_integral = torch.zeros_like(energy)
        boundary_epsilon = ssm.dt * 1e-4
        for step in range(start_step, stop_step):
            # At an observation boundary, use the controller after that
            # observation has been consumed.
            t = step * ssm.dt + boundary_epsilon
            u = self.ctrl.control(t, x[:, step, :], yt, y)
            energy = energy + 0.5 * torch.sum(u.square(), dim=1) * ssm.dt
            stochastic_integral = stochastic_integral + torch.sum(u * W[:, step, :], dim=1)

        pathwise_loss = energy - stochastic_integral
        loss = pathwise_loss.mean()
        metrics = {
            "loss": loss.item(),
            "control_energy": energy.mean().item(),
            "stochastic_integral": stochastic_integral.mean().item(),
        }
        return loss, metrics
