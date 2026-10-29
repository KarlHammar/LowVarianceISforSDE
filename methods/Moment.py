import torch

from training import TrainMethod, compute_logw_std, compute_ress


class Moment(TrainMethod):
    def trains_potential_with_controller(self):
        return True

    def compute_controller_loss_from_terms(self, terms):
        ctrl = self.ctrl
        V = terms["running_cost"] + terms["cost_compensation"] + terms["terminal_cost"]
        V_grouped = V.reshape(-1, self.B_minibatch_per_obs)
        logw_grouped = -V_grouped

        V0_model_num = ctrl.ssm.num_meas - 1 if self.network_num is None else self.network_num
        Vnet = ctrl.nn_models[f"V0_model_{V0_model_num}"](
            terms["X_start"],
            terms["y_remaining"],
        )

        loss = torch.mean((Vnet - V) ** 2)
        metrics = dict(
            loss=loss.item(),
            ress=compute_ress(logw_grouped, self.B_minibatch_per_obs),
            logw_std=compute_logw_std(logw_grouped),
        )
        return loss, metrics
