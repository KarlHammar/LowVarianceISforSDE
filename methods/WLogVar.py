import torch

from training import TrainMethod, compute_logw_std, compute_ress


class WLogVar(TrainMethod):
    def __init__(self, ctrl, params):
        super().__init__(ctrl, params)
        self.plain_weight = float(params["plain_weight"])
        if not 0.0 <= self.plain_weight <= 1.0:
            raise ValueError("plain_weight must lie in [0, 1]")

    def compute_controller_loss_from_terms(self, terms):
        logw_grouped = -self.grouped_total_cost(terms)

        correction = torch.exp(
            logw_grouped - torch.logsumexp(logw_grouped, dim=1, keepdim=True)
        ).detach()
        weighted_mean_logw = (correction * logw_grouped).sum(dim=1, keepdim=True)
        weighted_squared_deviation = (logw_grouped - weighted_mean_logw) ** 2
        weighted_var_log_weight = (correction * weighted_squared_deviation).sum(dim=1)

        logw_var = logw_grouped.var(dim=1, unbiased=True)
        var_log_weight = (
            self.plain_weight * logw_var
            + (1.0 - self.plain_weight) * weighted_var_log_weight
        )
        loss = var_log_weight.mean()

        metrics = dict(
            loss=loss.item(),
            plain_var_loss=logw_var.mean().item(),
            weighted_var_loss=weighted_var_log_weight.mean().item(),
            ress=compute_ress(logw_grouped, self.B_minibatch_per_obs),
            logw_std=compute_logw_std(logw_grouped),
        )
        return loss, metrics
