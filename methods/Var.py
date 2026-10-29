import math

import torch

from training import TrainMethod, compute_logw_std, compute_ress


class Var(TrainMethod):
    def compute_controller_loss_from_terms(self, terms):
        logw_grouped = -self.grouped_total_cost(terms)

        log_B = math.log(self.B_minibatch_per_obs)
        log_mean_w = torch.logsumexp(logw_grouped, dim=1) - log_B
        log_mean_w2 = torch.logsumexp(2 * logw_grouped, dim=1) - log_B

        mean_w = torch.exp(log_mean_w)
        mean_w2 = torch.exp(log_mean_w2)
        loss = (mean_w2 - mean_w**2).mean()

        metrics = dict(
            loss=loss.item(),
            ress=compute_ress(logw_grouped, self.B_minibatch_per_obs),
            logw_std=compute_logw_std(logw_grouped),
        )
        return loss, metrics
