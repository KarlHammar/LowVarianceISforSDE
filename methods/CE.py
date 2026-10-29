import torch

from training import TrainMethod, compute_logw_std, compute_ress


class CE(TrainMethod):
    def compute_controller_loss_from_terms(self, terms):
        logw_grouped = -self.grouped_total_cost(terms)
        logw_norm_grouped = logw_grouped - torch.logsumexp(
            logw_grouped,
            dim=1,
            keepdim=True,
        )
        correction = torch.exp(logw_norm_grouped).detach()
        loss = torch.sum(correction * logw_grouped) / self.M_obs_per_batch
        metrics = dict(
            loss=loss.item(),
            ress=compute_ress(logw_grouped, self.B_minibatch_per_obs),
            logw_std=compute_logw_std(logw_grouped),
        )
        return loss, metrics
