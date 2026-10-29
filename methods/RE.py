import torch

from training import TrainMethod, compute_logw_std, compute_ress


class RE(TrainMethod):
    def compute_controller_loss_from_terms(self, terms):
        running_cost = terms["running_cost"]
        terminal_cost = terms["terminal_cost"]
        logw_grouped = -self.grouped_total_cost(terms)
        loss = torch.mean(running_cost + terminal_cost)
        metrics = dict(
            loss=loss.item(),
            ress=compute_ress(logw_grouped, self.B_minibatch_per_obs),
            logw_std=compute_logw_std(logw_grouped),
        )
        return loss, metrics
