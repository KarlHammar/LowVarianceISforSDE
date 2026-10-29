from training import TrainMethod, compute_logw_std, compute_ress


class LogVar(TrainMethod):
    def compute_controller_loss_from_terms(self, terms):
        logw_grouped = -self.grouped_total_cost(terms)
        logw_var = logw_grouped.var(dim=1, unbiased=True)
        loss = logw_var.mean()
        metrics = dict(
            loss=loss.item(),
            ress=compute_ress(logw_grouped, self.B_minibatch_per_obs),
            logw_std=compute_logw_std(logw_grouped),
        )
        return loss, metrics
