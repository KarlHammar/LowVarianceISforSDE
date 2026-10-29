import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm


class TrainMethod:
    def __init__(self, ctrl, params):
        self.ctrl = ctrl

        self.training_procedure = params["training_procedure"]
        self.recursive_terminal_strategy = params.get(
            "recursive_terminal_strategy",
            "value",
        )
        if self.recursive_terminal_strategy not in {"value", "rollout"}:
            raise ValueError(
                "recursive_terminal_strategy must be either 'value' or 'rollout', "
                f"got {self.recursive_terminal_strategy!r}"
            )
        self.skip_last_potential_training = params["skip_last_potential_training"]
        self.skip_potential_training = params["skip_potential_training"]
        self.optimizer_name = params["optimizer"]
        self.optimizer_params = params["optimizer_params"]
        self.schedule_lr_gamma = params["schedule_lr_gamma"]
        self.schedule_lr_plateau_patience = params["schedule_lr_plateau_patience"]
        self.N_epochs = params["N_epochs"]
        self.M_obs_per_batch = params["M_obs_per_batch"]
        self.B_minibatch_per_obs = params["B_minibatch_per_obs"]
        self.K_potential_target = params.get(
            "K_potential_target",
            self.B_minibatch_per_obs,
        )
        self.use_checkpointing = params["use_checkpointing"]
        self.detached = params.get("detached", False)

        self.network_num = None
        self.tracked_metrics = {}
        self.lr_change_epochs = {}
        self.loss_running_avg = {}

    def train(self):
        self.tracked_metrics = {}

        if self.training_procedure == "recursive":
            self.train_recursive()
        elif self.training_procedure == "full":
            self.train_full()
        else:
            raise ValueError(f"Unknown training procedure: {self.training_procedure}")

    def train_recursive(self):
        for ii in range(self.ctrl.ssm.num_meas):
            self.network_num = ii
            if self.trains_potential_with_controller():
                optimizers, schedulers, training_targets = self.init_opt_sched(self.network_num, "both_joint")
                self.train_step(self.compute_controller_losses, optimizers, schedulers, training_targets)
            else:
                trains_potential = (
                    not (
                        self.recursive_terminal_strategy == "rollout"
                        and self.skip_potential_training
                    )
                    and not (
                        self.network_num == self.ctrl.ssm.num_meas - 1
                        and self.skip_last_potential_training
                    )
                )
                if trains_potential:
                    optimizers, schedulers, training_targets = self.init_opt_sched(
                        self.network_num,
                        "both_separate",
                    )
                    self.train_step(
                        self.compute_simultaneous_losses,
                        optimizers,
                        schedulers,
                        training_targets,
                    )
                else:
                    optimizers, schedulers, training_targets = self.init_opt_sched(
                        self.network_num,
                        "controller",
                    )
                    self.train_step(self.compute_controller_losses, optimizers, schedulers, training_targets)

    def train_full(self):
        self.network_num = None
        optimizers, schedulers, training_targets = self.init_opt_sched(
            self.network_num,
            "controller" if not self.trains_potential_with_controller() else "both_joint"
        )
        self.train_step(self.compute_controller_losses, optimizers, schedulers, training_targets)

        if self.skip_potential_training:
            return

        for ii in range(self.ctrl.ssm.num_meas):
            if ii == self.ctrl.ssm.num_meas - 1 and self.skip_last_potential_training:
                continue

            self.network_num = ii
            optimizers, schedulers, training_targets = self.init_opt_sched(
                self.network_num,
                "potential",
            )
            self.train_step(self.compute_potential_losses, optimizers, schedulers, training_targets)

    def init_opt_sched(self, network_num, training_target):
        if self.optimizer_name != "adam":
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        named_parameter_groups = self.get_named_parameter_groups(network_num, training_target)
        optimizers = []
        schedulers = []
        training_targets = []
        for target, trainable_par in named_parameter_groups:
            optimizer = torch.optim.Adam(trainable_par, **self.optimizer_params)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=self.schedule_lr_gamma, patience=self.schedule_lr_plateau_patience)
            
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            training_targets.append(target)

        return optimizers, schedulers, training_targets

    def get_named_parameter_groups(self, network_num, training_target):
        match training_target:
            case "controller":
                return [("controller", self.get_controller_optimizer_parameters(network_num))]
            case "potential":
                return [("potential", self.get_potential_optimizer_parameters(network_num))]
            case "both_separate":
                return [
                    ("controller", self.get_controller_optimizer_parameters(network_num)),
                    ("potential", self.get_potential_optimizer_parameters(network_num)),
                ]
            case "both_joint":
                joint_parameters = list(self.get_controller_optimizer_parameters(network_num))
                joint_parameters.extend(self.get_potential_optimizer_parameters(network_num))
                return [("coupled", joint_parameters)]
            case _:
                raise ValueError(f"Unknown training target: {training_target}")

    def compute_controller_losses(self):
        X_start, y, yt = self.sample_training_batch()
        terms = self.generate_value_terms(X_start, y, yt, self.network_num)
        loss, metrics = self.compute_controller_loss_from_terms(terms)
        target = "coupled" if self.trains_potential_with_controller() else "controller"
        return {target: loss}, metrics

    def compute_potential_losses(self):
        X_start, y, yt = self.sample_training_batch()
        terms = self.generate_value_terms(X_start, y, yt, self.network_num)
        loss = self.compute_potential_loss_from_terms(terms)
        return {"potential": loss}, {"V_net_loss": loss.item()}

    def compute_simultaneous_losses(self):
        X_start, y, yt = self.sample_training_batch()
        terms = self.generate_value_terms(X_start, y, yt, self.network_num)
        controller_loss, metrics = self.compute_controller_loss_from_terms(terms)
        potential_loss = self.compute_potential_loss_from_terms(terms)
        metrics["V_net_loss"] = potential_loss.item()
        return {"controller": controller_loss, "potential": potential_loss}, metrics

    def compute_controller_loss_from_terms(self, terms):
        raise NotImplementedError

    def compute_potential_loss_from_terms(self, terms):
        B = self.B_minibatch_per_obs
        K = self.K_potential_target

        V0_target = (
            terms["running_cost"].detach()
            + terms["cost_compensation"].detach()
            + terms["terminal_cost"].detach()
        )
        V0_target = V0_target.reshape((-1, B))
        if K < B:
            # Subsample only the inner realization axis used to construct the
            # value-network target. Controller losses still use all B paths.
            subset_idx = torch.rand(V0_target.shape, device=V0_target.device).topk(
                K,
                dim=1,
                largest=False,
            ).indices
            V0_target = V0_target.gather(1, subset_idx)
        V0_target = -torch.logsumexp(-V0_target, dim=1) + np.log(K)

        X_start = terms["X_start"]
        y_remaining = terms["y_remaining"]
        X_pred = X_start.reshape(-1, B, X_start.shape[-1])[:, 0, :]
        y_pred = y_remaining.reshape(
            -1,
            B,
            y_remaining.shape[1],
            y_remaining.shape[2],
        )[:, 0, :, :]

        V0_pred = self.ctrl.nn_models[f"V0_model_{self.network_num}"](X_pred, y_pred)
        return torch.mean((V0_pred - V0_target) ** 2)

    def generate_value_terms(self, X_start, y, yt, network_num):
        ctrl = self.ctrl
        running_cost_total = torch.zeros(
            X_start.shape[0],
            device=X_start.device,
            dtype=X_start.dtype,
        )
        cost_compensation_total = torch.zeros(
            X_start.shape[0],
            device=X_start.device,
            dtype=X_start.dtype,
        )
        terminal_cost_total = torch.zeros(
            X_start.shape[0],
            device=X_start.device,
            dtype=X_start.dtype,
        )

        if network_num is None:
            X_prev = X_start
            y_remaining = y
            t_prev = 0.0
            for step_idx in range(yt.shape[0]):
                t_next = float(yt[step_idx].item())
                X_next, running_cost, cost_compensation = self.simulate_training_interval_paths(
                    X_prev,
                    t_prev,
                    t_next,
                    y,
                    yt,
                    use_checkpointing=self.use_checkpointing,
                    detached=self.detached,
                )
                running_cost_total = running_cost_total + running_cost
                cost_compensation_total = cost_compensation_total + cost_compensation
                terminal_cost_total = terminal_cost_total - ctrl.ssm.meas.log_g(
                    X_next,
                    y[:, step_idx, :],
                )
                X_prev = X_next
                t_prev = t_next
        else:
            obs_idx = ctrl.ssm.num_meas - network_num - 1
            y_remaining = y[:, obs_idx:, :]
            t_start = obs_idx * ctrl.ssm.deltaT_meas
            if self.recursive_terminal_strategy == "rollout":
                # The first interval uses Z_model_{network_num}.  At each later
                # observation boundary ctrl.control automatically dispatches to
                # the already-trained controller with one fewer observation.
                X_prev = X_start
                t_prev = t_start
                for step_idx in range(obs_idx, ctrl.ssm.num_meas):
                    t_next = float(yt[step_idx].item())
                    X_next, running_cost, cost_compensation = self.simulate_training_interval_paths(
                        X_prev,
                        t_prev,
                        t_next,
                        y,
                        yt,
                        use_checkpointing=self.use_checkpointing,
                        detached=self.detached,
                    )
                    running_cost_total = running_cost_total + running_cost
                    cost_compensation_total = cost_compensation_total + cost_compensation
                    terminal_cost_total = terminal_cost_total - ctrl.ssm.meas.log_g(
                        X_next,
                        y[:, step_idx, :],
                    )
                    X_prev = X_next
                    t_prev = t_next
            else:
                t_end = t_start + ctrl.ssm.deltaT_meas
                X_end, running_cost, cost_compensation = self.simulate_training_interval_paths(
                    X_start,
                    t_start,
                    t_end,
                    y,
                    yt,
                    use_checkpointing=self.use_checkpointing,
                    detached=self.detached,
                )
                terminal_cost = -ctrl.ssm.meas.log_g(X_end, y_remaining[:, 0, :])
                if network_num > 0:
                    next_V_nn = ctrl.nn_models[f"V0_model_{network_num-1}"]
                    terminal_cost = terminal_cost + next_V_nn(X_end, y_remaining[:, 1:, :])
                running_cost_total = running_cost
                cost_compensation_total = cost_compensation
                terminal_cost_total = terminal_cost

        return {
            "X_start": X_start,
            "y_remaining": y_remaining,
            "running_cost": running_cost_total,
            "cost_compensation": cost_compensation_total,
            "terminal_cost": terminal_cost_total,
        }

    def simulate_training_interval_paths(self, X0, t0, t1, y, yt, use_checkpointing=False, detached=False):
        return self.ctrl.simulate_training_interval_paths(
            X0,
            t0,
            t1,
            y,
            yt,
            use_checkpointing=use_checkpointing,
            detached=detached,
        )

    def grouped_total_cost(self, terms):
        return (
            terms["running_cost"]
            + terms["cost_compensation"]
            + terms["terminal_cost"]
        ).reshape(-1, self.B_minibatch_per_obs)

    def get_controller_optimizer_parameters(self, network_num):
        if network_num is None:
            params = []
            for name, model in self.ctrl.nn_models.items():
                if name.startswith("Z_model_"):
                    params.extend(model.parameters())
            return params
        return self.ctrl.nn_models[f"Z_model_{network_num}"].parameters()
    
    def get_potential_optimizer_parameters(self, network_num):
        if network_num is None:
            return self.ctrl.nn_models[f"V0_model_{self.ctrl.ssm.num_meas-1}"].parameters()
        return self.ctrl.nn_models[f"V0_model_{network_num}"].parameters()
        
    def trains_potential_with_controller(self):
        return False

    def save_curr_metrics(self):
        if not self.tracked_metrics:
            return

        for name, tracked_metric in self.tracked_metrics.items():
            plt.figure()
            mask = np.isfinite(tracked_metric)
            filtered_metric = tracked_metric[mask]
            if filtered_metric.size == 0:
                plt.close()
                continue
            plotted_values = [filtered_metric]
            plt.plot(filtered_metric, label=name)
            window_size = 10
            if filtered_metric.size >= window_size:
                smoothed_loss = np.convolve(filtered_metric, np.ones(window_size) / window_size, mode="valid")
                plotted_values.append(smoothed_loss)
                plt.plot(range(window_size - 1, len(smoothed_loss) + window_size - 1), smoothed_loss, label="Smoothed Loss", color="orange")
            for e in self.metric_lr_change_epochs(name):
                plt.axvline(e, color="red", linestyle="--", alpha=0.7)
            plt.axvline(self.N_epochs, color="black", linestyle="--", alpha=0.7, label="End of Training")
            if np.any(np.concatenate(plotted_values) < 0):
                plt.yscale("linear")
            else:
                plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Epoch")
            plt.ylabel(f"Metric: {name}")
            plt.grid()
            plt.savefig(f"{self.ctrl.path}/metric_{name}.pdf")
            plt.close()

        pd.DataFrame(self.tracked_metrics).to_csv(f"{self.ctrl.path}/metrics_all.csv", index=False)

        if len(self.tracked_metrics) > 1:
            plt.figure()
            plotted_values = []
            for name, tracked_metric in self.tracked_metrics.items():
                mask = np.isfinite(tracked_metric)
                filtered_metric = tracked_metric[mask]
                if filtered_metric.size == 0:
                    continue
                plotted_values.append(filtered_metric)
                plt.plot(filtered_metric, label=f"Metric: {name}")
            for e in self.all_lr_change_epochs():
                plt.axvline(e, color="red", linestyle="--", alpha=0.7)
            plt.axvline(self.N_epochs, color="black", linestyle="--", alpha=0.7, label="End of Training")
            if plotted_values:
                if np.any(np.concatenate(plotted_values) < 0):
                    plt.yscale("linear")
                else:
                    plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.legend()
            plt.grid()
            plt.savefig(f"{self.ctrl.path}/metrics_all.pdf")
            plt.close()

    def repeat_training_samples(self, X0, y):
        X0 = X0.repeat_interleave(self.B_minibatch_per_obs, 0)
        y = y.repeat_interleave(self.B_minibatch_per_obs, 0)
        return X0, y

    def sample_training_batch(self):
        x, y, yt = self.ctrl.ssm.sample_observed_trajectories(self.M_obs_per_batch)
        if self.network_num is None:
            X_start = x[:, 0, :]
        else:
            step = int(round(self.ctrl.ssm.deltaT_meas / self.ctrl.ssm.dt))
            X_start = x[:, (self.ctrl.ssm.num_meas - self.network_num - 1) * step, :]
        X_start, y = self.repeat_training_samples(X_start, y)
        return X_start, y, yt

    def metric_lr_change_epochs(self, metric_name):
        if "coupled" in self.lr_change_epochs:
            return self.lr_change_epochs["coupled"]
        if metric_name.endswith("V_net_loss"):
            return self.lr_change_epochs.get("potential", [])
        return self.lr_change_epochs.get("controller", [])

    def all_lr_change_epochs(self):
        return sorted({
            epoch
            for epochs in self.lr_change_epochs.values()
            for epoch in epochs
        })

    def train_step(self, loss_fcn, optimizers, schedulers, training_targets):
        self.lr_change_epochs = {
            target: []
            for target in training_targets
        }
        self.loss_running_avg = {name: None for name in training_targets}
        for epoch in tqdm(range(self.N_epochs)):
            losses, metrics = loss_fcn()
            total_loss = sum(losses.values())

            if not torch.isfinite(total_loss):
                tqdm.write(
                    f"Non-finite loss encountered for network {self.network_num} at epoch {epoch}. Stopping training early."
                )
                break

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            for target in training_targets:
                losses[target].backward()

            for optimizer in optimizers:
                optimizer.step()

            self.post_step(epoch, losses, metrics, optimizers, schedulers, training_targets)

        for target in training_targets:
            self.save_stage_checkpoint(target)

        # Save the final metrics after training is complete
        self.save_curr_metrics()

    def models_for_checkpoint(self, target):
        models = {}
        if target in ("controller", "coupled"):
            indices = range(self.ctrl.ssm.num_meas) if self.network_num is None else [self.network_num]
            for i in indices:
                models[f"Z_model_{i}"] = self.ctrl.nn_models[f"Z_model_{i}"]
        if target in ("potential", "coupled"):
            i = self.ctrl.ssm.num_meas - 1 if self.network_num is None else self.network_num
            models[f"V0_model_{i}"] = self.ctrl.nn_models[f"V0_model_{i}"]
        return models

    def save_stage_checkpoint(self, target):
        models = self.models_for_checkpoint(target)
        self.ctrl.save_model(nn_models=models)

    def post_step(self, epoch, losses, metrics, optimizers, schedulers, training_targets):
        metrics = add_metric_prefix(self.network_num, metrics)
        for optimizer, scheduler, target in zip(optimizers, schedulers, training_targets):
            loss_value = losses[target]
            old_lrs = [pg["lr"] for pg in optimizer.param_groups]
            scheduler.step(loss_value.detach())
            new_lrs = [pg["lr"] for pg in optimizer.param_groups]
            if any(n != o for n, o in zip(new_lrs, old_lrs)):
                self.lr_change_epochs[target].append(epoch)
                scheduler.best = float("inf")
                scheduler.num_bad_epochs = 0
            loss_scalar = loss_value.item()
            previous_avg = self.loss_running_avg[target]
            self.loss_running_avg[target] = (
                loss_scalar if previous_avg is None
                else 0.9 * previous_avg + 0.1 * loss_scalar
            )

        for metric_name, metric_value in metrics.items():
            if metric_name not in self.tracked_metrics:
                self.tracked_metrics[metric_name] = np.full(self.N_epochs, np.nan)
            self.tracked_metrics[metric_name][epoch] = metric_value

        loss_scalar = sum(loss.item() for loss in losses.values())

        tqdm.write(
            f"Network {self.network_num} \t Loss: {loss_scalar:.4f} \t Running Avg: {sum(self.loss_running_avg.values()):.4f}"
        )

        if (epoch + 1) % 100 == 0:
            self.save_curr_metrics()
            for target in training_targets:
                self.save_stage_checkpoint(target)


def add_metric_prefix(network_num, metrics):
    if network_num is None:
        return metrics
    return {
        f"{network_num}_{metric_name}": metric_value
        for metric_name, metric_value in metrics.items()
    }


def compute_ress(grouped_log_weights, B_minibatch_per_obs):
    with torch.no_grad():
        grouped_log_weights = grouped_log_weights.detach()
        normalized_log_weights = grouped_log_weights - torch.logsumexp(
            grouped_log_weights,
            dim=1,
            keepdim=True,
        )
        ess = torch.exp(-torch.logsumexp(2 * normalized_log_weights, dim=1))
        relative_ess = ess / B_minibatch_per_obs
        return relative_ess.mean().item()


def compute_logw_std(grouped_log_weights):
    with torch.no_grad():
        grouped_log_weights = grouped_log_weights.detach()
        logw_std = torch.std(
            grouped_log_weights,
            dim=1,
            unbiased=True,
        )
        return logw_std.mean().item()


def create_train_method(ctrl, params):
    from methods.CE import CE
    from methods.JointCE import JointCE
    from methods.LogVar import LogVar
    from methods.WLogVar import WLogVar
    from methods.RE import RE
    from methods.RECV import RECV
    from methods.Var import Var
    from methods.Moment import Moment

    method_classes = {
        "Moment": Moment,
        "RECV": RECV,
        "RE": RE,
        "Var": Var,
        "LogVar": LogVar,
        "WLogVar": WLogVar,
        "CE": CE,
        "JointCE": JointCE,
    }
    method_class = method_classes.get(params["name"])
    if method_class is None:
        raise ValueError(f"Unknown method name: {params['name']}")
    return method_class(ctrl, params)
