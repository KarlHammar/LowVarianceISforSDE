import torch
import csv
from logz_statistics import bootstrap_std
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from SSM.SDEs.ou import OrnsteinUhlenbeck
from SSM.meas.normal import NormalObservation
from SSM.SSM import SSM
from controller import controller, multiNnController, optimalOUcontroller, emptyController
from tqdm import tqdm
import os
import datetime
import shutil
from visualize import *
from train import load_model_params


class Tester:

    def __init__(self, par: dict, names, smt_config, twist_config, test_name=None, test_type='smoothing'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.par = par
        self.names = names
        self.smt_config = smt_config
        self.twist_config = twist_config
        self.test_name = test_name
        self.test_type = test_type
        self.ssm = SSM(par["ssm_config"], self.device)

        # setup testing parameters, this is essentially filling in smoothing_config with things from name0.par
        if self.smt_config['deltaT_meas'] is None:
            self.smt_config['deltaT_meas'] = self.ssm.deltaT_meas
        if self.smt_config['num_meas'] is None:
            self.smt_config['num_meas'] = self.ssm.num_meas
        if self.twist_config['lookahead'] is None:
            self.twist_config['lookahead'] = self.ssm.num_meas

        # Set up results directory for this run
        if self.test_name is not None:
            self.results_root = os.path.join('results', self.test_type, self.test_name)
        elif len(names) > 1:
            # Save to results/ for comparisons
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"comparison_{'_'.join(names)}_{timestamp}"
            self.results_root = os.path.join('results', run_name)
        else:
            # Save to models/ as before
            self.results_root = f'models/{names[0]}'
        self._prepare_results_dir()
        self.visualizer = SmoothingVisualizer(self.results_root, self.ssm)
        
        # Initialize all controllers to test
        self.ctrls_to_test = {
            'no_control': emptyController(
                self.par["ssm_config"],
                self.device,
            )
        }
        for name in names:
            ctrl = self.load_trained_controller(name)
            self.ctrls_to_test[name] = ctrl
        if isinstance(self.ssm.sde, OrnsteinUhlenbeck) and isinstance(
            self.ssm.meas, NormalObservation
        ):
            include_opt = self.test_type != 'twisting' or self.twist_config.get('include_opt', False)
            if include_opt:
                lookaheads = {1}
                if self.test_type == 'smoothing':
                    lookaheads.add(int(self.ssm.num_meas))
                elif self.test_type == 'twisting':
                    for lookahead in self._as_list(self.twist_config['lookahead']):
                        lookaheads.add(int(lookahead))
                for lookahead in sorted(lookaheads):
                    if lookahead < 1:
                        continue
                    opt_ctrl = optimalOUcontroller(
                        self.par["ssm_config"],
                        self.device,
                        lookahead_limit=lookahead,
                    )
                    self.ctrls_to_test[f'optimal({lookahead})'] = opt_ctrl

    def _as_list(self, value):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _prepare_results_dir(self):
        if self.test_name is not None and os.path.isdir(self.results_root):
            shutil.rmtree(self.results_root)
        os.makedirs(self.results_root, exist_ok=True)
        slurm_job_id = os.getenv("SLURM_JOB_ID", "").strip()
        with open(os.path.join(self.results_root, 'test_params.json'), 'w') as f:
            json.dump({
                'name': self.test_name,
                'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'slurm_job_id': int(slurm_job_id) if slurm_job_id else -1,
                'test_type': self.test_type,
                'model_names': self.names,
                'smoothing_config': self.smt_config,
                'twisting_config': self.twist_config,
                'train_params': self.par,
            }, f, indent=4)

    def load_trained_controller(self, name):
        par = load_model_params(name)
        ctrl = multiNnController(
            par["ssm_config"],
            par["nn_config"],
            self.device,
            name=par["name"],
            path=par["path"],
        )
        ctrl.load_model()
        ctrl.eval()
        return ctrl

    def generate_measurements_smoothing(self, M):
        if self.smt_config['use_fixed_measurement']:
            y = torch.tensor(
                self.smt_config['fixed_measurement_vector'],
                device=self.device,
                dtype=torch.float32,
            )  # (meas_dim)
            Y = y.unsqueeze(0).repeat(self.smt_config['num_meas'], 1)  # (num_meas, meas_dim)
        else:
            X0 = self.ssm.get_initial_mean(1, device=self.device)

            t_final = self.smt_config['deltaT_meas'] * self.smt_config['num_meas']
            x_paths = self.ssm.simulate_paths(X0, t_final)
            step = int(round(self.smt_config['deltaT_meas'] / self.ssm.dt))
            # For each path, simulate B_minibatch_per_obs mesurements at the end
            Y = torch.zeros((self.smt_config['num_meas'], self.ssm.meas.dimension), device=self.device)
            for ii in range(self.smt_config['num_meas']):
                state = x_paths[0, (ii + 1) * step, :].unsqueeze(0)
                Y[ii, :] = self.ssm.meas.measure(state)  # shape (1, dim_meas)
        Y = Y.unsqueeze(0).repeat(M, 1, 1)  # (M, num_meas, meas_dim)

        yt = torch.tensor(
            [self.smt_config['deltaT_meas'] * (ii + 1) for ii in range(self.smt_config['num_meas'])],
            device=self.device,
            dtype=torch.float32,
        )  # (num_meas)
        return Y, yt

    def simulate_controlled(self, M, ctrl: controller, tvec: torch.Tensor, W_vec: torch.Tensor, Y: torch.Tensor, yt: torch.Tensor):
        x0 = self.ssm.get_initial_mean(M, device=W_vec.device)

        t0 = 0
        t1 = self.smt_config['deltaT_meas'] * self.smt_config['num_meas']
        _, running_cost, cost_compensation, x = ctrl.simulate_controlled_paths(x0, t0, t1, Y, yt, W_vec, return_full=True)

        girsanov_term = - running_cost - cost_compensation
        yt_idxs = []
        for t in yt:
            eq = torch.isclose(tvec, t, atol=1e-6)
            if not torch.any(eq):
                raise ValueError(f"yt={t.item()} not found in xt.")
            idx = torch.where(eq)[0][0]
            yt_idxs.append(idx)
        yt_idxs = torch.tensor(yt_idxs, device=x.device, dtype=torch.long)
        likelihood_term = torch.zeros(x.shape[0], device=x.device)
        for ii, yt_idx in enumerate(yt_idxs):
            likelihood_term += ctrl.ssm.meas.log_g(x[:, yt_idx, :], Y[:, ii, :])
        log_unnorm_weight = girsanov_term + likelihood_term        
        log_unnorm_weight = log_unnorm_weight.cpu().detach().numpy()
        Xmat = x.cpu().detach().numpy()

        return Xmat, log_unnorm_weight

    def generate_new_Y_samples(self):
        self.tvec = torch.from_numpy(np.arange(0, self.smt_config['deltaT_meas'] * self.smt_config['num_meas'] + self.ssm.dt, self.ssm.dt)).float().to(self.device)
        self.Y, self.yt = self.generate_measurements_smoothing(self.smt_config['num_paths'])
        self.W_vec = torch.randn(
            self.smt_config['num_paths'], self.tvec.shape[0],
            self.ssm.sde.dimension_noise, device=self.device,
        ) * np.sqrt(self.ssm.dt)

    def est_correct_logz(self, observation_id=0):
        analytical_log_z = None
        try:
            if self.ssm.init_config["name"] != "fixed":
                raise TypeError("Kalman truth currently requires a fixed initial state.")
            x0 = self.ssm.get_initial_mean(1, device=self.device, dtype=torch.float64)
            analytical_log_z = self.compute_twisting_kalman_logz(
                self.Y[:1], self.yt, x0
            )
            print(f"Analytical smoothing log Z: {analytical_log_z:.10f}")

            reference = dict(controller='kalman', num_bootstrap_samples=0,
                             log_z_hat=analytical_log_z, log_z_hat_se=0.0,
                             log_z_hat_ci95_low=analytical_log_z,
                             log_z_hat_ci95_high=analytical_log_z)
            self.write_observation_reference(observation_id, 'Analytical', reference)
            return reference
        except (TypeError, ValueError) as exc:
            print(f"Skipping analytical smoothing log Z: {exc}")

        correct_num_batches = 100
        correct_batch_size = 100000
        correct_num_paths = correct_num_batches * correct_batch_size
        ctrl = emptyController(
            self.par["ssm_config"],
            self.device,
        )
        Y_correct = self.Y[:1].repeat(correct_batch_size, 1, 1)

        batch_log_weight_sums = []
        batch_log_weight_sq_sums = []
        for _ in tqdm(range(correct_num_batches), desc='Estimating correct log Z'):
            W_vec = torch.randn(
                correct_batch_size,
                self.tvec.shape[0],
                self.ssm.sde.dimension_noise,
                device=self.device,
            ) * np.sqrt(self.ssm.dt)
            with torch.inference_mode():
                Xmat, log_weights = self.simulate_controlled(
                    correct_batch_size,
                    ctrl,
                    self.tvec,
                    W_vec,
                    Y_correct,
                    self.yt,
                )

            log_weights = np.asarray(log_weights, dtype=np.float64)
            batch_log_weight_sums.append(np.logaddexp.reduce(log_weights))
            batch_log_weight_sq_sums.append(np.logaddexp.reduce(2.0 * log_weights))
            del Xmat, log_weights, W_vec

        log_num_paths = np.log(correct_num_paths)
        log_z = float(np.logaddexp.reduce(batch_log_weight_sums) - log_num_paths)
        log_mean_w_sq = np.logaddexp.reduce(batch_log_weight_sq_sums) - log_num_paths
        rel_var = max(float(np.exp(log_mean_w_sq - 2.0 * log_z) - 1.0), 0.0)
        log_z_se = float(np.sqrt(rel_var / correct_num_paths))
        log_z_ci95_low = log_z - 1.96 * log_z_se
        log_z_ci95_high = log_z + 1.96 * log_z_se
        reference = {
            'controller': 'Correct',
            'num_bootstrap_samples': correct_num_paths,
            'log_z_hat': log_z,
            'log_z_hat_se': log_z_se,
            'log_z_hat_ci95_low': log_z_ci95_low,
            'log_z_hat_ci95_high': log_z_ci95_high,
        }
        self.write_observation_reference(observation_id, 'Estimate', reference)
        return reference

    def write_observation_reference(self, observation_id, label, reference):
        path = os.path.join(self.results_root, 'smoothing_observation_references.csv')
        with open(path, 'a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['observation_id', 'label', *reference])
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(dict(observation_id=observation_id, label=label, **reference))

    def write_twisting_logz_estimates(self, results):
        estimates_path = os.path.join(self.results_root, 'twisting_logz_estimates.csv')
        with open(estimates_path, 'w') as f:
            f.write("controller,sample_idx,log_z_hat\n")
            for name, traces in results.items():
                for sample_idx, trace in enumerate(traces):
                    f.write(f"{name},{sample_idx},{float(trace['log_norm_const']):.10f}\n")

    def test_smoothing(self):
        results = {ctrl_name: [] for ctrl_name in self.ctrls_to_test.keys()}

        paths = {name: os.path.join(self.results_root, name) for name in (
            'smoothing_observation_estimates.csv',
            'smoothing_observation_references.csv')}
        # Clear files if they exist, so we can append to them
        for path in paths.values():
            with open(path, 'w'):
                pass
        with open(paths['smoothing_observation_estimates.csv'], 'w', newline='') as estimates_file:
            estimates_writer = csv.writer(estimates_file)
            estimates_writer.writerow(['observation_id', 'controller', 'log_z_hat', 'status',
                                      'log_z_hat_bootstrap_sd'])
            for test_idx in range(self.smt_config['num_tests']):
                self.generate_new_Y_samples()
                self.est_correct_logz(test_idx)
                for ctrl_name, ctrl in self.ctrls_to_test.items():
                    ess, log_norm_const, log_weights = self.run_smoothing(ctrl, l_plot=False)
                    valid = len(log_weights) >= 2 and np.isfinite(log_weights).all()
                    status = 'ok' if valid else 'failed'
                    sd = bootstrap_std(log_weights, 2000) if valid else np.nan
                    estimates_writer.writerow([test_idx, ctrl_name, log_norm_const, status, sd])
                    run = dict(ess=ess, log_z_hat=log_norm_const,
                               logw_std=float(np.std(log_weights, ddof=1)))
                    results[ctrl_name].append(run)
                estimates_file.flush()

        csv_header = "controller,ess,ess_std,log_z_hat_first,log_z_hat_std_first,log_z_hat_std_mean"
        pretty_header = "controller\tess\tess_std\tlog_z_hat_first\tlog_z_hat_std_first\tlog_z_hat_std_mean"

        print(pretty_header)
        with open(os.path.join(self.results_root, 'smoothing_results.csv'), 'w') as f:
            f.write(csv_header + "\n")
            for name in self.ctrls_to_test.keys():
                ess_vals = [run['ess'] for run in results[name]]
                log_z_hat_vals = [run['log_z_hat'] for run in results[name]]
                log_z_hat_std_vals = [run['logw_std'] for run in results[name]]
                if not ess_vals or not log_z_hat_vals or not log_z_hat_std_vals:
                    continue
                ess_mean = float(np.mean(ess_vals))
                ess_std = float(np.std(ess_vals, ddof=1))
                log_z_hat_first = float(log_z_hat_vals[0])
                log_z_hat_std_first = float(log_z_hat_std_vals[0])
                log_z_hat_std_mean = float(np.mean(log_z_hat_std_vals))
                row = (
                    f"{name},{ess_mean:.4f},{ess_std:.4f},"
                    f"{log_z_hat_first:.6f},{log_z_hat_std_first:.6f},{log_z_hat_std_mean:.6f}"
                )
                pretty_row = (
                    f"{name}\t{ess_mean:.4f}\t{ess_std:.4f}\t"
                    f"{log_z_hat_first:.6f}\t{log_z_hat_std_first:.6f}\t{log_z_hat_std_mean:.6f}"
                )

                print(pretty_row)
                f.write(row + "\n")
        

    def run_smoothing(self, ctrl: multiNnController, l_plot=False):
        M = self.smt_config['num_paths']

        with torch.inference_mode():
            Xmat, log_unnorm_weight = self.simulate_controlled(M, ctrl, self.tvec, self.W_vec, self.Y, self.yt)

        # Save metrics
        log_total = np.logaddexp.reduce(log_unnorm_weight)
        log_norm_weight = log_unnorm_weight - log_total
        norm_weight = np.exp(log_norm_weight)

        ess = 1 / (np.sum(norm_weight**2) * M)
        # normalization const estimate
        log_norm_const = log_total - np.log(len(log_unnorm_weight))

        # Save plots
        if l_plot:
            self.visualizer.plot_paths(ctrl.name, Xmat, self.yt, self.Y, ctrl)
            if isinstance(ctrl, emptyController):
                self.visualizer.plot_facit_paths(Xmat, norm_weight, self.tvec, self.yt, self.Y)
            self.visualizer.plot_heatmap(
                ctrl,
                ctrl.name,
                self.tvec,
                self.yt,
                self.Y,
                x_limits=self._heatmap_state_limits(ctrl),
            )
        
        return ess, log_norm_const, log_unnorm_weight

    def _heatmap_state_limits(self, ctrl, num_std=3.0):
        standardization = self.par["nn_config"]["Z_model_config"]["standardization"]
        x_mean = float(np.ravel(standardization["x_mean"])[0])
        x_std = float(np.ravel(standardization["x_std"])[0])
        x_std = max(abs(x_std), 1e-12)
        return x_mean - num_std * x_std, x_mean + num_std * x_std

    def generate_measurement_sequence(self, N, num_paths):
        #simulate sde to deltaT_meas*N and pick out measurements
        X0 = self.ssm.get_initial_mean(1, device=self.device)

        # Simulate on the fixed dt grid and sample the measurement states from it.
        t_final = self.ssm.deltaT_meas * N
        x_paths = self.ssm.simulate_paths(X0, t_final)
        step = int(round(self.ssm.deltaT_meas / self.ssm.dt))
        sample_times = torch.tensor(
            [self.ssm.deltaT_meas * ii for ii in range(0, N + 1)],
            device=self.device,
            dtype=torch.float32,
        )
        X = x_paths[0, ::step, :].unsqueeze(1)

        y = self.ssm.meas.measure(X[1:,0,:]) # shape (M_obs_per_batch, dim_meas)
        yt = sample_times[1:]  # shape (M_obs_per_batch)

        # batch up measurements
        y = y.unsqueeze(0).repeat_interleave(num_paths, 0)  # shape (num_paths, num_meas, dim_meas)
        return y, yt

    def _ou_transition_moments(self, delta_t):
        A = self.ssm.sde.A
        Sigma = self.ssm.sde.Sigma
        d = A.shape[0]
        M = torch.zeros((2 * d, 2 * d), device=A.device, dtype=torch.float64)
        A = A.to(dtype=torch.float64)
        Sigma = Sigma.to(dtype=torch.float64)
        M[:d, :d] = -A
        M[:d, d:] = Sigma @ Sigma.T
        M[d:, d:] = A.T

        expM = torch.linalg.matrix_exp(M * float(delta_t))
        F = expM[:d, :d]
        G = expM[:d, d:]
        exp_AT = expM[d:, d:]
        integral = exp_AT.T @ G

        Q = F @ integral @ F.T
        Q = 0.5 * (Q + Q.T)

        return F, Q

    def compute_twisting_kalman_logz(self, y, yt, x0):
        if not isinstance(self.ssm.sde, OrnsteinUhlenbeck):
            raise TypeError("Kalman truth is only implemented for OU SDEs.")
        if not isinstance(self.ssm.meas, NormalObservation):
            raise TypeError("Kalman truth is only implemented for normal observations.")

        y_single = y[0].to(device=self.device, dtype=torch.float64)
        yt = yt.to(device=self.device, dtype=torch.float64)
        H = self.ssm.meas.H.to(dtype=torch.float64)
        R = self.ssm.meas.R.to(dtype=torch.float64)
        mu = self.ssm.sde.mu.to(dtype=torch.float64)
        dim_y = H.shape[0]

        mean = x0[0].to(device=self.device, dtype=torch.float64)
        dim = mean.shape[0]
        cov = torch.zeros((dim, dim), device=self.device, dtype=torch.float64)
        log_z = torch.zeros((), device=self.device, dtype=torch.float64)
        t_prev = 0.0

        for obs_idx in range(yt.shape[0]):
            delta_t = float(yt[obs_idx].detach().cpu().item()) - t_prev
            F, Q = self._ou_transition_moments(delta_t)

            # Predict:
            #   x_t | y_{1:t-1} ~ N(pred_mean, pred_cov)
            pred_mean = mu + F @ (mean - mu)
            pred_cov = F @ cov @ F.T + Q

            # Observation prediction:
            #   y_t | y_{1:t-1} ~ N(H pred_mean, innov_cov)
            innovation = y_single[obs_idx] - H @ pred_mean
            innov_cov = H @ pred_cov @ H.T + R
            innov_cov_inv = torch.linalg.inv(innov_cov)

            # Accumulate log p(y_t | y_{1:t-1}).
            mahalanobis = innovation @ innov_cov_inv @ innovation
            sign, log_det = torch.linalg.slogdet(innov_cov)
            if sign <= 0:
                raise ValueError("Kalman innovation covariance is not positive definite.")
            log_z = log_z - 0.5 * (
                mahalanobis
                + log_det
                + dim_y * np.log(2.0 * np.pi)
            )

            # Update:
            #   x_t | y_{1:t} ~ N(mean, cov)
            gain = pred_cov @ H.T @ innov_cov_inv
            mean = pred_mean + gain @ innovation
            cov = pred_cov - gain @ H @ pred_cov
            cov = 0.5 * (cov + cov.T)
            t_prev = float(yt[obs_idx].detach().cpu().item())

        return float(log_z.detach().cpu().item())

    def write_twisting_kalman_logz(self, y, yt, x0):
        log_z = self.compute_twisting_kalman_logz(y, yt, x0)
        path = os.path.join(self.results_root, 'twisting_correct_logz.csv')
        with open(path, 'w') as f:
            f.write("label,method,num_meas,log_z\n")
            f.write(f"Correct,kalman,{yt.shape[0]},{log_z:.10f}\n")
        print(f"Kalman log Z: {log_z:.10f}")
        return log_z

    def _extract_observation_block(self, step_idx, y, yt, ctrl: controller, lookahead):
        remaining = yt.shape[0] - step_idx
        max_lookahead = ctrl.ssm.num_meas
        num_meas_left = min(lookahead, max_lookahead, remaining)
        y_block = y[:, step_idx:step_idx + num_meas_left, :]
        yt_block = yt[step_idx:step_idx + num_meas_left]
        return y_block, yt_block


    def _resample_indices_from_log_weights(self, log_weights, method='multinomial'):
        weights = torch.exp(log_weights)
        return self._resample_indices(weights, method=method)

    def _resample_indices(self, weights, method='multinomial'):
        if method == 'multinomial':
            return torch.multinomial(weights, weights.shape[0], replacement=True)
        if method == 'systematic':
            n = weights.shape[0]
            weight_sum = torch.sum(weights)
            if not torch.isfinite(weight_sum) or float(weight_sum.detach().item()) <= 0.0:
                raise ValueError("Systematic resampling requires positive finite weights.")
            weights = weights / weight_sum
            positions = (
                torch.arange(n, device=weights.device, dtype=weights.dtype)
                + torch.rand(1, device=weights.device, dtype=weights.dtype)
            ) / n
            cdf = torch.cumsum(weights, dim=0)
            cdf[-1] = 1.0
            return torch.searchsorted(cdf, positions, right=False)
        else:
            raise ValueError(f"Unknown resampling method: {method}")


    def _effective_sample_size_from_log_weights(self, log_weights):
        return torch.exp(-torch.logsumexp(2 * log_weights, dim=0))


    def _previous_twisting_potential(self, ctrl: controller, step_idx, t_prev, x_anc, y_block, yt_block):
        if step_idx == 0:
            return torch.zeros(x_anc.shape[0], device=x_anc.device, dtype=x_anc.dtype)
        return ctrl.log_twisting_potential(t_prev, x_anc, y_block, yt_block)

    def _next_twisting_potential(self, ctrl: controller, t_next, xt, y_next_block, yt_next_block):
        if yt_next_block.numel() == 0:
            return torch.zeros(xt.shape[0], device=xt.device, dtype=xt.dtype)
        return ctrl.log_twisting_potential(t_next, xt, y_next_block, yt_next_block)


    def _collect_twisting_runs(self):
        twisting_runs = []
        combos = self._build_twisting_combos()
        for ctrl_name, ctrl in self.ctrls_to_test.items():
            if not ctrl.supports_twisting_potential():
                print(f"Skipping twisted smoothing for {ctrl_name}: no twisting potential is available.")
                continue
            ctrl_lookahead = getattr(ctrl, 'lookahead_limit', None)
            seen_no_control = set()
            for combo in combos:
                if ctrl_lookahead is not None and ctrl_lookahead != combo['lookahead']:
                    continue
                if ctrl_name == 'no_control':
                    dedup_key = (
                        combo['num_meas'],
                        combo['resampling_mode'],
                        combo['resampling_strategy'],
                        combo['resampling_threshold_rel'],
                    )
                    if dedup_key in seen_no_control:
                        continue
                    seen_no_control.add(dedup_key)
                twisting_runs.append({
                    'ctrl_name': ctrl_name,
                    'ctrl': ctrl,
                    **combo,
                })
        return twisting_runs

    def _build_twisting_combos(self):
        num_meas = int(self.twist_config['num_meas'])
        lookahead_list = [int(val) for val in self._as_list(self.twist_config['lookahead'])]
        mode_list = self._as_list(self.twist_config['resampling_modes'])
        strategy_list = self._as_list(self.twist_config['resampling_strategies'])
        threshold_list = [float(val) for val in self._as_list(self.twist_config['adaptive_thresholds'])]

        combos = []
        for lookahead in lookahead_list:
            for mode in mode_list:
                if mode == 'never':
                    combos.append({
                        'num_meas': num_meas,
                        'lookahead': lookahead,
                        'resampling_mode': mode,
                        'resampling_strategy': strategy_list[0],
                        'resampling_threshold_rel': None,
                    })
                    continue
                for strategy in strategy_list:
                    if mode == 'adaptive':
                        for threshold_rel in threshold_list:
                            combos.append({
                                'num_meas': num_meas,
                                'lookahead': lookahead,
                                'resampling_mode': mode,
                                'resampling_strategy': strategy,
                                'resampling_threshold_rel': threshold_rel,
                            })
                    else:
                        combos.append({
                            'num_meas': num_meas,
                            'lookahead': lookahead,
                            'resampling_mode': mode,
                            'resampling_strategy': strategy,
                            'resampling_threshold_rel': None,
                        })
        return combos



        
    


    def _observation_time_grid(self, yt):
        return torch.cat([
            torch.zeros(1, device=yt.device, dtype=yt.dtype),
            yt,
        ])

    def _heatmap_time_grid(self, yt):
        t_final = float(yt[-1].item())
        full_points = int(np.round(t_final / self.ssm.dt)) + 1
        num_points = min(self.twist_config.get('plot_heatmap_num_points', 400), full_points)
        return torch.linspace(0.0, t_final, num_points, device=self.device, dtype=yt.dtype)

    def _backtrack_observation_paths(self, x_hist, ancestor_hist, terminal_indices):
        num_output_paths = x_hist[-1].shape[0]
        smoothed_paths = torch.zeros(
            (terminal_indices.shape[0], len(x_hist), x_hist[-1].shape[1]),
            device=self.device,
            dtype=x_hist[-1].dtype,
        )
        curr = terminal_indices
        smoothed_paths[:, -1, :] = x_hist[-1][curr, :]

        for hist_idx in range(len(ancestor_hist) - 1, -1, -1):
            curr = ancestor_hist[hist_idx][curr]
            smoothed_paths[:, hist_idx, :] = x_hist[hist_idx][curr, :]

        return smoothed_paths

    def _write_twisting_summary(self, results):
        csv_header = "controller,ess,ess_std,log_z_hat,log_z_hat_std"
        pretty_header = "controller\tess\tess_std\tlog_z_hat\tlog_z_hat_std"

        print(pretty_header)
        summary_path = os.path.join(self.results_root, 'twisted_smoothing_results.csv')
        with open(summary_path, 'w') as f:
            f.write(csv_header + "\n")
            for result_name, traces in results.items():
                ess_means = [float(np.mean(trace['ess'])) for trace in traces if len(trace['ess']) > 0]
                logz_vals = [float(trace['log_norm_const']) for trace in traces if 'log_norm_const' in trace]
                if not ess_means or not logz_vals:
                    continue

                row = (
                    f"{result_name},"
                    f"{np.mean(ess_means):.4f},{np.std(ess_means, ddof=1):.4f},"
                    f"{np.mean(logz_vals):.6f},{np.std(logz_vals, ddof=1):.6f}"
                )
                pretty_row = (
                    f"{result_name}\t"
                    f"{np.mean(ess_means):.4f}\t{np.std(ess_means, ddof=1):.4f}\t"
                    f"{np.mean(logz_vals):.6f}\t{np.std(logz_vals, ddof=1):.6f}"
                )
                print(pretty_row)
                f.write(row + "\n")

    def _plot_twisting_curves(self, results):
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_by_ctrl = {}
        color_idx = 0

        for result_name, traces in results.items():
            ess_traces = [np.asarray(trace['ess'], dtype=float) for trace in traces if len(trace['ess']) > 0]
            if not ess_traces:
                continue
            min_len = min(len(trace) for trace in ess_traces)
            ess_mat = np.stack([trace[:min_len] for trace in ess_traces], axis=0)
            mean_ess = np.mean(ess_mat, axis=0)
            xvals = np.arange(1, min_len + 1)
            std_ess = np.std(ess_mat, axis=0, ddof=1)

            ctrl_name = result_name.split(" (", 1)[0]
            if "always/" in result_name or "adaptive(" in result_name:
                mode = "resampled"
            else:
                mode = "no_resampling"
            if ctrl_name not in color_by_ctrl:
                color_by_ctrl[ctrl_name] = color_cycle[color_idx % len(color_cycle)]
                color_idx += 1
            color = color_by_ctrl[ctrl_name]
            linestyle = "-" if mode == "resampled" else "--"

            ax.plot(xvals, mean_ess, label=result_name, color=color, linestyle=linestyle, linewidth=2)
            if ess_mat.shape[0] > 1:
                lower = np.maximum(mean_ess - std_ess, 0.0)
                upper = mean_ess + std_ess
                ax.fill_between(xvals, lower, upper, color=color, alpha=0.12)

        ax.set_xlabel('Smoothing step')
        ax.set_ylabel('ESS')
        ax.grid(True, which='major', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.savefig(os.path.join(self.results_root, 'twisted_smoothing_results.pdf'), bbox_inches='tight')
        plt.close(fig)

    def _twisting_run_name(self, run):
        mode = run['resampling_mode']
        if mode == 'adaptive':
            suffix = f"adaptive_{run['resampling_threshold_rel']}_{run['resampling_strategy']}"
        elif mode == 'always':
            suffix = f"always_{run['resampling_strategy']}"
        else:
            suffix = "never"
        return f"{run['ctrl_name']}_L{run['lookahead']}_N{run['num_meas']}_{suffix}"

    def run_twisted_smoothing(self, run, x0, y, yt):
        ctrl = run['ctrl']
        X = x0
        x_hist = [X.clone()]
        ancestor_hist = []
        ess_vec = []
        log_norm_const = torch.zeros((), device=self.device, dtype=X.dtype)
        M = X.shape[0]
        curr_log_weights = torch.full(
            (M,),
            -np.log(M),
            device=self.device,
            dtype=X.dtype,
        )
        yt = torch.as_tensor(yt, device=self.device)

        run_label = self._twisting_run_name(run)
        for step_idx in tqdm(range(yt.shape[0]), desc=f"{run_label}"):
            if step_idx == 0:
                ancestor_indices = torch.arange(M, device=X.device)
                X_anc = X
            else:
                ess = self._effective_sample_size_from_log_weights(curr_log_weights)
                should_resample = run['resampling_mode'] == 'always' or (run['resampling_mode'] == 'adaptive' and float(ess.detach().item()) <= run['resampling_threshold_rel'] * M)
                if should_resample:
                    ancestor_indices = self._resample_indices_from_log_weights(curr_log_weights, method=run['resampling_strategy'])
                    X_anc = X[ancestor_indices, :]
                    curr_log_weights = torch.full(
                        (M,),
                        -np.log(M),
                        device=self.device,
                        dtype=X.dtype,
                    )
                else:
                    ancestor_indices = torch.arange(M, device=X.device)
                    X_anc = X

            y_block, yt_block = self._extract_observation_block(step_idx, y, yt, ctrl, lookahead=run['lookahead'])
            y_twist_next_block, yt_twist_next_block = self._extract_observation_block(step_idx + 1, y, yt, ctrl, lookahead=run['lookahead'])

            y_block_anc = y_block[ancestor_indices, :, :]
            y_next = y_block_anc[:, 0, :]
            t_prev = 0.0 if step_idx == 0 else float(yt[step_idx - 1].item())
            t_next = float(yt_block[0].item())

            N_steps = int(round((t_next - t_prev) / ctrl.ssm.dt))
            W_vec = torch.randn(
                X_anc.shape[0],
                N_steps,
                ctrl.ssm.sde.dimension_noise,
                device=X_anc.device,
                dtype=X_anc.dtype,
            ) * np.sqrt(ctrl.ssm.dt)
            XT, running_cost, cost_compensation = ctrl.simulate_controlled_paths(
                X_anc, t_prev, t_next, y_block_anc, yt_block, W_vec
            )

            log_eta_prev = self._previous_twisting_potential(ctrl, step_idx, t_prev, X_anc, y_block_anc, yt_block)
            log_likelihood = self.ssm.meas.log_g(XT, y_next)
            if y_twist_next_block.numel() > 0:
                y_twist_next_block = y_twist_next_block[ancestor_indices, :, :]
            log_eta_next = self._next_twisting_potential(ctrl, t_next, XT, y_twist_next_block, yt_twist_next_block)
            
            log_w = log_likelihood - running_cost - cost_compensation + log_eta_next - log_eta_prev
            curr_log_weights = curr_log_weights + log_w
            if not torch.all(torch.isfinite(curr_log_weights)):
                raise FloatingPointError("Encountered non-finite log weights: ")
            log_norm_increment = torch.logsumexp(curr_log_weights, dim=0)
            curr_log_weights = curr_log_weights - log_norm_increment

            log_norm_const = log_norm_const + log_norm_increment
            ess = self._effective_sample_size_from_log_weights(curr_log_weights)
            ess_vec.append((ess / M).detach().item())

            X = XT
            ancestor_hist.append(ancestor_indices)
            x_hist.append(X.clone())
        
        terminal_indices = self._resample_indices_from_log_weights(
            curr_log_weights,
            method=run['resampling_strategy'],
        )
        smoothed_paths = self._backtrack_observation_paths(
            x_hist,
            ancestor_hist,
            terminal_indices=terminal_indices,
        )
        
        trace = {
            'ess': ess_vec,
            'log_norm_const': float(log_norm_const.detach().cpu().item()),
            'norm_const': float(torch.exp(log_norm_const).detach().cpu().item()),
            'smoothed_paths': smoothed_paths.cpu().numpy(),
        }

        return trace

    def test_twisting(self):
        print("Running twisted smoothing test...")
        twisting_runs = self._collect_twisting_runs()
        if not twisting_runs:
            print("No twisted smoothing runs were scheduled.")
            return

        results = {}

        # Use a fix measurement sequence for twisting tests
        y, yt = self.generate_measurement_sequence(
            self.twist_config['num_meas'],
            num_paths=self.twist_config['num_paths'],
        )
        x0 = self.ssm.get_initial_mean(
            self.twist_config['num_paths'],
            device=self.device,
        )
        np.save(os.path.join(self.results_root, 'twisting_measurements.npy'), y[0].cpu().numpy())
        np.save(os.path.join(self.results_root, 'twisting_yt.npy'), yt.cpu().numpy())
        np.save(os.path.join(self.results_root, 'twisting_x0.npy'), x0.cpu().numpy())
        try:
            self.write_twisting_kalman_logz(y, yt, x0)
        except (TypeError, ValueError) as exc:
            print(f"Skipping Kalman log Z truth: {exc}")
        for run in twisting_runs:
            trace_name = self._twisting_run_name(run)
            results[trace_name] = []
            for ii in range(self.twist_config['num_tests']):
                with torch.inference_mode():
                    trace = self.run_twisted_smoothing(run, x0.clone(), y, yt)
                results[trace_name].append(trace)

                if ii == 0:
                    self.visualizer.plot_paths(
                        trace_name,
                        trace['smoothed_paths'],
                        yt,
                        y,
                        None,
                        plot_measurements=True,
                        plot_measurement_likelihoods=False,
                        plot_state_heatmap=False,
                    )
        
        self.write_twisting_logz_estimates(results)
        self._write_twisting_summary(results)
        self._plot_twisting_curves(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '--model-name', dest='model_name', type=str, default='test', help='Name for model to test')
    parser.add_argument('--extra', '--extra_names', '--extra-model-names', dest='extra_model_names', type=str, nargs='*', default=[], help='Additional model names to test')
    parser.add_argument('--test-name', type=str, default=None, help='Optional name for this test run output directory')
    parser.add_argument('--test', '--test-type', dest='test_type', type=str, choices=['smoothing', 'twisting'], default='smoothing', help='Which test to run')

    parser.add_argument('--seed', type=int, default=0, help='Reproducible evaluation seed (independent of training seeds)')
    parser.add_argument('--num-paths', type=int, default=100, help='Shared number of paths for the selected test type')
    parser.add_argument('--num-meas', type=int, default=None, help='Shared number of measurements for the selected test type')
    parser.add_argument('--num-tests', type=int, default=1, help='Shared number of independently generated test measurement sequences for smoothing and twisting')

    parser.add_argument('--smt-use-fixed-measurement', action='store_true', default=False, help='smoothing_config[use_fixed_measurement]')
    parser.add_argument('--smt-fixed-measurement-vector', type=float, nargs='*', default=[1], help='smoothing_config[fixed_measurement_vector]')
    parser.add_argument('--smt-deltaT-meas', type=float, default=None, help='smoothing_config[deltaT_meas]')
    
    
    parser.add_argument('--twist-lookahead', type=int, nargs='+', default=[2], help='twisting_config[lookahead]')
    parser.add_argument('--twist-include-opt', action='store_true', default=False, help='Include optimal OU controllers in twisting runs')
    parser.add_argument('--twist-resampling-modes', choices=['never', 'adaptive', 'always'], nargs='+', default=['always'], help='Twisting resampling modes')
    parser.add_argument('--twist-adaptive-thresholds', type=float, nargs='+', default=[0.5], help='Relative ESS thresholds for adaptive twisting resampling')
    parser.add_argument('--twist-resampling-strategies', choices=['multinomial', 'systematic'], nargs='+', default=['multinomial'], help='Twisting resampling strategies')
    parser.add_argument('--twist-plot-heatmap-num-points', type=int, default=None, help='twisting_config[plot_heatmap_num_points]')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print('Testing models:', [args.model_name] + args.extra_model_names)
    print('Test type:', args.test_type)
    if args.test_name is not None:
        print('Test name:', args.test_name)

    if len(args.extra_model_names) > 0:
        print("Warning: Checks for training objective match between methods not verified. Take care.")

    shared_num_meas = args.num_meas

    # this becomes the master controller which decides the sde and measurement model
    smoothing_config = {
        'num_paths': args.num_paths,
        'num_tests': args.num_tests,
        'use_fixed_measurement': args.smt_use_fixed_measurement,
        'fixed_measurement_vector': args.smt_fixed_measurement_vector,
        'num_meas': args.num_meas,  # use par.num_meas if None
        'deltaT_meas': args.smt_deltaT_meas,  # use par.deltaT_meas if None
    }
    twisting_config = {
        'num_meas': args.num_meas if args.num_meas is not None else 100,
        'num_tests': args.num_tests,
        'num_paths': args.num_paths,
        'lookahead': args.twist_lookahead,
        'include_opt': args.twist_include_opt,
        'resampling_modes': args.twist_resampling_modes,
        'adaptive_thresholds': args.twist_adaptive_thresholds,
        'resampling_strategies': args.twist_resampling_strategies,
    }
    if args.twist_plot_heatmap_num_points is not None:
        twisting_config['plot_heatmap_num_points'] = args.twist_plot_heatmap_num_points

    with open(f'models/{args.model_name}/params.json', 'r') as f:
        par = json.load(f)
    par.setdefault("path", os.path.join("models", args.model_name))

    tester = Tester(
        par,
        [args.model_name, *args.extra_model_names],
        smoothing_config,
        twisting_config,
        test_name=args.test_name,
        test_type=args.test_type,
    )
    if args.test_type == 'smoothing':
        tester.test_smoothing()
    elif args.test_type == 'twisting':
        tester.test_twisting()
