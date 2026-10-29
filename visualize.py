import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm


class SmoothingVisualizer:
    """Plot smoothing results using the existing OU and Lorenz plotting styles."""

    def __init__(self, results_root, ssm):
        self.results_root = results_root
        self.ssm = ssm

    def plot_paths(self, ctrl_name, Xmat, yt, Y, ctrl=None, plot_measurements=True, plot_measurement_likelihoods=True, plot_state_heatmap=True):
        dim = int(Xmat.shape[2])
        if dim == 3:
            self._plot_paths_3d(ctrl_name, Xmat, Y, plot_measurements=plot_measurements)
        if dim > 3 and plot_state_heatmap:
            self._plot_paths_state_heatmap(ctrl_name, Xmat, yt, Y, plot_measurements=plot_measurements)
        self._plot_paths_nd(ctrl_name, Xmat, yt, Y, ctrl, plot_measurements=plot_measurements, plot_measurement_likelihoods=plot_measurement_likelihoods)

    def plot_facit_paths(self, Xmat, norm_weight, tvec, yt, Y):
        xmat_np = self._to_numpy(Xmat)
        indices = self._stratified_resample_indices(norm_weight, xmat_np.shape[0])
        self.plot_paths("facit", xmat_np[indices], yt, Y, ctrl=None, plot_state_heatmap=False)

    def plot_heatmap(self, ctrl, ctrl_name, tvec, yt, Y, x_limits=(-30.0, 30.0)):
        # Create heatmap over first coordinate
        x_min, x_max = x_limits
        x1 = torch.linspace(float(x_min), float(x_max), 100, device=yt.device, dtype=yt.dtype)
        x_batch = torch.zeros((100, self.ssm.sde.dimension_state), device=yt.device, dtype=yt.dtype)
        x_batch[:, 0] = x1

        Z = torch.zeros((tvec.shape[0], x1.shape[0]), device=yt.device, dtype=yt.dtype)
        for ii in range(tvec.shape[0]):
            Z[ii, :] = ctrl.control(
                tvec[ii].clone(),
                x_batch,
                yt,
                Y[0:1, :].repeat_interleave(100, 0),
            )[:, 0]

        # Plot
        clamp = 3
        norm = TwoSlopeNorm(vcenter=0, vmin=-clamp, vmax=clamp)
        Z = Z.clamp(-clamp, clamp)
        plt.contourf(
            tvec.cpu().detach().numpy(),
            x_batch[:, 0].cpu().detach().numpy(),
            Z.clamp(-clamp, clamp).T.detach().cpu().numpy(),
            40,
            cmap='seismic',
            norm=norm,
        )

        plt.colorbar(label='control(time, x)')
        plt.xlabel('time')
        plt.ylabel('x')
        plt.ylim(float(x_min), float(x_max))
        out_path = os.path.join(self.results_root, f'map_{ctrl_name}.pdf')
        plt.savefig(out_path)
        plt.clf()

    def _plot_paths_3d(self, ctrl_name, Xmat, Y, plot_measurements=True):
        xmat_np = self._to_numpy(Xmat)
        y_np = self._to_numpy(Y)
        meas_dim = y_np.shape[2] if y_np.ndim >= 3 else 0
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(xmat_np.shape[0]):
            x_path = xmat_np[i, :, 0]
            y_path = xmat_np[i, :, 1]
            z_path = xmat_np[i, :, 2]
            # ax.plot(x_path, y_path, z_path, color=colors[i % len(colors)], label=f'Realization {i+1}')
            ax.plot(x_path, y_path, z_path, color='black', alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if meas_dim != 3:
            warnings.warn(
                "Observation will not be plotted in the 3D volume; see the marginal plots instead.",
                RuntimeWarning,
                stacklevel=2,
            )

        terminal_points = xmat_np[:, -1, :3]
        ax.scatter(
            terminal_points[:, 0],
            terminal_points[:, 1],
            terminal_points[:, 2],
            color='tab:blue',
            alpha=0.2,
            s=20,
        )

        if plot_measurements and meas_dim == 3:
            ax.scatter(y_np[0, :, 0], y_np[0, :, 1], y_np[0, :, 2], color='red', s=30)
            terminal_measurement = y_np[0, -1, :3]
            ax.scatter(
                [terminal_measurement[0]],
                [terminal_measurement[1]],
                [terminal_measurement[2]],
                color='crimson',
                marker='x',
                s=120,
                linewidths=2,
            )

        out_path = os.path.join(self.results_root, f'test_{ctrl_name}_3d.pdf')
        plt.savefig(out_path)
        plt.clf()

    def _plot_paths_nd(self, ctrl_name, Xmat, yt, Y, ctrl=None, plot_measurements=True, plot_measurement_likelihoods=True):
        colors = ['black', 'red', 'blue', 'green', 'orange', 'purple']
        dim = Xmat.shape[2]
        xmat_np = self._to_numpy(Xmat)
        tvec = self._infer_time_grid(Xmat, yt)
        tvec_np = self._to_numpy(tvec)
        yt_np = self._to_numpy(yt)
        y_np = self._to_numpy(Y)
        ncols = int(np.ceil(np.sqrt(dim)))
        nrows = int(np.ceil(dim / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 3 * nrows),
            squeeze=False,
            sharex=True,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()
        observed_dims = self._observed_state_dims()

        for ii in range(dim):
            ax = axes_flat[ii]
            for jj in range(len(xmat_np)):
                ax.plot(tvec_np, xmat_np[jj, :, ii].transpose(), color=colors[ii % len(colors)], alpha=0.01, label=f'X_{ii+1}')
            if plot_measurements:
                for meas_idx in observed_dims.get(ii, []):
                    for kk in range(yt_np.shape[0]):
                        mean = float(y_np[0, kk, meas_idx])
                        var = self._measurement_variance(meas_idx, mean=mean)
                        self._plot_measurement(
                            ax=ax,
                            x0=yt_np[kk],
                            mean=mean,
                            var=var,
                            width=0.05,
                            plot_likelihood=plot_measurement_likelihoods,
                        )
            ax.set_title(f'X_{ii+1}(t)')
            ax.set_ylabel('X')

        for ax in axes_flat[dim:]:
            ax.set_visible(False)

        for ii in range(dim):
            ax = axes_flat[ii]
            series = xmat_np[:, :, ii].reshape(-1)
            axis_arrays = [series]
            for meas_idx in observed_dims.get(ii, []):
                axis_arrays.append(y_np[0, :, meas_idx])
            y_min, y_max = self._compute_axis_limits(*axis_arrays)
            ax.set_ylim([y_min, y_max])
            ax.set_xlabel('Time')
        out_path = os.path.join(self.results_root, f'test_{ctrl_name}.pdf')
        plt.savefig(out_path)
        plt.clf()

    def _plot_paths_state_heatmap(self, ctrl_name, Xmat, yt, Y, plot_measurements=True):
        xmat_np = self._to_numpy(Xmat)
        tvec = self._infer_time_grid(Xmat, yt)
        tvec_np = self._to_numpy(tvec)
        yt_np = self._to_numpy(yt)
        y_np = self._to_numpy(Y)
        observed_dims = self._observed_state_dims()

        path = xmat_np[0]
        dim = path.shape[1]
        terminal_states = xmat_np[:, -1, :]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

        k_idx = np.arange(dim)
        for terminal_state in terminal_states:
            axes[0, 0].plot(k_idx, terminal_state, color="black", alpha=0.08, lw=1.0)
        axes[0, 0].plot(k_idx, terminal_states.mean(axis=0), color="tab:blue", lw=1.5, label="mean terminal state")
        axes[0, 0].axhline(0.0, color="black", lw=1, ls="--", alpha=0.6)
        axes[0, 0].set_title(r"$x_k(t_{end})$")
        axes[0, 0].set_xlabel("k")
        axes[0, 0].set_ylabel("x")
        if plot_measurements and y_np.ndim >= 3 and yt_np.size > 0:
            final_obs = y_np[0, -1]
            obs_state_idx = []
            obs_values = []
            for state_idx, meas_indices in observed_dims.items():
                if not meas_indices:
                    continue
                obs_state_idx.append(state_idx)
                obs_values.append(final_obs[meas_indices[0]])
            if obs_state_idx:
                axes[0, 0].scatter(
                    np.asarray(obs_state_idx, dtype=float),
                    np.asarray(obs_values, dtype=float),
                    color="crimson",
                    marker="x",
                    s=50,
                    linewidths=2,
                    label="final observation",
                    zorder=3,
                )
        axes[0, 0].legend(loc="best", fontsize=8, frameon=False)

        centered_path = path - path.mean(axis=1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centered_path, axis=1))
        wavenumbers = np.arange(spectrum.shape[1])
        im_spec = axes[0, 1].imshow(
            spectrum.T,
            aspect="auto",
            origin="lower",
            cmap="cividis",
            extent=[float(tvec_np[0]), float(tvec_np[-1]), float(wavenumbers[0]), float(wavenumbers[-1])],
        )
        axes[0, 1].set_title("Spatial Spectrum")
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("mode")
        if plot_measurements and y_np.ndim >= 3:
            for t_obs in yt_np:
                axes[0, 1].axvline(float(t_obs), color="white", alpha=0.15, lw=1.0)
        fig.colorbar(im_spec, ax=axes[0, 1], orientation="horizontal", fraction=0.05, pad=0.12)

        im0 = axes[1, 0].imshow(
            path.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[float(tvec_np[0]), float(tvec_np[-1]), 0, dim - 1],
        )
        axes[1, 0].set_title(r"$x_k(t)$")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("k")
        if plot_measurements and y_np.ndim >= 3:
            for t_obs in yt_np:
                axes[1, 0].axvline(float(t_obs), color="white", alpha=0.15, lw=1.0)
        fig.colorbar(im0, ax=axes[1, 0], orientation="horizontal", fraction=0.05, pad=0.12)

        std_over_paths = xmat_np.std(axis=0)
        im1 = axes[1, 1].imshow(
            std_over_paths.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[float(tvec_np[0]), float(tvec_np[-1]), 0, dim - 1],
        )
        axes[1, 1].set_title(r"$\mathrm{std}[x_k(t)]$")
        axes[1, 1].set_xlabel("t")
        axes[1, 1].set_ylabel("k")
        if plot_measurements and y_np.ndim >= 3:
            for t_obs in yt_np:
                axes[1, 1].axvline(float(t_obs), color="black", alpha=0.15, lw=1.0)
        fig.colorbar(im1, ax=axes[1, 1], orientation="horizontal", fraction=0.05, pad=0.12)

        out_path = os.path.join(self.results_root, f'test_{ctrl_name}_heatmap.pdf')
        plt.savefig(out_path)
        plt.clf()

    def _infer_time_grid(self, Xmat, yt):
        num_time_points = int(Xmat.shape[1])
        obs_grid = torch.cat((torch.zeros(1, device=yt.device, dtype=yt.dtype), yt))
        if obs_grid.shape[0] == num_time_points:
            return obs_grid

        return torch.arange(num_time_points, device=yt.device, dtype=yt.dtype) * float(self.ssm.dt)

    def _observed_state_dims(self):
        # The path overlay below represents Euclidean Gaussian observations as
        # state-space error bars. Non-Euclidean observation models have neither
        # a covariance ``R`` in these coordinates
        # nor component values that can be plotted directly as latent states.
        if not hasattr(self.ssm.meas, "R"):
            return {}
        observed_dims = {}
        H = self.ssm.meas.H.detach().cpu().numpy()
        for meas_idx, row in enumerate(H):
            nonzero = np.flatnonzero(~np.isclose(row, 0.0))
            if nonzero.size != 1:
                continue
            state_idx = int(nonzero[0])
            observed_dims.setdefault(state_idx, []).append(meas_idx)
        return observed_dims

    def _measurement_variance(self, meas_idx, mean=None):
        if not hasattr(self.ssm.meas, "R"):
            raise TypeError(
                "Euclidean measurement variance is unavailable for this observation model"
            )
        R = self.ssm.meas.R.detach().cpu().numpy()
        if R.ndim == 0:
            return float(R)
        if R.ndim == 1:
            return float(R[meas_idx])
        return float(R[meas_idx, meas_idx])

    def _stratified_resample_indices(self, weights, num_samples):
        weights = np.asarray(weights, dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("Stratified resampling requires positive weights.")
        weights = weights / weight_sum
        positions = (np.random.rand(num_samples) + np.arange(num_samples)) / num_samples
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions, side='left')

    def _to_numpy(self, value):
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _compute_axis_limits(self, *arrays, pad_fraction=0.1, equal_pad=1.0, fallback=(-1.0, 1.0)):
        finite_values = []
        for array in arrays:
            values = self._to_numpy(array).reshape(-1)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_values.append(finite)
        if not finite_values:
            warnings.warn(
                "No finite values available for plot limits; using fallback axis range.",
                RuntimeWarning,
                stacklevel=2,
            )
            return fallback
        values = np.concatenate(finite_values)
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if np.isclose(y_min, y_max):
            pad = equal_pad
        else:
            pad = pad_fraction * (y_max - y_min)
        return y_min - pad, y_max + pad

    def _plot_measurement(self, ax, x0, mean, var, width, plot_likelihood=True):
        if not np.isfinite(x0) or not np.isfinite(mean) or not np.isfinite(var):
            return
        if var <= 0.0:
            ax.plot([float(x0)], [float(mean)], 'o', color='black', markersize=4)
            return
        if not plot_likelihood:
            ax.plot([float(x0)], [float(mean)], 'o', color='black', markersize=4)
            return
        plot_gaussian_errorbar(ax=ax, x0=x0, mean=mean, var=var)


def plot_gaussian_errorbar(ax, x0, mean, var, color='red'):
    x0 = float(x0)
    mean = float(mean)
    std = float(np.sqrt(var))
    ax.errorbar([x0], [mean], yerr=[[std], [std]], fmt='o', color=color, capsize=3, markersize=4)
