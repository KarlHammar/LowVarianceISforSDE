"""Simulate a configured state-space model and inspect its paths and observations.

Run from the repository root, for example::

    python visualize_ssm.py --ssm_config default --num-paths 5 --seed 0

The tool loads a model from ``configs/``, prints summary statistics for the
simulated states and observations, and saves path and observation plots in
``visualizations/`` (or ``--output-dir``). For three-dimensional states it
also saves a 3D path plot.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from SSM.SSM import SSM


def load_sde_config(sde_name, config_path='configs/sde_configs.json'):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    return configs[sde_name]


def load_meas_config(meas_name, config_path='configs/meas_configs.json'):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    return configs[meas_name]


def load_ssm_config(ssm_name, config_path='configs/ssm_configs.json'):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    ssm_params = dict(configs[ssm_name])
    ssm_params['sde_config'] = load_sde_config(ssm_params['sde_config'])
    ssm_params['meas_config'] = load_meas_config(ssm_params['meas_config'])
    return ssm_params


def one_hot_observation_map(meas_config):
    if meas_config['name'] != 'normal':
        return {}

    mapping = {}
    for meas_idx, row in enumerate(meas_config['H']):
        nonzero = [idx for idx, value in enumerate(row) if value != 0]
        if len(nonzero) == 1:
            state_idx = nonzero[0]
            scale = row[state_idx]
            mapping[meas_idx] = (state_idx, scale)
    return mapping


def plot_state_paths(paths, observations, obs_times, dt, meas_config, out_path):
    paths_np = paths.detach().cpu()
    obs_np = observations.detach().cpu()
    obs_times_np = obs_times.detach().cpu()
    n_paths, n_steps, dim_state = paths_np.shape
    time_grid = torch.arange(n_steps, dtype=torch.float32) * dt
    obs_map = one_hot_observation_map(meas_config)

    fig, axes = plt.subplots(dim_state, 1, figsize=(10, max(3, 2.5 * dim_state)), squeeze=False, sharex=True)
    colors = ['black', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown']

    for state_idx in range(dim_state):
        ax = axes[state_idx, 0]
        for path_idx in range(n_paths):
            ax.plot(
                time_grid,
                paths_np[path_idx, :, state_idx],
                color=colors[path_idx % len(colors)],
                alpha=0.45,
                linewidth=1,
            )

        for meas_idx, (observed_state_idx, scale) in obs_map.items():
            if observed_state_idx != state_idx:
                continue
            for path_idx in range(n_paths):
                ax.scatter(
                    obs_times_np,
                    obs_np[path_idx, :, meas_idx] / scale,
                    marker='x',
                    color='tab:red',
                    s=35,
                    linewidths=1.5,
                    label=f'observation {meas_idx}' if path_idx == 0 else None,
                )

        ax.set_ylabel(f'x[{state_idx}]')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best')

    axes[-1, 0].set_xlabel('time')
    fig.suptitle('Latent SDE paths')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_observations(observations, obs_times, out_path):
    obs_np = observations.detach().cpu()
    obs_times_np = obs_times.detach().cpu()
    _, _, dim_obs = obs_np.shape

    fig, axes = plt.subplots(dim_obs, 1, figsize=(10, max(3, 2.5 * dim_obs)), squeeze=False, sharex=True)
    for meas_idx in range(dim_obs):
        ax = axes[meas_idx, 0]
        for path_idx in range(obs_np.shape[0]):
            ax.plot(
                obs_times_np,
                obs_np[path_idx, :, meas_idx],
                marker='o',
                linestyle='none',
                alpha=0.75,
            )
        ax.set_ylabel(f'y[{meas_idx}]')

    axes[-1, 0].set_xlabel('time')
    fig.suptitle('Observations')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _format_float_list(values):
    return '[' + ', '.join(f'{float(value):.6g}' for value in values) + ']'


def print_data_stats(name, data):
    data_cpu = data.detach().cpu().to(torch.float64)
    dim = data_cpu.shape[-1]
    flat = data_cpu.reshape(-1, dim)

    print(f"{name} shape: {tuple(data_cpu.shape)}")
    print(f"{name} global mean: {float(flat.mean()):.6g}")
    print(f"{name} global std: {float(flat.std(unbiased=False)):.6g}")
    print(f"{name} per-dim mean: {_format_float_list(flat.mean(dim=0))}")
    print(f"{name} per-dim std: {_format_float_list(flat.std(dim=0, unbiased=False))}")


def main():
    parser = argparse.ArgumentParser(description='Visualize state-space model paths and observations')
    parser.add_argument('--ssm_config', type=str, required=True, help='SSM config name')
    parser.add_argument('--num-paths', type=int, default=5, help='Number of realizations')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Directory for plots')
    parser.add_argument('--seed', type=int, default=None, help='Optional torch random seed')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssm_config = load_ssm_config(args.ssm_config)
    ssm = SSM(ssm_config, device)

    x0 = ssm.sample_initial_states(args.num_paths, device=device)
    paths = ssm.simulate_paths(x0, ssm.T)
    observations, obs_times = ssm.simulate_observations(paths)

    print(f"SSM: {args.ssm_config}")
    print(f"SDE: {ssm_config['sde_config']['name']}")
    print(f"Measurement: {ssm_config['meas_config']['name']}")
    print(f"Number of realizations: {args.num_paths}")
    print(f"Time horizon: {ssm.T}")
    print(f"Time step: {ssm.dt}")
    print(f"Measurement times: {obs_times.detach().cpu().tolist()}")
    print_data_stats('x', paths)
    print_data_stats('y', observations)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.ssm_config)
    plot_state_paths(paths, observations, obs_times, ssm.dt, ssm_config['meas_config'], f'{prefix}_paths.png')
    plot_observations(observations, obs_times, f'{prefix}_observations.png')

    print(f"Saved plots with prefix: {prefix}")


if __name__ == '__main__':
    main()
