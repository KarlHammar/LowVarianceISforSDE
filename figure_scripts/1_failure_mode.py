import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch


PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PAPER_DIR)
OUTPUT_DIR = os.path.join(PAPER_DIR, "output")

# Settings: models, simulation size, and histogram appearance.
LOGVAR_SEED = 4
BOOTSTRAP_PATHS = 100_000
LOGVAR_A_SEED = 0
SIM_SEED = 123
NUM_PATHS = 512
MAX_PLOT_PATHS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIST_TIME = 0.4
HIST_BINS = 30
HIST_ALPHA = 0.35

PDF_PATH = os.path.join(OUTPUT_DIR, "1_failure_mode.pdf")

sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from SSM.SSM import SSM
from controller import multiNnController
from train import load_model_params


# Load both learned controllers and generate one shared observation.
torch.manual_seed(SIM_SEED)
np.random.seed(SIM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(DEVICE)
name_logvar = f"lor63_hard_seed{LOGVAR_SEED}_LogVar"
name_logvar_a = f"lor63_hard_seed{LOGVAR_A_SEED}_LogVar_a"

par = load_model_params(name_logvar)
par_logvar_a = load_model_params(name_logvar_a)

ssm = SSM(par["ssm_config"], device)
observation_std = torch.sqrt(torch.diag(ssm.meas.R)).detach().cpu().numpy()
ctrl_logvar = multiNnController(par["ssm_config"], par["nn_config"], device, name=par["name"], path=par["path"])
ctrl_logvar.load_model()
ctrl_logvar.eval()

ctrl_logvar_a = multiNnController(par_logvar_a["ssm_config"], par_logvar_a["nn_config"], device, name=par_logvar_a["name"], path=par_logvar_a["path"])
ctrl_logvar_a.load_model()
ctrl_logvar_a.eval()

t1 = ssm.deltaT_meas * ssm.num_meas
n_steps = int(round(t1 / ssm.dt))
times = np.arange(n_steps + 1) * ssm.dt

truth_x0 = ssm.get_initial_mean(1, device=device)
truth_path = ssm.simulate_paths(truth_x0, t1)
observations, observation_times = ssm.simulate_observations(truth_path)
observations = observations[:1].repeat(NUM_PATHS, 1, 1)

x0 = ssm.get_initial_mean(NUM_PATHS, device=device)
brownian_increments = torch.randn(
    NUM_PATHS,
    n_steps,
    ssm.sde.dimension_noise,
    device=device,
) * np.sqrt(ssm.dt)

# Simulate both controllers and the weighted bootstrap reference.
with torch.inference_mode():
    _, _, _, paths_logvar = ctrl_logvar.simulate_controlled_paths(
        x0,
        0.0,
        t1,
        observations,
        observation_times,
        brownian_increments,
        return_full=True,
    )
    _, _, _, paths_logvar_a = ctrl_logvar_a.simulate_controlled_paths(
        x0,
        0.0,
        t1,
        observations,
        observation_times,
        brownian_increments,
        return_full=True,
    )

    bootstrap_x0 = ssm.sample_initial_states(BOOTSTRAP_PATHS, device=device)
    bootstrap_paths = ssm.simulate_paths(bootstrap_x0, t1)
    bootstrap_log_weights = torch.zeros(BOOTSTRAP_PATHS, device=device)
    observation_indices = torch.round(observation_times / ssm.dt).to(torch.long)
    for obs_idx, path_idx in enumerate(observation_indices):
        bootstrap_log_weights += ssm.meas.log_g(
            bootstrap_paths[:, path_idx, :], observations[:1, obs_idx, :]
        )

    bootstrap_indices = torch.multinomial(
        torch.softmax(bootstrap_log_weights, dim=0),
        NUM_PATHS,
        replacement=True,
    )
    paths_bootstrap = bootstrap_paths[bootstrap_indices]

# Plot path clouds above the three marginal histograms.
paths_logvar = paths_logvar.cpu().numpy()
paths_logvar_a = paths_logvar_a.cpu().numpy()
paths_bootstrap = paths_bootstrap.cpu().numpy()
observations = observations[:1].detach().cpu().numpy()[0]
observation_times = observation_times.detach().cpu().numpy()
n_plot = min(MAX_PLOT_PATHS, NUM_PATHS)
path_clouds = [
    (paths_logvar, "C0"),
    (paths_bootstrap, "C1"),
    (paths_logvar_a, "C2"),
]
LEGEND_ENTRIES = [("LogVar", "C0"), ("LogVar(a)", "C2"), ("Ground truth", "C1")]
hist_index = int(np.argmin(np.abs(times - HIST_TIME)))
hist_time = times[hist_index]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
    }
)

combined_fig, combined_axes = plt.subplots(2, 3, figsize=(10.5, 4.8))

for dim, ax in enumerate(combined_axes[0]):
    for obs_time in observation_times:
        ax.axvline(obs_time, color="0.75", linestyle=":", linewidth=0.8, zorder=0)
    for paths, color in path_clouds:
        for path in paths[:n_plot]:
            ax.plot(times, path[:, dim], color=color, alpha=0.04, linewidth=0.7, zorder=1)
    ax.errorbar(
        observation_times,
        observations[:, dim],
        yerr=observation_std[dim],
        color="crimson",
        fmt="none",
        linewidth=0.9,
        capsize=2.0,
        zorder=2,
    )
    ax.set_xlabel("time")
    ax.set_ylabel("")

for dim, ax in enumerate(combined_axes[1]):
    hist_values = [paths[:, hist_index, dim] for paths, _ in path_clouds]
    hist_min = min(values.min() for values in hist_values)
    hist_max = max(values.max() for values in hist_values)
    bins = np.linspace(hist_min, hist_max, HIST_BINS + 1)
    for values, (_, color) in zip(hist_values, path_clouds):
        ax.hist(values, bins=bins, density=True, color=color, alpha=HIST_ALPHA)
    ax.set_xlabel(f"{['x', 'y', 'z'][dim]} at t={hist_time:g}")
    ax.set_ylabel("")
    if dim == 2:
        ax.legend(
            handles=[
                Patch(facecolor=color, alpha=HIST_ALPHA, label=label)
                for label, color in LEGEND_ENTRIES
            ],
            loc="upper right",
        )

combined_fig.tight_layout()
combined_fig.savefig(PDF_PATH, bbox_inches="tight")
plt.close(combined_fig)
