import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PAPER_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PAPER_DIR, "output")
FIGURE_WIDTH_CM = 12.0
FIGURE_HEIGHT_CM = 6.0
TRAINING_SEEDS = (0, 1, 2, 3, 4)

# Settings: edit these values, then run this script directly.
RUN_PREFIX = "spring_5masses_smo10"
NUM_PATHS = 4096
SIMULATION_SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "2_l2_error.pdf")
TIKZ_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "2_l2_error.tex")
BOUNDARY_EPS = None  # None uses ssm.dt * 1e-4, just after each observation.

sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from SSM.SSM import SSM
from controller import multiNnController, optimalOUcontroller
from train import load_model_params


def load_controller(name, device):
    par = load_model_params(name)

    ctrl = multiNnController(par["ssm_config"], par["nn_config"], device, name=par["name"], path=par["path"])
    ctrl.load_model()
    ctrl.eval()
    return ctrl, par


def has_complete_controller_weights(name):
    par = load_model_params(name)
    model_path = par["path"]
    num_meas = par["ssm_config"]["num_meas"]
    return all(
        os.path.isfile(os.path.join(model_path, f"state_dict_Z_model_{index}.pt"))
        for index in range(num_meas)
    )


# Load the five training seeds for each method.
torch.manual_seed(SIMULATION_SEED)
np.random.seed(SIMULATION_SEED)

device = torch.device(DEVICE)
recursive_names = [f"{RUN_PREFIX}_seed{seed}_WLogVar" for seed in TRAINING_SEEDS]
full_names = [f"{RUN_PREFIX}_seed{seed}_WLogVarFull" for seed in TRAINING_SEEDS]
for name in recursive_names + full_names:
    if not has_complete_controller_weights(name):
        raise FileNotFoundError(f"{name}: missing controller weights; all five training seeds are required")

first_controller, par = load_controller(recursive_names[0], device)
ssm = SSM(par["ssm_config"], device)
opt_controller = optimalOUcontroller(par["ssm_config"], device, lookahead_limit=ssm.num_meas)

x0 = ssm.get_initial_mean(NUM_PATHS, device=device)
x_paths = ssm.simulate_paths(x0, ssm.T)
y, yt = ssm.simulate_observations(x_paths)

controllers = {
    "recursive": [first_controller] + [load_controller(name, device)[0] for name in recursive_names[1:]],
    "full": [load_controller(name, device)[0] for name in full_names],
}
boundary_eps = BOUNDARY_EPS
if boundary_eps is None:
    boundary_eps = ssm.dt * 1e-4
# At exact observation times, evaluate just after the measurement has been consumed.
# This avoids float32/searchsorted ambiguity about whether y_i is still "future".

# Evaluate all controllers on the same paths and observations.
times = np.arange(x_paths.shape[1]) * ssm.dt
curves = {"recursive": [], "full": []}
with torch.inference_mode():
    for group, group_controllers in controllers.items():
        for ctrl in group_controllers:
            curve = []
            for step in range(x_paths.shape[1]):
                t_eval = step * ssm.dt + boundary_eps
                x_t = x_paths[:, step, :]
                opt_u = opt_controller.control(t_eval, x_t, yt, y)
                u = ctrl.control(t_eval, x_t, yt, y)
                curve.append(torch.linalg.vector_norm(u - opt_u, dim=1).mean().item())
            curves[group].append(curve)
recursive_curves = curves["recursive"]
full_curves = curves["full"]

times = np.asarray(times)
recursive_curves = np.asarray(recursive_curves)
full_curves = np.asarray(full_curves)
recursive_mean = recursive_curves.mean(axis=0)
full_mean = full_curves.mean(axis=0)

print("time,recursive,full")
for t, recursive, full in zip(times, recursive_mean, full_mean):
    print(f"{t:.6g},{recursive:.8g},{full:.8g}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#b0b0b0",
        "grid.alpha": 1.0,
    }
)
cm_to_inches = 1.0 / 2.54
fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH_CM * cm_to_inches, FIGURE_HEIGHT_CM * cm_to_inches)
)

ax.plot(times, recursive_mean, color="C1", linewidth=1.0, label="Recursive")
if recursive_curves.shape[0] > 1:
    recursive_std = recursive_curves.std(axis=0, ddof=1)
    ax.fill_between(
        times,
        np.maximum(recursive_mean - recursive_std, 0.0),
        recursive_mean + recursive_std,
        color="C1",
        alpha=0.16,
        linewidth=0,
    )

ax.plot(times, full_mean, color="C0", linewidth=1.0, label="Full")
if full_curves.shape[0] > 1:
    full_std = full_curves.std(axis=0, ddof=1)
    ax.fill_between(
        times,
        np.maximum(full_mean - full_std, 0.0),
        full_mean + full_std,
        color="C0",
        alpha=0.16,
        linewidth=0,
    )

for obs_time in yt.detach().cpu().numpy():
    ax.axvline(obs_time, color="0.72", linestyle=":", linewidth=0.7, zorder=0)

ax.set_xlabel(r"Time, $t$")
ax.set_ylabel(r"$\mathbf{E}_{\mathbb{P}^{X,Y}_t}[\|u^\theta-u^\star\|_2]$")
ax.set_xlim(times[0], times[-1])
ax.set_ylim(bottom=0.0)
ax.legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig(OUTPUT_PATH)
if TIKZ_OUTPUT_PATH:
    os.makedirs(os.path.dirname(TIKZ_OUTPUT_PATH), exist_ok=True)

    import matplot2tikz


    matplot2tikz.save(TIKZ_OUTPUT_PATH, figure=fig)
    with open(TIKZ_OUTPUT_PATH, "r", encoding="utf-8") as handle:
        tikz = handle.read()
    tikz = tikz.replace(
        "\\begin{axis}[",
        "\\begin{axis}[\nwidth=12cm,\nheight=6cm,",
        1,
    )
    with open(TIKZ_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(tikz)
    print(f"Saved TikZ plot to {TIKZ_OUTPUT_PATH}", file=sys.stderr)
plt.close(fig)
print(f"Saved plot to {OUTPUT_PATH}", file=sys.stderr)
