import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

# Settings

SEED = 0
MODEL_NAMES = [
    f"ou_1d_von_mises_new_seed{SEED}_RE",
    f"ou_1d_von_mises_new_seed{SEED}_CE",
    f"ou_1d_von_mises_new_seed{SEED}_Moment",
    f"ou_1d_von_mises_new_seed{SEED}_LogVar",
    f"ou_1d_von_mises_new_seed{SEED}_WLogVar",
]

LABELS = {
    "RE": "RE",
    "CE": "CE",
    "Moment": "Moment",
    "LogVar": "LogVar",
    "WLogVar": "wLogVar",
}

SIMULATION_SEED = 3317       # fixes the observation and all plotted samples
MODEL_PARTICLES = 50_000     # samples drawn from each learned controller
BOOTSTRAP_PARTICLES = 2_000_000
BOOTSTRAP_BATCH_SIZE = 100_000
HISTOGRAM_BINS = 160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PAPER_DIR)
OUTPUT_STEM = os.path.join(PAPER_DIR, "output", "von_mises_endtime")


# Imports in this project expect neuralSDE/ to be the working directory.
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from SSM.SSM import SSM
from controller import multiNnController
from train import load_model_params

torch.manual_seed(SIMULATION_SEED)
np.random.seed(SIMULATION_SEED)
device = torch.device(DEVICE)

reference_params = load_model_params(MODEL_NAMES[0])
ssm = SSM(reference_params["ssm_config"], device)
t_end = ssm.deltaT_meas * ssm.num_meas
n_steps = int(round(t_end / ssm.dt))

truth_x0 = ssm.get_initial_mean(1, device=device)
truth_path = ssm.simulate_paths(truth_x0, t_end)
observation, observation_times = ssm.simulate_observations(truth_path)
observed_angle = torch.atan2(observation[0, 0, 1], observation[0, 0, 0]).item()


# Large bootstrap posterior: simulate from the prior and weight by likelihood.
# Batching keeps the full trajectory tensor small enough for the GPU.

bootstrap_endpoints = []
bootstrap_log_weights = []

with torch.inference_mode():
    particles_left = BOOTSTRAP_PARTICLES
    while particles_left > 0:
        batch_size = min(BOOTSTRAP_BATCH_SIZE, particles_left)
        x0 = ssm.sample_initial_states(batch_size, device=device)
        paths = ssm.simulate_paths(x0, t_end)

        log_weight = torch.zeros(batch_size, device=device)
        for obs_index, obs_time in enumerate(observation_times):
            path_index = int(round(float(obs_time) / ssm.dt))
            y = observation[:, obs_index, :].repeat(batch_size, 1)
            log_weight += ssm.meas.log_g(paths[:, path_index, :], y)

        bootstrap_endpoints.append(paths[:, -1, 0].cpu())
        bootstrap_log_weights.append(log_weight.cpu())
        particles_left -= batch_size

bootstrap_endpoints = torch.cat(bootstrap_endpoints).numpy()
bootstrap_log_weights = torch.cat(bootstrap_log_weights)
bootstrap_weights = torch.softmax(bootstrap_log_weights, dim=0).numpy()


# Learned marginals: draw unweighted samples from every controlled process.

model_endpoints = {}

for model_name in MODEL_NAMES:
    print(f"Sampling {model_name}")
    params = load_model_params(model_name)
    controller = multiNnController(
        params["ssm_config"],
        params["nn_config"],
        device,
        name=params["name"],
        path=params["path"],
        compile_rollouts=False,
    )
    controller.load_model()
    controller.eval()

    y = observation.repeat(MODEL_PARTICLES, 1, 1)
    x0 = ssm.get_initial_mean(MODEL_PARTICLES, device=device)
    brownian_increments = (
        torch.randn(
            MODEL_PARTICLES,
            n_steps,
            ssm.sde.dimension_noise,
            device=device,
        )
        * np.sqrt(ssm.dt)
    )

    with torch.inference_mode():
        endpoint, _, _ = controller.simulate_controlled_paths(
            x0,
            0.0,
            t_end,
            y,
            observation_times,
            brownian_increments,
        )

    model_endpoints[model_name] = endpoint[:, 0].cpu().numpy()

# Plot all densities on common bins.

all_endpoints = [bootstrap_endpoints, *model_endpoints.values()]
x_min = min(np.quantile(values, 0.001) for values in all_endpoints)
x_max = max(np.quantile(values, 0.999) for values in all_endpoints)
bin_edges = np.linspace(x_min, x_max, HISTOGRAM_BINS + 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

posterior_density, _ = np.histogram(
    bootstrap_endpoints,
    bins=bin_edges,
    weights=bootstrap_weights,
)
posterior_density /= np.diff(bin_edges)

# Okabe-Ito inspired: distinct and colorblind-friendly.
COLORS = {
    "RE": "#0072B2",
    "CE": "#D55E00",
    "Moment": "#009E73",
    "LogVar": "#CC79A7",
    "WLogVar": "#E69F00",
    "Moment_a": "#56B4E9",
    "LogVar_a": "#8E6C8A",
}

method_names = []
method_colors = []
model_densities = []

for model_name, endpoints in model_endpoints.items():
    method = model_name.split(f"seed{SEED}_", 1)[1]
    density, _ = np.histogram(endpoints, bins=bin_edges, density=True)
    method_names.append(LABELS[method])
    method_colors.append(COLORS[method])
    model_densities.append(density)

# The observation is periodic: mark every terminal state with the observed angle.
period = float(ssm.meas.period)
observation_scale = period / float(ssm.meas.H[0, 0])
base_observed_state = observation_scale * observed_angle / (2.0 * np.pi)
k_min = int(np.floor((x_min - base_observed_state) / observation_scale)) - 1
k_max = int(np.ceil((x_max - base_observed_state) / observation_scale)) + 1
observation_states = [
    base_observed_state + k * observation_scale
    for k in range(k_min, k_max + 1)
]

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "text.color": "#303030",
    "xtick.color": "#343A40",
    "ytick.color": "#343A40",
})

os.makedirs(os.path.dirname(OUTPUT_STEM), exist_ok=True)

# Log-density ridge plot. Densities below the floor are clipped for display.

LOG_DENSITY_FLOOR = 1e-4
log_floor = np.log10(LOG_DENSITY_FLOOR)
log_positions = np.arange(len(method_names))[::-1] * 0.62
log_bootstrap = np.log10(np.maximum(posterior_density, LOG_DENSITY_FLOOR))
log_model_densities = [
    np.log10(np.maximum(density, LOG_DENSITY_FLOOR))
    for density in model_densities
]
log_top = max(
    float(log_bootstrap.max()),
    *(float(density.max()) for density in log_model_densities),
)
log_range = max(log_top - log_floor, 1e-12)

log_fig, log_ax = plt.subplots(
    figsize=(6.75, 0.52 * len(method_names) + 0.75),
    facecolor="white",
    constrained_layout=True,
)


for y, name, color, log_method in zip(
    log_positions,
    method_names,
    method_colors,
    log_model_densities,
):
    bootstrap_height = 0.48 * (log_bootstrap - log_floor) / log_range
    method_height = 0.48 * (log_method - log_floor) / log_range

    log_ax.fill_between(
        bin_centres,
        y,
        y + bootstrap_height,
        color="#D9D9D9",
        alpha=0.55,
        linewidth=0,
        zorder=1,
    )
    log_ax.plot(
        bin_centres,
        y + bootstrap_height,
        color="#4D4D4D",
        linewidth=0.9,
        zorder=2,
    )
    log_ax.fill_between(
        bin_centres,
        y,
        y + method_height,
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=3,
    )
    log_ax.plot(
        bin_centres,
        y + method_height,
        color=color,
        linestyle="-",
        linewidth=1.35,
        zorder=4,
    )
    log_ax.axhline(y, color="#B8B8B8", linewidth=0.55, zorder=0)
    log_ax.text(
        x_max - 0.025 * (x_max - x_min),
        y + 0.45,
        name,
        ha="right",
        va="top",
        fontsize=8,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
        zorder=6,
    )

for observed_state in observation_states:
    if x_min <= observed_state <= x_max:
        log_ax.axvline(
            observed_state,
            color="#555555",
            linestyle=(0, (2, 2)),
            linewidth=0.7,
            alpha=0.38,
            zorder=5,
        )

log_ax.set_yticks([])
log_ax.set_xlim(x_min, x_max)
log_ax.set_ylim(-0.12, log_positions[0] + 0.56)
log_ax.set_xlabel(r"Terminal state, $X_T$")
log_ax.set_ylabel(r"Empirical density (log scale)")
log_ax.tick_params(axis="x", labelsize=8, width=0.6, length=3)
log_ax.spines["left"].set_visible(False)
log_ax.spines["bottom"].set_color("#303030")
log_ax.spines["bottom"].set_linewidth(0.7)
log_ax.grid(False)

# Apply the serif typography consistently across the panel.
for text_object in [
    log_ax.xaxis.label,
    log_ax.yaxis.label,
    *log_ax.get_xticklabels(),
    *log_ax.texts,
]:
    text_object.set_fontfamily("STIXGeneral")

log_ridge_path = OUTPUT_STEM + ".pdf"
log_fig.savefig(log_ridge_path, bbox_inches="tight")
log_fig.savefig(log_ridge_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")

# Vector-native PGFPlots/TikZ version for inclusion in the paper.
import matplot2tikz

log_tikz_path = OUTPUT_STEM + ".tex"
matplot2tikz.save(log_tikz_path, figure=log_fig)
# matplot2tikz 0.5.4 incorrectly escapes the underscore inside this math label.
with open(log_tikz_path, "r", encoding="utf-8") as tikz_file:
    tikz_text = tikz_file.read()
    tikz_text = tikz_text.replace("\\begin{axis}[", "\\begin{axis}[\nwidth=0.8\\linewidth,\nheight=0.4\\linewidth,\nytick=\\empty,")
    tikz_text = tikz_text.replace("\n  scale=0.4,", "")
with open(log_tikz_path, "w", encoding="utf-8") as tikz_file:
    tikz_file.write(tikz_text.replace(r"X\_T", r"X_T"))
plt.close(log_fig)

print(f"Saved {log_ridge_path}, a 300 dpi PNG, and {log_tikz_path}")
