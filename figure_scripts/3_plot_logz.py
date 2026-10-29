#!/usr/bin/env python3
"""Plot twisted smoothing log normalizing-constant estimates."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Settings: edit these paths, then run this script directly.
THIS_DIR = Path(__file__).resolve().parent
NEURALSDE_DIR = THIS_DIR.parent
RESULTS_FOLDER = NEURALSDE_DIR / "results/twisting/spring_2masses_smo10_steady_methods"
OUTPUT_PATH = THIS_DIR / "output/3_twisting_logz.pdf"

MODE_LABELS = {
    "always": "resampling",
    "never": "no resampling",
    "adaptive": "adaptive",
}
MODE_ORDER = ["always", "never", "adaptive"]
COLORS = {
    "always": "#4c78a8",
    "never": "#f58518",
    "adaptive": "#54a24b",
}


# 1. Read the reference log Z.
# Twisting currently logs an analytical Correct row, not an Estimate row.
reference_csv_path = RESULTS_FOLDER / "twisting_correct_logz.csv"
with reference_csv_path.open(newline="") as reference_file:
    reference_row = next(
        reference_record for reference_record in csv.DictReader(reference_file)
        if reference_record["label"] == "Correct"
    )
reference_logz = float(reference_row["log_z"])

# 2. Read repeated particle-filter estimates, grouped by mode and lookahead.
estimates_csv_path = RESULTS_FOLDER / "twisting_logz_estimates.csv"
logz_estimates_by_mode = defaultdict(lambda: defaultdict(list))
lookaheads = {0}
all_logz_estimates = []
always_resampling_logz_estimates = []
with estimates_csv_path.open(newline="") as estimates_file:
    for estimate_row in csv.DictReader(estimates_file):
        controller_name = estimate_row["controller"]
        if "_L" not in controller_name:
            continue
        # Example: no_control_L1_N1000_always_stratified.
        controller_method, particle_filter_settings = controller_name.rsplit("_L", 1)
        lookahead, particle_count_text, resampling_mode, *resampling_options = particle_filter_settings.split("_")
        if resampling_mode not in MODE_ORDER:
            continue
        lookahead = 0 if controller_method == "no_control" else int(lookahead)
        logz_estimate = float(estimate_row["log_z_hat"])
        logz_estimates_by_mode[resampling_mode][lookahead].append(logz_estimate)
        lookaheads.add(lookahead)
        all_logz_estimates.append(logz_estimate)
        if resampling_mode == "always":
            always_resampling_logz_estimates.append(logz_estimate)
if not logz_estimates_by_mode:
    raise ValueError(f"No plottable samples found in {estimates_csv_path}.")

# 3. Choose axis limits from the always-resampling runs, or all runs if absent.
lookaheads = sorted(lookaheads)
resampling_modes = [mode for mode in MODE_ORDER if mode in logz_estimates_by_mode]
mode_offsets = np.linspace(-0.16, 0.16, len(resampling_modes)) if len(resampling_modes) > 1 else [0.0]
axis_limit_estimates = always_resampling_logz_estimates or all_logz_estimates
y_min = min(reference_logz, min(axis_limit_estimates))
y_max = max(reference_logz, max(axis_limit_estimates))
span = y_max - y_min
if span <= 0:
    span = max(abs(y_min), 1.0)
pad = 0.08 * span
y_min -= pad
y_max += pad
arrow_pad = 0.04 * (y_max - y_min)

# 4. Draw each box directly at its lookahead tick, offset by resampling mode.
fig, ax = plt.subplots(figsize=(5.4, 3.0), constrained_layout=True)
legend_handles = []
for resampling_mode, mode_offset in zip(resampling_modes, mode_offsets):
    color = COLORS[resampling_mode]
    estimates_by_lookahead = logz_estimates_by_mode[resampling_mode]
    for lookahead_tick, lookahead in enumerate(lookaheads, start=1):
        if lookahead not in estimates_by_lookahead:
            continue
        logz_estimates = estimates_by_lookahead[lookahead]
        box_position = lookahead_tick + mode_offset
        boxplot_artists = ax.boxplot(
            logz_estimates, positions=[box_position], widths=0.22,
            patch_artist=True, manage_ticks=False, showfliers=False,
        )
        for part in ("boxes", "whiskers", "caps", "medians"):
            for plot_artist in boxplot_artists[part]:
                plot_artist.set_color(color)
        for box in boxplot_artists["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.45)

        # Mark boxes whose estimates lie entirely outside the displayed range.
        if max(logz_estimates) < y_min:
            ax.annotate(
                "",
                xy=(box_position, y_min),
                xytext=(box_position, y_min + arrow_pad),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.1),
                annotation_clip=False,
            )
        if min(logz_estimates) > y_max:
            ax.annotate(
                "",
                xy=(box_position, y_max),
                xytext=(box_position, y_max - arrow_pad),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.1),
                annotation_clip=False,
            )

    legend_handles.append(
        plt.Line2D([0], [0], color=color, lw=6, alpha=0.45, label=MODE_LABELS.get(resampling_mode, resampling_mode))
    )

# 5. Add the reference line, lookahead labels, and mode legend.
ax.axhline(reference_logz, color="black", linestyle="--", linewidth=1.1)
ax.set_ylim(y_min, y_max)
ax.set_xticks(range(1, len(lookaheads) + 1))
ax.set_xticklabels([rf"$L={lookahead}$" for lookahead in lookaheads])
ax.set_ylabel("")
ax.yaxis.tick_left()
ax.grid(True, axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(handles=legend_handles, frameon=False, loc="best")

# 6. Save the figure.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
