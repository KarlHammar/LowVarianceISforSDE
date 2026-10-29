import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Settings for this figure.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "..", "models")
OUTPUT_TEMPLATE = "1_train_traces_{slug}.png"
SMOOTHING_WINDOW = 100
TRAINING_SEEDS = (0, 1, 2, 3, 4)
START_EPOCH = 201
MAX_ABS_VALUE = 1e12
MIN_LOG_VALUE = 1e-12
EXPERIMENTS = [
    ("Spring 2D", "spring_1mass_high"),
    ("Spring 20D", "spring_10masses_smo1"),
    (r"Lorenz 63 $\sigma_y = 5$", "lor63_easy"),
    (r"Lorenz 63 $\sigma_y = 1$", "lor63_hard"),
    ("Lorenz 96 8D", "lor96_8d_1m"),
    ("Lorenz 96 16D", "lor96_16d_1m"),
    ("Von Mises", "ou_1d_von_mises_new"),
]
METHOD_ORDER = ["no_control", "RE", "CE", "Moment", "LogVar", "WLogVar", "Moment_a", "LogVar_a"]
METHOD_LABELS = {
    "no_control": "Bootstrap",
    "RE": "RE",
    "CE": "CE",
    "Moment": "Moment",
    "LogVar": "LogVar",
    "WLogVar": "wLogVar",
    "Moment_a": "Moment(a)",
    "LogVar_a": "LogVar(a)",
}
OUTPUT_DIR = os.path.join(THIS_DIR, "output")


# 1. Read the training metrics for each experiment, grouped by method and seed.
os.makedirs(OUTPUT_DIR, exist_ok=True)
for title, run_prefix in EXPERIMENTS:
    methods = [method for method in METHOD_ORDER if method != "no_control"]
    if run_prefix == "ou_1d_von_mises_new":
        methods = ["RE", "CE", "Moment", "LogVar", "WLogVar"]
    metrics_by_method = defaultdict(list)
    for method in methods:
        for training_seed in TRAINING_SEEDS:
            run_name = f"{run_prefix}_seed{training_seed}_{method}"
            metrics_path = os.path.join(MODELS_DIR, run_name, "metrics_all.csv")
            metrics = pd.read_csv(metrics_path, float_precision="round_trip")
            metrics_by_method[method].append((training_seed, metrics))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 2. Create stacked loss plots on the left and full-height summary plots beside them.
    fig, axes = plt.subplots(
        len(methods), 3, figsize=(11.5, 3.8),
        sharex="col", squeeze=False,
    )
    loss_axes = axes[:, 0]
    # Replace the small axes in columns 2 and 3 with full-height subplots.
    for ax in axes[:, 1:].flat:
        ax.remove()
    ress_axis = fig.add_subplot(len(methods), 3, (2, 3 * len(methods) - 1))
    log_weight_std_axis = fig.add_subplot(len(methods), 3, (3, 3 * len(methods)))

    # 3. LOSS: one stacked plot per method.
    for method_index, method in enumerate(methods):
        ax = loss_axes[method_index]
        color = colors[method_index % len(colors)]
        # Read, filter, and smooth loss for each seed.
        seed_traces = []
        for training_seed, metrics in metrics_by_method[method]:
            # Full training uses plain names; recursive training prefixes stage 0.
            column = "loss" if "loss" in metrics else "0_loss"
            metric_values = metrics[column].to_numpy(dtype=float)
            valid = np.isfinite(metric_values) & (np.abs(metric_values) <= MAX_ABS_VALUE)
            valid &= metric_values != 0
            metric_values = metric_values[valid]
            if not len(metric_values):
                raise ValueError(f"{title}, {method}, seed {training_seed}: no valid values in {column}")
            # Trailing moving average, using all available epochs near the start.
            smoothed_values = pd.Series(metric_values).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
            seed_traces.append(smoothed_values)

        visible_losses = np.concatenate([trace[START_EPOCH - 1:] for trace in seed_traces]) if seed_traces else []
        yscale = "log" if len(visible_losses) and np.all(np.asarray(visible_losses) > MIN_LOG_VALUE) else "linear"
        has_trace = False
        if seed_traces:
            # Average over seeds in log space on log axes, otherwise in ordinary space.
            trace_length = min(map(len, seed_traces))
            seed_values = np.stack([trace[:trace_length] for trace in seed_traces], axis=1)
            if yscale == "log":
                seed_values = np.log10(seed_values)
            mean_values = seed_values.mean(axis=1)
            if len(seed_traces) > 1:
                seed_std = seed_values.std(axis=1, ddof=1)
                lower, upper = mean_values - seed_std, mean_values + seed_std
                if yscale == "log":
                    lower, upper = 10 ** lower, 10 ** upper
            if yscale == "log":
                mean_values = 10 ** mean_values
            keep = np.arange(1, trace_length + 1) >= START_EPOCH
            epochs = np.arange(1, trace_length + 1)[keep]
            if len(epochs):
                # Draw the seed average and its spread after the initial epochs.
                ax.plot(epochs, mean_values[keep], color=color, linewidth=1.0, label=METHOD_LABELS.get(method, method))
                if len(seed_traces) > 1:
                    ax.fill_between(epochs, lower[keep], upper[keep], color=color, alpha=0.14, linewidth=0)
                has_trace = True

        # Give each method its own compact loss row, with shared epoch ticks.
        is_last = method_index == len(methods) - 1
        # Format this loss row: epoch axis, scale, and grid.
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        if yscale == "log":
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(True, which="major", alpha=0.28)
        ax.grid(True, which="minor", alpha=0.14, linewidth=0.45)
        if is_last:
            ax.set_xlabel("Cumulative training epochs")

        # Keep labels inside each short loss row so adjacent rows do not collide.
        y_min, y_max = ax.get_ylim()
        if yscale == "log":
            candidates = mticker.LogLocator(base=10, subs=(1, 2, 5), numticks=100).tick_values(y_min, y_max)
            transform = np.log10
        else:
            candidates = mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(y_min, y_max)
            transform = np.asarray

        # Leave space at row boundaries so labels on adjacent rows cannot collide.
        low, high = transform([y_min, y_max])
        positions = transform(candidates)
        candidates = candidates[(positions > low + 0.12 * (high - low)) &
                                (positions < high - 0.12 * (high - low))]
        if len(candidates) < 2:
            # Narrow log ranges need ordinary round numbers between decades.
            candidates = mticker.MaxNLocator(nbins=8).tick_values(y_min, y_max)
            positions = transform(candidates[candidates > 0]) if yscale == "log" else transform(candidates)
            if yscale == "log":
                candidates = candidates[candidates > 0]
            candidates = candidates[(positions > low + 0.12 * (high - low)) &
                                    (positions < high - 0.12 * (high - low))]
        if len(candidates) > 2:
            candidates = candidates[[0, -1]]
        ax.yaxis.set_major_locator(mticker.FixedLocator(candidates))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.2g}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis="y", labelsize=9, pad=3)
        ax.tick_params(axis="x", labelsize=9, pad=3)
        ax.text(0.02, 0.86, METHOD_LABELS.get(method, method), transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))
        if method_index == len(methods) // 2:
            ax.set_ylabel("Loss")
        if not is_last:
            ax.tick_params(labelbottom=False)
        if method_index > 0:
            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_color("0.55")
            ax.spines["top"].set_linewidth(0.6)
        ax.spines["bottom"].set_color("black" if is_last else "0.55")
        ax.spines["bottom"].set_linewidth(0.8 if is_last else 0.6)
        if not has_trace:
            ax.text(0.5, 0.5, "No trace", transform=ax.transAxes, ha="center", va="center", fontsize=8)

    # 4. RELATIVE ESS: all methods on the middle panel.
    ax = ress_axis
    summary_has_traces = False
    for method_index, method in enumerate(methods):
        color = colors[method_index % len(colors)]
        # Read, filter, and smooth relative ESS for each seed.
        seed_traces = []
        for training_seed, metrics in metrics_by_method[method]:
            # Full training uses plain names; recursive training prefixes stage 0.
            column = "ress" if "ress" in metrics else "0_ress"
            metric_values = metrics[column].to_numpy(dtype=float)
            valid = np.isfinite(metric_values) & (np.abs(metric_values) <= MAX_ABS_VALUE)
            valid &= metric_values != 0
            valid &= (metric_values >= 0) & (metric_values <= 1)
            metric_values = metric_values[valid]
            if not len(metric_values):
                raise ValueError(f"{title}, {method}, seed {training_seed}: no valid values in {column}")
            # Trailing moving average, using all available epochs near the start.
            smoothed_values = pd.Series(metric_values).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
            seed_traces.append(smoothed_values)

        if seed_traces:
            # Average relative ESS across seeds in ordinary space.
            trace_length = min(map(len, seed_traces))
            seed_values = np.stack([trace[:trace_length] for trace in seed_traces], axis=1)
            mean_values = seed_values.mean(axis=1)
            if len(seed_traces) > 1:
                seed_std = seed_values.std(axis=1, ddof=1)
                lower, upper = mean_values - seed_std, mean_values + seed_std
                lower, upper = np.maximum(lower, 0), np.minimum(upper, 1)
            keep = np.arange(1, trace_length + 1) >= START_EPOCH
            epochs = np.arange(1, trace_length + 1)[keep]
            if len(epochs):
                # Draw the seed average and its spread after the initial epochs.
                ax.plot(epochs, mean_values[keep], color=color, linewidth=1.0, label=METHOD_LABELS.get(method, method))
                if len(seed_traces) > 1:
                    ax.fill_between(epochs, lower[keep], upper[keep], color=color, alpha=0.14, linewidth=0)
                summary_has_traces = True

    ax.set_ylabel(f"Relative ESS ({SMOOTHING_WINDOW}-epoch mean)")
    # Format the relative ESS panel.
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", alpha=0.14, linewidth=0.45)
    ax.set_xlabel("Cumulative training epochs")
    if not summary_has_traces:
        ax.text(0.5, 0.5, "No traces", transform=ax.transAxes, ha="center", va="center")

    # 5. LOG-WEIGHT STANDARD DEVIATION: all methods on the right panel.
    ax = log_weight_std_axis
    summary_has_traces = False
    for method_index, method in enumerate(methods):
        color = colors[method_index % len(colors)]
        # Read, filter, and smooth log-weight standard deviation for each seed.
        seed_traces = []
        for training_seed, metrics in metrics_by_method[method]:
            # Full training uses plain names; recursive training prefixes stage 0.
            column = "logw_std" if "logw_std" in metrics else "0_logw_std"
            metric_values = metrics[column].to_numpy(dtype=float)
            valid = np.isfinite(metric_values) & (np.abs(metric_values) <= MAX_ABS_VALUE)
            valid &= metric_values != 0
            valid &= metric_values >= MIN_LOG_VALUE
            metric_values = metric_values[valid]
            if not len(metric_values):
                raise ValueError(f"{title}, {method}, seed {training_seed}: no valid values in {column}")
            # Trailing moving average, using all available epochs near the start.
            smoothed_values = pd.Series(metric_values).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
            seed_traces.append(smoothed_values)

        if seed_traces:
            # Average the smoothed log-weight standard deviations across seeds in log space.
            trace_length = min(map(len, seed_traces))
            seed_values = np.stack([trace[:trace_length] for trace in seed_traces], axis=1)
            seed_values = np.log10(seed_values)
            mean_values = seed_values.mean(axis=1)
            if len(seed_traces) > 1:
                seed_std = seed_values.std(axis=1, ddof=1)
                lower, upper = mean_values - seed_std, mean_values + seed_std
                lower, upper = 10 ** lower, 10 ** upper
            mean_values = 10 ** mean_values
            keep = np.arange(1, trace_length + 1) >= START_EPOCH
            epochs = np.arange(1, trace_length + 1)[keep]
            if len(epochs):
                # Draw the seed average and its spread after the initial epochs.
                ax.plot(epochs, mean_values[keep], color=color, linewidth=1.0, label=METHOD_LABELS.get(method, method))
                if len(seed_traces) > 1:
                    ax.fill_between(epochs, lower[keep], upper[keep], color=color, alpha=0.14, linewidth=0)
                summary_has_traces = True

    ax.set_ylabel(f"Std(log weight) ({SMOOTHING_WINDOW}-epoch mean)")
    # Format the log-weight standard deviation panel.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", alpha=0.14, linewidth=0.45)
    ax.set_xlabel("Cumulative training epochs")
    if not summary_has_traces:
        ax.text(0.5, 0.5, "No traces", transform=ax.transAxes, ha="center", va="center")

    # 6. Set the spacing and save this experiment's figure.
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.92, wspace=0.32, hspace=0)
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_TEMPLATE.format(slug=slug))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
