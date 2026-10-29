from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Settings for this figure.
THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent / "models"
SMOOTHING_WINDOW = 100
TRAINING_SEEDS = (0, 1, 2, 3, 4)
OUTPUT_DIR = THIS_DIR / "output"
EXPERIMENTS = [
    ("Spring", "spring_5masses_smo10", "spring"),
    ("Lorenz 63", "lorenz63_smo10", "lorenz_63"),
    ("Lorenz 96", "lorenz96_smo10", "lorenz_96"),
]


# 1. Read full and recursive training runs for each experiment.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for title, run_prefix, output_name in EXPERIMENTS:
    full_run_folders = [MODELS_DIR / f"{run_prefix}_seed{seed}_WLogVarFull" for seed in TRAINING_SEEDS]
    recursive_run_folders = [MODELS_DIR / f"{run_prefix}_seed{seed}_WLogVar" for seed in TRAINING_SEEDS]
    full_metrics = [pd.read_csv(folder / "metrics_all.csv") for folder in full_run_folders]
    recursive_metrics = [pd.read_csv(folder / "metrics_all.csv") for folder in recursive_run_folders]
    controller_loss_columns = sorted(
        (column for column in recursive_metrics[0]
         if len(column.split("_")) == 2 and column.endswith("_loss")
         and column.split("_", 1)[0].isdigit()),
        key=lambda column: int(column.split("_", 1)[0]),
    )
    if not controller_loss_columns:
        raise ValueError(f"{title}: missing recursive controller-loss columns")
    fig, ax = plt.subplots(figsize=(9, 3.8))

    # 2. Plot full controller training using absolute smoothed losses.
    seed_traces = []
    for seed_metrics in full_metrics:
        loss_values = seed_metrics["loss"].dropna().to_numpy(dtype=float)
        loss_values = loss_values[np.isfinite(loss_values) & (loss_values > 1e-12)]
        if not len(loss_values):
            raise ValueError("No positive finite loss values in loss")
        smoothed_losses = pd.Series(loss_values).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
        seed_traces.append(smoothed_losses)

    # Use the common trace length and average absolute smoothed losses in log space.
    trace_length = min(map(len, seed_traces))
    log_seed_losses = np.log10(np.stack([trace[:trace_length] for trace in seed_traces]))
    mean_loss = 10 ** log_seed_losses.mean(axis=0)
    epochs = 0 + np.arange(1, trace_length + 1)
    ax.plot(epochs, mean_loss, color="C0", linewidth=1)
    if len(seed_traces) > 1:
        log_seed_std = log_seed_losses.std(axis=0, ddof=1)
        lower_loss = 10 ** (np.log10(mean_loss) - log_seed_std)
        upper_loss = 10 ** (np.log10(mean_loss) + log_seed_std)
        ax.fill_between(epochs, lower_loss, upper_loss, color="C0", alpha=0.14, linewidth=0)
    label_index = min(len(epochs) - 1, len(epochs) // 8)
    ax.text(epochs[label_index], mean_loss[label_index], "Full", color="C0",
            fontsize=9, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8})
    full_training_epochs = trace_length
    ax.axvline(full_training_epochs, color="C0", linestyle="--", linewidth=0.8)

    # 3. Plot each recursive stage; controller and value training share its epoch offset.
    stage_start = 0
    for stage_index, controller_loss_column in enumerate(controller_loss_columns):
        network_index = len(controller_loss_columns) - stage_index

        # Controller loss for this stage.
        seed_traces = []
        for seed_metrics in recursive_metrics:
            loss_values = seed_metrics[controller_loss_column].dropna().to_numpy(dtype=float)
            loss_values = loss_values[np.isfinite(loss_values) & (loss_values > 1e-12)]
            if not len(loss_values):
                raise ValueError("No positive finite loss values in " + controller_loss_column)
            smoothed_losses = pd.Series(loss_values).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
            seed_traces.append(smoothed_losses)

        # Use the common trace length and average absolute smoothed losses in log space.
        trace_length = min(map(len, seed_traces))
        log_seed_losses = np.log10(np.stack([trace[:trace_length] for trace in seed_traces]))
        mean_loss = 10 ** log_seed_losses.mean(axis=0)
        epochs = stage_start + np.arange(1, trace_length + 1)
        ax.plot(epochs, mean_loss, color="C1", linewidth=1)
        if len(seed_traces) > 1:
            log_seed_std = log_seed_losses.std(axis=0, ddof=1)
            lower_loss = 10 ** (np.log10(mean_loss) - log_seed_std)
            upper_loss = 10 ** (np.log10(mean_loss) + log_seed_std)
            ax.fill_between(epochs, lower_loss, upper_loss, color="C1", alpha=0.14, linewidth=0)
        label_index = min(len(epochs) - 1, len(epochs) // 8)
        ax.text(epochs[label_index], mean_loss[label_index], rf"$u_{{{network_index}}}$", color="C1",
                fontsize=9, va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8})
        stage_end = int(epochs[-1])

        # Value loss for the same stage. The last stage intentionally has no value training.
        if stage_index < len(controller_loss_columns) - 1:
            value_loss_column = controller_loss_column.replace("_loss", "_V_net_loss")
            seed_traces = []
            for seed_metrics in recursive_metrics:
                loss_values = seed_metrics[value_loss_column].dropna().to_numpy(dtype=float)
                loss_values = loss_values[np.isfinite(loss_values) & (loss_values > 1e-12)]
                if not len(loss_values):
                    raise ValueError("No positive finite loss values in " + value_loss_column)
                smoothed_losses = pd.Series(loss_values).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
                seed_traces.append(smoothed_losses)

            # Use the common trace length and average absolute smoothed losses in log space.
            trace_length = min(map(len, seed_traces))
            log_seed_losses = np.log10(np.stack([trace[:trace_length] for trace in seed_traces]))
            mean_loss = 10 ** log_seed_losses.mean(axis=0)
            epochs = stage_start + np.arange(1, trace_length + 1)
            ax.plot(epochs, mean_loss, color="C2", linewidth=1)
            if len(seed_traces) > 1:
                log_seed_std = log_seed_losses.std(axis=0, ddof=1)
                lower_loss = 10 ** (np.log10(mean_loss) - log_seed_std)
                upper_loss = 10 ** (np.log10(mean_loss) + log_seed_std)
                ax.fill_between(epochs, lower_loss, upper_loss, color="C2", alpha=0.14, linewidth=0)
            label_index = min(len(epochs) - 1, len(epochs) // 8)
            ax.text(epochs[label_index], mean_loss[label_index], rf"$V_{{{network_index}}}$", color="C2",
                    fontsize=9, va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8})
            stage_end = max(stage_end, int(epochs[-1]))
        stage_start = stage_end
        ax.axvline(stage_end, color="0.75", linestyle=":", linewidth=0.5)

    # 4. Format the axes and save this experiment's figure.
    ax.set_xlabel("Cumulative training epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=True))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="y", labelsize=10, pad=4)
    ax.set_xlim(1, max(stage_end, full_training_epochs))
    ax.grid(True, which="both", alpha=0.22)
    fig.tight_layout()
    output_file = OUTPUT_DIR / f"2_train_traces_{output_name}.png"
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_file}")
