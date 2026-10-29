from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Settings for this figure.
THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent / "models"
SMOOTHING_WINDOW = 100
OUTPUT_PATH = THIS_DIR / "output" / "3_twisting_train_traces.png"
RUN_NAME = "spring_2masses_smo10_steady_WLogVarFull"

# 1. Read this one named training run; do not pool other matching folders.
metrics = pd.read_csv(MODELS_DIR / RUN_NAME / "metrics_all.csv")
value_loss_columns = sorted(
    (column for column in metrics if column.endswith("_V_net_loss")),
    key=lambda column: int(column.split("_", 1)[0]),
)
if not value_loss_columns:
    raise ValueError(f"{RUN_NAME}: missing value-loss columns")
fig, ax = plt.subplots(figsize=(9, 3.8))

# 2. Plot controller training using absolute smoothed losses.
loss_values = metrics["loss"].dropna().to_numpy(dtype=float)
loss_values = loss_values[np.isfinite(loss_values) & (loss_values > 1e-12)]
if not len(loss_values):
    raise ValueError("No positive finite loss values in loss")
smoothed_losses = pd.Series(loss_values).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
epochs = np.arange(1, len(smoothed_losses) + 1)
ax.plot(epochs, smoothed_losses, color="C1", linewidth=1)
label_index = min(len(epochs) - 1, len(epochs) // 8)
ax.text(epochs[label_index], smoothed_losses[label_index], r"$u$", color="C1",
        fontsize=9, va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8})
stage_start = int(epochs[-1])
ax.axvline(stage_start, color="0.75", linestyle=":", linewidth=0.5)

# 3. Plot the value-network stages consecutively after controller training.
for stage_index, value_loss_column in enumerate(value_loss_columns):
    network_index = len(value_loss_columns) - stage_index
    loss_values = metrics[value_loss_column].dropna().to_numpy(dtype=float)
    loss_values = loss_values[np.isfinite(loss_values) & (loss_values > 1e-12)]
    if not len(loss_values):
        raise ValueError("No positive finite loss values in " + value_loss_column)
    smoothed_losses = pd.Series(loss_values).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
    epochs = stage_start + np.arange(1, len(smoothed_losses) + 1)
    ax.plot(epochs, smoothed_losses, color="C2", linewidth=1)
    label_index = min(len(epochs) - 1, len(epochs) // 8)
    ax.text(epochs[label_index], smoothed_losses[label_index], rf"$V_{{{network_index}}}$", color="C2",
            fontsize=9, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8})
    stage_start = int(epochs[-1])
    ax.axvline(stage_start, color="0.75", linestyle=":", linewidth=0.5)

# 4. Format the axes and save the figure.
ax.set_xlabel("Cumulative training epochs")
ax.set_ylabel("Loss")
ax.set_yscale("log")
ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=True))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.tick_params(axis="y", labelsize=10, pad=4)
ax.set_xlim(1, stage_start)
ax.grid(True, which="both", alpha=0.22)
fig.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
