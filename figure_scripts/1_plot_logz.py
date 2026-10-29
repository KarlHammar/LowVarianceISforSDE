import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# Settings for this figure.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = [
    ('Spring 2D', "../results/smoothing/spring_1mass_high"),
    ('Spring 20D', "../results/smoothing/spring_10masses_smo1"),
    ('Lorenz 63 $\\sigma_y = 5$', "../results/smoothing/lor63_easy"),
    ('Lorenz 63 $\\sigma_y = 1$', "../results/smoothing/lor63_hard"),
    ('Lorenz 96 8D', "../results/smoothing/lor96_8d_1m"),
    ('Lorenz 96 16D', "../results/smoothing/lor96_16d_1m"),
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


OUTPUT_PATH = os.path.join(OUTPUT_DIR, "1_scaling_difficulty_logz.pdf")
DEFAULT_Y_LIMIT_OFFSETS = (3.0, 1.0)
Y_LIMIT_OFFSETS = {
    "Spring 2D": (0.009, 0.015),
    "Spring 20D": (0.04, 0.06),
    r"Lorenz 63 $\sigma_y = 5$": (0.18, 0.22),
    r"Lorenz 63 $\sigma_y = 1$": (0.80, 0.80),
    "Lorenz 96 8D": (0.04, 0.06),
    "Lorenz 96 16D": (0.09, 0.11),
}
TRAINING_SEEDS = (0, 1, 2, 3, 4)


from logz_observations import load_errors


def main():
    experiment_data = []
    for title, relative_folder in EXPERIMENTS:
        data = load_errors(os.path.join(THIS_DIR, relative_folder))
        by_method = defaultdict(list)
        for controller, result in data.items():
            if controller == 'no_control':
                method, seed = 'no_control', -1
            else:
                seed, method = controller.split('_seed', 1)[1].split('_', 1)
                seed = int(seed)
            by_method[method].append((seed, result))
        for method in set(METHOD_ORDER) - {'no_control'}:
            seeds = {seed for seed, _ in by_method[method]}
            if seeds != set(TRAINING_SEEDS):
                raise ValueError(f'{title}: {method} requires seeds {TRAINING_SEEDS}; found {sorted(seeds)}')
        experiment_data.append((title, by_method))

    available = {method for _, data in experiment_data for method in data}
    methods = [method for method in METHOD_ORDER if method in available]
    methods += sorted(available - set(methods))
    height_ratios = [0.5 if title.startswith(('Spring', 'Lorenz 96')) else 1.0
                     for title, _ in experiment_data]
    fig, axes = plt.subplots(
        len(experiment_data), 1, figsize=(8.0, max(2.0, sum(height_ratios))),
        sharex=True, squeeze=False, gridspec_kw={'height_ratios': height_ratios},
    )
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, (title, data) in zip(axes[:, 0], experiment_data):
        below, above = Y_LIMIT_OFFSETS.get(title, DEFAULT_Y_LIMIT_OFFSETS)
        y_min, y_max = -below, above
        arrow_pad = 0.04 * (y_max - y_min)
        for method_index, method in enumerate(methods, 1):
            seeds = sorted(data[method])
            for seed_index, (seed, result) in enumerate(seeds):
                position = method_index + (seed_index - (len(seeds) - 1) / 2) * 0.08
                color = colors[(seed + 1) % len(colors)]
                errors, failed = result['errors'], len(result['failed'])
                if errors.size:
                    artists = ax.boxplot(
                        errors, positions=[position], widths=0.055, patch_artist=True,
                        manage_ticks=False, showfliers=False, whis=[2.5, 97.5],
                        showmeans=False,
                    )
                    for box in artists['boxes']:
                        box.set(facecolor=color, edgecolor=color, alpha=0.45, linewidth=0.9)
                    for part in ('whiskers', 'caps', 'medians'):
                        for artist in artists[part]:
                            artist.set(color=color, linewidth=0.9)
                    ci_low, ci_high = np.quantile(errors, [0.025, 0.975])
                    if not failed and ci_high < y_min:
                        ax.annotate(
                            '', xy=(position, y_min), xytext=(position, y_min + arrow_pad),
                            arrowprops=dict(arrowstyle='-|>', color=color, linewidth=1.0),
                            annotation_clip=False,
                        )
                    if not failed and ci_low > y_max:
                        ax.annotate(
                            '', xy=(position, y_max), xytext=(position, y_max - arrow_pad),
                            arrowprops=dict(arrowstyle='-|>', color=color, linewidth=1.0),
                            annotation_clip=False,
                        )
                if failed:
                    ax.plot(position, 0.08, marker='x', color=color, markersize=4,
                            transform=ax.get_xaxis_transform())
                    warnings.warn(f'{title}, {method}, seed {seed}: failed observations '
                                  f'{result["failed"]}; displayed errors are conditional on success.')
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='black', linestyle='--', linewidth=1.1)
        ax.text(0.015, 0.94, title, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))
        ax.grid(axis='y', alpha=0.25)
    axes[-1, 0].set_xticks(range(1, len(methods) + 1))
    axes[-1, 0].set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    fig.legend(
        handles=[Patch(facecolor=colors[(seed + 1) % len(colors)],
                       edgecolor=colors[(seed + 1) % len(colors)],
                       alpha=0.45, label=f'seed {seed}') for seed in TRAINING_SEEDS],
        loc='lower center', bbox_to_anchor=(0.5, 0.88), borderaxespad=0.15,
        ncol=len(TRAINING_SEEDS), frameon=False, fontsize=7,
        handlelength=1.0, handleheight=0.8, columnspacing=1.2,
    )
    fig.subplots_adjust(hspace=0.0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
