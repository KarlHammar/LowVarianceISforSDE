"""Print observation-averaged RESS and bootstrap SD, mean ± SD across training seeds."""
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent

# Settings for this table.
EXPERIMENTS = [
    ('Spring 2D', "../results/smoothing/spring_1mass_high"),
    ('Spring 20D', "../results/smoothing/spring_10masses_smo1"),
    ('Lorenz 63 $\\sigma_y = 5$', "../results/smoothing/lor63_easy"),
    ('Lorenz 63 $\\sigma_y = 1$', "../results/smoothing/lor63_hard"),
    ('Lorenz 96 8D', "../results/smoothing/lor96_8d_1m"),
    ('Lorenz 96 16D', "../results/smoothing/lor96_16d_1m"),
]
METHOD_ORDER = ["no_control", "RE", "CE", "Moment", "LogVar", "WLogVar", "Moment_a", "LogVar_a"]
TRAINING_SEEDS = (0, 1, 2, 3, 4)
OUTPUT_NAME = '1_logz_mean_std.tex'
INCLUDE_ANALYTIC = False

METHOD_LABELS = dict(no_control=r'\texttt{Bootstrap}', RE=r'$\re$', CE=r'$\ce$',
                     Moment=r'$\moment$', LogVar=r'$\logvar$', WLogVar=r'$\wlogvar$',
                     Moment_a=r'$\momenta$', LogVar_a=r'$\logvara$')

from decimal import Decimal
import math
from statistics import mean, stdev

import csv


def read_rows(path):
    with open(path, newline='') as stream:
        yield from csv.DictReader(stream)

def parse_method_and_seed(name):
    if name == 'no_control' or name.startswith('optimal('):
        return name, -1
    seed, method = name.split('_seed', 1)[1].split('_', 1)
    return method, int(seed)

def calculate_ress_and_logz_sd(folder, methods, seeds, include_analytic):
    data = {}
    for metric, filename, column in (
        ('ress', 'smoothing_results.csv', 'ess'),
        ('std', 'smoothing_observation_estimates.csv', 'log_z_hat_bootstrap_sd'),
    ):
        controllers = defaultdict(list)
        for row in read_rows(folder / filename):
            controllers[row['controller']].append(float(row[column]))
        data[metric] = defaultdict(list)
        for controller, values in controllers.items():
            if controller == 'Correct' or (controller.startswith('optimal(') and
                    not (include_analytic and controller == 'optimal(10)')):
                continue
            method, _ = parse_method_and_seed(controller)
            data[metric][method].append(mean(values))

    return {
        method: {metric: {'mean': mean(values[method]),
                          'std': sd_across_training_seeds(values[method])}
                 for metric, values in data.items()}
        for method in data['ress']
    }

def build_value_matrix(EXPERIMENTS=EXPERIMENTS, METHOD_ORDER=METHOD_ORDER,
                       TRAINING_SEEDS=TRAINING_SEEDS, INCLUDE_ANALYTIC=INCLUDE_ANALYTIC):
    """Rows follow METHOD_ORDER; columns follow EXPERIMENTS. Values are unscaled."""
    matrix = {'ress': {'mean': [], 'std': []}, 'std': {'mean': [], 'std': []}}
    for method in METHOD_ORDER:
        matrix['ress']['mean'].append([])
        matrix['ress']['std'].append([])
        matrix['std']['mean'].append([])
        matrix['std']['std'].append([])

    for title, folder in EXPERIMENTS:
        statistics = calculate_ress_and_logz_sd(
            THIS_DIR / folder, METHOD_ORDER, TRAINING_SEEDS, INCLUDE_ANALYTIC)
        for row, method in enumerate(METHOD_ORDER):
            if method not in statistics:
                for metric in matrix.values():
                    for values in metric.values():
                        values[row].append(None)
                continue
            matrix['ress']['mean'][row].append(statistics[method]['ress']['mean'])
            matrix['ress']['std'][row].append(statistics[method]['ress']['std'])
            matrix['std']['mean'][row].append(statistics[method]['std']['mean'])
            matrix['std']['std'][row].append(statistics[method]['std']['std'])
    return matrix


def sd_across_training_seeds(values):
    if len(values) == 1:
        return None
    return stdev(values) if all(map(math.isfinite, values)) else math.nan


def format_value(value, uncertainty=None):
    """Three significant digits; uncertainty matches the mean's decimal places."""
    mantissa, exponent = f'{value:.2e}'.split('e')
    exponent = int(exponent)
    if uncertainty is not None:
        if not math.isfinite(uncertainty):
            return r'\mathrm{NaN}'
        if abs(value) > 1000:
            return rf'{uncertainty / 10**exponent:.2f}\mathrm{{e}}{{{exponent}}}'
        return f'{uncertainty:.{max(0, 2 - exponent)}f}'
    if abs(value) > 1000:
        return rf'{mantissa}\mathrm{{e}}{{{exponent}}}'
    return format(Decimal(f'{value:.2e}'), 'f')


def rank_background(score, ranked):
    """Give each distinct displayed value an equally spaced shade."""
    if not math.isfinite(score):
        return r'\cellcolor[rgb]{1.000,1.000,1.000}'
    fraction = ranked.index(score) / (len(ranked) - 1) if len(ranked) > 1 else 0.0
    red = 0.80 + 0.20 * fraction
    green = 0.88 + 0.12 * fraction
    return rf'\cellcolor[rgb]{{{red:.3f},{green:.3f},1.000}}'


def format_matrix(matrix, metric, EXPERIMENTS=EXPERIMENTS, METHOD_ORDER=METHOD_ORDER):
    """Turn the numeric mean/std matrices into a matrix of LaTeX cells."""
    means = matrix[metric]['mean']
    uncertainties = matrix[metric]['std']
    cells = [['' for experiment in EXPERIMENTS] for method in METHOD_ORDER]
    direction = -1 if metric == 'ress' else 1

    for col, (title, folder) in enumerate(EXPERIMENTS):
        # Reference ranks receive equally spaced shades; displayed ties share a shade.
        scores = []
        for row, method in enumerate(METHOD_ORDER):
            score = direction * float(f'{100 * means[row][col]:.2e}') if means[row][col] is not None else math.nan
            scores.append(score)
        ranked = sorted({score for method, score in zip(METHOD_ORDER, scores)
                         if not method.startswith('optimal(') and math.isfinite(score)})
        color_ranks = sorted({score for score in scores if math.isfinite(score)})

        for row, method in enumerate(METHOD_ORDER):
            if means[row][col] is None:
                continue
            value = 100 * means[row][col]
            score = scores[row]
            
            background = '' if title == 'Spring 2D' else rank_background(score, color_ranks)
            
            if not math.isfinite(value):
                cells[row][col] = background + '--'
                continue
            
            text = format_value(value)
            if uncertainties[row][col] is not None:
                text += rf' \pm {format_value(value, 100 * uncertainties[row][col])}'
            
            if title != 'Spring 2D' and not method.startswith('optimal(') and score in ranked[:2]:
                command = 'mathbf' if score == ranked[0] else 'underline'
                text = '\\' + command + '{' + text + '}'
            cells[row][col] = background + '$' + text + '$'
    return cells


def print_value_rows(file, cells, METHOD_ORDER=METHOD_ORDER, METHOD_LABELS=METHOD_LABELS):
    for row, method in enumerate(METHOD_ORDER):
        print(METHOD_LABELS[method] + ' & ' + ' & '.join(cells[row]) + r' \\', file=file)


def build_latex_table(file, formatted_matrix, METHOD_ORDER=METHOD_ORDER, METHOD_LABELS=METHOD_LABELS,
                      column_labels=None, font_size=r'\scriptsize'):
    if column_labels is None:
        column_labels = [r'Spring $d=2$', r'Spring $d=20$',
                         r'L63 $\sigma_y = 5$', r'L63 $\sigma_y = 1$',
                         r'L96 $d=8$', r'L96 $d=16$']
    if font_size:
        print(font_size, file=file)
    print(r'\setlength{\tabcolsep}{3pt}', file=file)
    print(r'\begin{tabular}{l' + 'c' * len(column_labels) + '}', file=file)
    print(r'\toprule', file=file)
    print('Method & ' + ' & '.join(column_labels) + r' \\', file=file)

    print(r'\midrule', file=file)
    print(r'\multicolumn{' + str(len(column_labels) + 1) + r'}{c}{Mean RESS $\uparrow$ ($\times 100$)} \\', file=file)
    print(r'\midrule', file=file)
    print_value_rows(file, formatted_matrix['ress'], METHOD_ORDER, METHOD_LABELS)

    print(r'\midrule', file=file)
    print(r'\multicolumn{' + str(len(column_labels) + 1) + r'}{c}{Mean $\operatorname{sd}(\hat{\ell}_{1000})$ $\downarrow$ ($\times 100$)} \\', file=file)
    print(r'\midrule', file=file)
    print_value_rows(file, formatted_matrix['std'], METHOD_ORDER, METHOD_LABELS)

    print(r'\bottomrule', file=file)
    print(r'\end{tabular}', file=file)


def main():
    matrix = build_value_matrix()
    formatted_matrix = {
        'ress': format_matrix(matrix, 'ress'),
        'std': format_matrix(matrix, 'std'),
    }
    output_path = THIS_DIR / 'output' / OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as file:
        build_latex_table(file, formatted_matrix)
    print(output_path)


if __name__ == '__main__':
    main()
