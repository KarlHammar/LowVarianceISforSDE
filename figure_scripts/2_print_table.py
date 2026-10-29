"""Print observation-averaged RESS and bootstrap SD, mean ± SD across training seeds."""
from importlib import import_module
from pathlib import Path

table = import_module('1_print_table')

THIS_DIR = Path(__file__).resolve().parent

# Settings for this table.
EXPERIMENTS = [
    ("Spring", "../results/smoothing/spring_5masses_smo10_10"),
    ("Lorenz 63", "../results/smoothing/lorenz63_smo10_10"),
    ("Lorenz 96", "../results/smoothing/lorenz96_smo10_10"),
]
METHOD_ORDER = ["no_control", "WLogVar", "WLogVarFull", "optimal(10)"]
TRAINING_SEEDS = (0, 1, 2, 3, 4)
OUTPUT_NAME = '2_logz_mean_std.tex'
INCLUDE_ANALYTIC = True

METHOD_LABELS = {'no_control': r'\texttt{Bootstrap}', 'WLogVar': 'Recursive',
                 'WLogVarFull': 'Full', 'optimal(10)': 'Analytic'}


def main():
    matrix = table.build_value_matrix(EXPERIMENTS, METHOD_ORDER, TRAINING_SEEDS, INCLUDE_ANALYTIC)
    formatted = {metric: table.format_matrix(matrix, metric, EXPERIMENTS, METHOD_ORDER)
                 for metric in ('ress', 'std')}
    output_path = THIS_DIR / 'output' / OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as file:
        table.build_latex_table(file, formatted, METHOD_ORDER, METHOD_LABELS,
                                [title for title, _ in EXPERIMENTS], None)
    print(output_path)


if __name__ == '__main__':
    main()
