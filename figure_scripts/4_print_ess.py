import csv
from pathlib import Path


# Settings: this ablation uses one training seed and varies K, with B fixed at 100.
THIS_DIR = Path(__file__).resolve().parent
RESULTS_FOLDER = THIS_DIR.parent / "results/smoothing/spring_5masses_smo10_WLogVarK_sweep"
RUN_PREFIX = "spring_5masses_smo10_seed0"
K_VALUES = (1, 2, 5, 10, 20, 50, 100)
OUTPUT_PATH = THIS_DIR / "output/4_K_ablation_ess_table.tex"


# 1. Read the recorded mean RESS and its standard deviation across test runs.
with (RESULTS_FOLDER / "smoothing_results.csv").open(newline="") as results_file:
    results_by_controller = {}
    for result in csv.DictReader(results_file):
        controller_name = result["controller"]
        if controller_name in results_by_controller:
            raise ValueError(f"Duplicate results for {controller_name}")
        results_by_controller[controller_name] = result

# 2. Write the bootstrap baseline, each K, and the full-lookahead analytic baseline.
bootstrap_result = results_by_controller["no_control"]
latex_lines = [
    r"\begin{tabular}{c cc}",
    r"\toprule",
    r"Controller / $K$ & RESS & RESS std. \\",
    r"\midrule",
    f"Bootstrap & {float(bootstrap_result['ess']):.4f} & {float(bootstrap_result['ess_std']):.4f}" + r" \\",
]
for k in K_VALUES:
    result = results_by_controller[f"{RUN_PREFIX}_WLogVarK{k}"]
    latex_lines.append(rf"$K={k}$ & {float(result['ess']):.4f} & {float(result['ess_std']):.4f}" + r" \\")
analytic_result = results_by_controller["optimal(10)"]
latex_lines.extend([
    r"\midrule",
    f"Analytic & {float(analytic_result['ess']):.4f} & {float(analytic_result['ess_std']):.4f}" + r" \\",
    r"\bottomrule",
    r"\end{tabular}",
])

# 3. Save the LaTeX table.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(latex_lines) + "\n")
print(f"Wrote {OUTPUT_PATH}")
