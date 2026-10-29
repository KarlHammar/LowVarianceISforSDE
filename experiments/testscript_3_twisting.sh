#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

print_test_banner() {
  local label="$1"
  printf '\n'
  printf '========================================================================\n'
  printf '========================== RUNNING TEST: %s\n' "$label"
  printf '========================================================================\n'
  printf '\n'
}

# spring 2 masses steady: method comparison
print_test_banner "spring 2 masses steady method comparison"
python test.py \
  --model-name spring_2masses_smo10_steady_WLogVarFull \
  --seed 0 \
  --test-type twisting \
  --test-name spring_2masses_smo10_steady_methods \
  --num-paths 100 \
  --num-meas 1000 \
  --num-tests 100 \
  --twist-lookahead 1 3 10 \
  --twist-resampling-modes always never \
  --twist-resampling-strategies systematic
