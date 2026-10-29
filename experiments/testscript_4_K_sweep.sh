#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

echo "RUNNING TEST: spring 5 masses WLogVar K sweep"
python test.py \
  --model-name spring_5masses_smo10_seed0_WLogVarK5 \
  --extra-model-names \
    spring_5masses_smo10_seed0_WLogVarK1 \
    spring_5masses_smo10_seed0_WLogVarK2 \
    spring_5masses_smo10_seed0_WLogVarK10 \
    spring_5masses_smo10_seed0_WLogVarK20 \
    spring_5masses_smo10_seed0_WLogVarK50 \
    spring_5masses_smo10_seed0_WLogVarK100 \
  --seed 0 \
  --test-type smoothing \
  --test-name spring_5masses_smo10_WLogVarK_sweep \
  --num-paths 1000 \
  --num-tests 100 \
  --num-meas 10
