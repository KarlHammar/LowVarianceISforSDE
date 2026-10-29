#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

# jobscript 21: spring 5 masses 10meas tests
echo "RUNNING TEST: spring 5 masses 10meas"
python test.py \
  --model-name spring_5masses_smo10_seed0_WLogVar \
  --extra-model-names \
    spring_5masses_smo10_seed0_WLogVarFull \
    spring_5masses_smo10_seed1_WLogVar spring_5masses_smo10_seed1_WLogVarFull \
    spring_5masses_smo10_seed2_WLogVar spring_5masses_smo10_seed2_WLogVarFull \
    spring_5masses_smo10_seed3_WLogVar spring_5masses_smo10_seed3_WLogVarFull \
    spring_5masses_smo10_seed4_WLogVar spring_5masses_smo10_seed4_WLogVarFull \
  --test-type smoothing \
  --test-name spring_5masses_smo10_10 \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100 \
  --num-meas 10

# lorenz63 10meas tests
echo "RUNNING TEST: lorenz63 10meas"
python test.py \
  --model-name lorenz63_smo10_seed0_WLogVar \
  --extra-model-names \
    lorenz63_smo10_seed0_WLogVarFull \
    lorenz63_smo10_seed1_WLogVar lorenz63_smo10_seed1_WLogVarFull \
    lorenz63_smo10_seed2_WLogVar lorenz63_smo10_seed2_WLogVarFull \
    lorenz63_smo10_seed3_WLogVar lorenz63_smo10_seed3_WLogVarFull \
    lorenz63_smo10_seed4_WLogVar lorenz63_smo10_seed4_WLogVarFull \
  --test-type smoothing \
  --test-name lorenz63_smo10_10 \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100 \
  --num-meas 10

# lorenz96 10meas tests
echo "RUNNING TEST: lorenz96 10meas"
python test.py \
  --model-name lorenz96_smo10_seed0_WLogVar \
  --extra-model-names \
    lorenz96_smo10_seed0_WLogVarFull \
    lorenz96_smo10_seed1_WLogVar lorenz96_smo10_seed1_WLogVarFull \
    lorenz96_smo10_seed2_WLogVar lorenz96_smo10_seed2_WLogVarFull \
    lorenz96_smo10_seed3_WLogVar lorenz96_smo10_seed3_WLogVarFull \
    lorenz96_smo10_seed4_WLogVar lorenz96_smo10_seed4_WLogVarFull \
  --test-type smoothing \
  --test-name lorenz96_smo10_10 \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100 \
  --num-meas 10
