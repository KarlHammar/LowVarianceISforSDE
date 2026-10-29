#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

# jobscript 11: spring 1 mass high
echo "RUNNING TEST: spring 1 mass high"
python test.py \
  --model-name spring_1mass_high_seed0_RE \
  --extra-model-names \
    spring_1mass_high_seed0_CE spring_1mass_high_seed0_Moment spring_1mass_high_seed0_LogVar spring_1mass_high_seed0_WLogVar spring_1mass_high_seed0_Moment_a spring_1mass_high_seed0_LogVar_a \
    spring_1mass_high_seed1_RE spring_1mass_high_seed1_CE spring_1mass_high_seed1_Moment spring_1mass_high_seed1_LogVar spring_1mass_high_seed1_WLogVar spring_1mass_high_seed1_Moment_a spring_1mass_high_seed1_LogVar_a \
    spring_1mass_high_seed2_RE spring_1mass_high_seed2_CE spring_1mass_high_seed2_Moment spring_1mass_high_seed2_LogVar spring_1mass_high_seed2_WLogVar spring_1mass_high_seed2_Moment_a spring_1mass_high_seed2_LogVar_a \
    spring_1mass_high_seed3_RE spring_1mass_high_seed3_CE spring_1mass_high_seed3_Moment spring_1mass_high_seed3_LogVar spring_1mass_high_seed3_WLogVar spring_1mass_high_seed3_Moment_a spring_1mass_high_seed3_LogVar_a \
    spring_1mass_high_seed4_RE spring_1mass_high_seed4_CE spring_1mass_high_seed4_Moment spring_1mass_high_seed4_LogVar spring_1mass_high_seed4_WLogVar spring_1mass_high_seed4_Moment_a spring_1mass_high_seed4_LogVar_a \
  --test-type smoothing \
  --test-name spring_1mass_high \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100

# jobscript 12: spring 10 masses 1meas
echo "RUNNING TEST: spring 10 masses 1meas"
python test.py \
  --model-name spring_10masses_smo1_seed0_RE \
  --extra-model-names \
    spring_10masses_smo1_seed0_CE spring_10masses_smo1_seed0_Moment spring_10masses_smo1_seed0_LogVar spring_10masses_smo1_seed0_WLogVar spring_10masses_smo1_seed0_Moment_a spring_10masses_smo1_seed0_LogVar_a \
    spring_10masses_smo1_seed1_RE spring_10masses_smo1_seed1_CE spring_10masses_smo1_seed1_Moment spring_10masses_smo1_seed1_LogVar spring_10masses_smo1_seed1_WLogVar spring_10masses_smo1_seed1_Moment_a spring_10masses_smo1_seed1_LogVar_a \
    spring_10masses_smo1_seed2_RE spring_10masses_smo1_seed2_CE spring_10masses_smo1_seed2_Moment spring_10masses_smo1_seed2_LogVar spring_10masses_smo1_seed2_WLogVar spring_10masses_smo1_seed2_Moment_a spring_10masses_smo1_seed2_LogVar_a \
    spring_10masses_smo1_seed3_RE spring_10masses_smo1_seed3_CE spring_10masses_smo1_seed3_Moment spring_10masses_smo1_seed3_LogVar spring_10masses_smo1_seed3_WLogVar spring_10masses_smo1_seed3_Moment_a spring_10masses_smo1_seed3_LogVar_a \
    spring_10masses_smo1_seed4_RE spring_10masses_smo1_seed4_CE spring_10masses_smo1_seed4_Moment spring_10masses_smo1_seed4_LogVar spring_10masses_smo1_seed4_WLogVar spring_10masses_smo1_seed4_Moment_a spring_10masses_smo1_seed4_LogVar_a \
  --test-type smoothing \
  --test-name spring_10masses_smo1 \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100

# jobscript 13: lor63 easy
echo "RUNNING TEST: lor63 easy"
python test.py \
  --model-name lor63_easy_seed0_RE \
  --extra-model-names \
    lor63_easy_seed0_CE lor63_easy_seed0_LogVar lor63_easy_seed0_WLogVar lor63_easy_seed0_LogVar_a lor63_easy_seed0_Moment lor63_easy_seed0_Moment_a \
    lor63_easy_seed1_RE lor63_easy_seed1_CE lor63_easy_seed1_LogVar lor63_easy_seed1_WLogVar lor63_easy_seed1_LogVar_a lor63_easy_seed1_Moment lor63_easy_seed1_Moment_a \
    lor63_easy_seed2_RE lor63_easy_seed2_CE lor63_easy_seed2_LogVar lor63_easy_seed2_WLogVar lor63_easy_seed2_LogVar_a lor63_easy_seed2_Moment lor63_easy_seed2_Moment_a \
    lor63_easy_seed3_RE lor63_easy_seed3_CE lor63_easy_seed3_LogVar lor63_easy_seed3_WLogVar lor63_easy_seed3_LogVar_a lor63_easy_seed3_Moment lor63_easy_seed3_Moment_a \
    lor63_easy_seed4_RE lor63_easy_seed4_CE lor63_easy_seed4_LogVar lor63_easy_seed4_WLogVar lor63_easy_seed4_LogVar_a lor63_easy_seed4_Moment lor63_easy_seed4_Moment_a \
  --test-type smoothing \
  --test-name lor63_easy \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100

# jobscript 14: lor63 hard
echo "RUNNING TEST: lor63 hard"
python test.py \
  --model-name lor63_hard_seed0_RE \
  --extra-model-names \
    lor63_hard_seed0_CE lor63_hard_seed0_LogVar lor63_hard_seed0_WLogVar lor63_hard_seed0_LogVar_a lor63_hard_seed0_Moment lor63_hard_seed0_Moment_a \
    lor63_hard_seed1_RE lor63_hard_seed1_CE lor63_hard_seed1_LogVar lor63_hard_seed1_WLogVar lor63_hard_seed1_LogVar_a lor63_hard_seed1_Moment lor63_hard_seed1_Moment_a \
    lor63_hard_seed2_RE lor63_hard_seed2_CE lor63_hard_seed2_LogVar lor63_hard_seed2_WLogVar lor63_hard_seed2_LogVar_a lor63_hard_seed2_Moment lor63_hard_seed2_Moment_a \
    lor63_hard_seed3_RE lor63_hard_seed3_CE lor63_hard_seed3_LogVar lor63_hard_seed3_WLogVar lor63_hard_seed3_LogVar_a lor63_hard_seed3_Moment lor63_hard_seed3_Moment_a \
    lor63_hard_seed4_RE lor63_hard_seed4_CE lor63_hard_seed4_LogVar lor63_hard_seed4_WLogVar lor63_hard_seed4_LogVar_a lor63_hard_seed4_Moment lor63_hard_seed4_Moment_a \
  --test-type smoothing \
  --test-name lor63_hard \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100

# jobscript 15: lor96 8d 1meas
echo "RUNNING TEST: lor96 8d 1meas"
python test.py \
  --model-name lor96_8d_1m_seed0_RE \
  --extra-model-names \
    lor96_8d_1m_seed0_CE lor96_8d_1m_seed0_Moment lor96_8d_1m_seed0_LogVar lor96_8d_1m_seed0_WLogVar lor96_8d_1m_seed0_Moment_a lor96_8d_1m_seed0_LogVar_a \
    lor96_8d_1m_seed1_RE lor96_8d_1m_seed1_CE lor96_8d_1m_seed1_Moment lor96_8d_1m_seed1_LogVar lor96_8d_1m_seed1_WLogVar lor96_8d_1m_seed1_Moment_a lor96_8d_1m_seed1_LogVar_a \
    lor96_8d_1m_seed2_RE lor96_8d_1m_seed2_CE lor96_8d_1m_seed2_Moment lor96_8d_1m_seed2_LogVar lor96_8d_1m_seed2_WLogVar lor96_8d_1m_seed2_Moment_a lor96_8d_1m_seed2_LogVar_a \
    lor96_8d_1m_seed3_RE lor96_8d_1m_seed3_CE lor96_8d_1m_seed3_Moment lor96_8d_1m_seed3_LogVar lor96_8d_1m_seed3_WLogVar lor96_8d_1m_seed3_Moment_a lor96_8d_1m_seed3_LogVar_a \
    lor96_8d_1m_seed4_RE lor96_8d_1m_seed4_CE lor96_8d_1m_seed4_Moment lor96_8d_1m_seed4_LogVar lor96_8d_1m_seed4_WLogVar lor96_8d_1m_seed4_Moment_a lor96_8d_1m_seed4_LogVar_a \
  --test-type smoothing \
  --test-name lor96_8d_1m \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100

# jobscript 16: lor96 16d 1meas
echo "RUNNING TEST: lor96 16d 1meas"
python test.py \
  --model-name lor96_16d_1m_seed0_RE \
  --extra-model-names \
    lor96_16d_1m_seed0_CE lor96_16d_1m_seed0_Moment lor96_16d_1m_seed0_LogVar lor96_16d_1m_seed0_WLogVar lor96_16d_1m_seed0_Moment_a lor96_16d_1m_seed0_LogVar_a \
    lor96_16d_1m_seed1_RE lor96_16d_1m_seed1_CE lor96_16d_1m_seed1_Moment lor96_16d_1m_seed1_LogVar lor96_16d_1m_seed1_WLogVar lor96_16d_1m_seed1_Moment_a lor96_16d_1m_seed1_LogVar_a \
    lor96_16d_1m_seed2_RE lor96_16d_1m_seed2_CE lor96_16d_1m_seed2_Moment lor96_16d_1m_seed2_LogVar lor96_16d_1m_seed2_WLogVar lor96_16d_1m_seed2_Moment_a lor96_16d_1m_seed2_LogVar_a \
    lor96_16d_1m_seed3_RE lor96_16d_1m_seed3_CE lor96_16d_1m_seed3_Moment lor96_16d_1m_seed3_LogVar lor96_16d_1m_seed3_WLogVar lor96_16d_1m_seed3_Moment_a lor96_16d_1m_seed3_LogVar_a \
    lor96_16d_1m_seed4_RE lor96_16d_1m_seed4_CE lor96_16d_1m_seed4_Moment lor96_16d_1m_seed4_LogVar lor96_16d_1m_seed4_WLogVar lor96_16d_1m_seed4_Moment_a lor96_16d_1m_seed4_LogVar_a \
  --test-type smoothing \
  --test-name lor96_16d_1m \
  --num-paths 1000 \
  --seed 0 \
  --num-tests 100
