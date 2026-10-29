#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

PREFIX=spring_1mass_high
COMMON_ARGS=(
  --ssm_config spring_1masses_smo1
  --nn_config multi_default_2d
  --override method_config.N_epochs=10000
)

for seed in 0 1 2 3 4; do
  RUN_PREFIX=${PREFIX}_seed${seed}

  python train.py --name ${RUN_PREFIX}_RE            --method_config RE          --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_CE            --method_config CE          --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_Moment        --method_config Moment      --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_LogVar        --method_config LogVar      --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_WLogVar       --method_config WLogVar     --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_Moment_a      --method_config Moment      --seed ${seed} "${COMMON_ARGS[@]}" --override method_config.detached=False &
  python train.py --name ${RUN_PREFIX}_LogVar_a      --method_config LogVar      --seed ${seed} "${COMMON_ARGS[@]}" --override method_config.detached=False &
  wait

done
wait
