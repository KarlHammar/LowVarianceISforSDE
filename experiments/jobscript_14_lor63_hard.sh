#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

PREFIX=lor63_hard
COMMON_ARGS=(
  --ssm_config lorenz63_hard
  --nn_config multi_default_3d
  --override method_config.N_epochs=30000
  --override method_config.use_checkpointing=True
  --override nn_config.Z_model_config.standardization.x_std=20
  --override nn_config.Z_model_config.standardization.y_std=20
  --override nn_config.V0_model_config.standardization.x_std=20
  --override nn_config.V0_model_config.standardization.y_std=20
)

for seed in 0 1 2 3 4; do
  RUN_PREFIX=${PREFIX}_seed${seed}

  python train.py --name ${RUN_PREFIX}_RE            --method_config RE          --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_CE            --method_config CE          --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_LogVar        --method_config LogVar      --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_WLogVar       --method_config WLogVar     --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_LogVar_a      --method_config LogVar      --seed ${seed} "${COMMON_ARGS[@]}" --override method_config.detached=False &
  python train.py --name ${RUN_PREFIX}_Moment        --method_config Moment      --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_Moment_a      --method_config Moment      --seed ${seed} "${COMMON_ARGS[@]}" --override method_config.detached=False &
  wait

done
wait
