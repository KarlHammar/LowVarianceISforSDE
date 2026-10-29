#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

PREFIX=lorenz96_smo10
COMMON_ARGS=(
  --ssm_config lorenz96_8d_10m
  --nn_config multi_default_8d
  --override method_config.optimizer_params.lr=0.001
  --override method_config.use_checkpointing=False
  --override method_config.N_epochs=30000
  --override nn_config.Z_model_config.standardization.x_std=3
  --override nn_config.Z_model_config.standardization.y_std=3
  --override nn_config.V0_model_config.standardization.x_std=3
  --override nn_config.V0_model_config.standardization.y_std=3
)

for seed in 0 1 2 3 4; do
  RUN_PREFIX=${PREFIX}_seed${seed}

  python train.py --name ${RUN_PREFIX}_WLogVar        --method_config WLogVar          --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_WLogVarFull    --method_config WLogVar          --seed ${seed} "${COMMON_ARGS[@]}" --override method_config.optimizer_params.lr=0.001 --override method_config.training_procedure=full --override method_config.N_epochs=50000 --override method_config.skip_potential_training=true &

done

wait
