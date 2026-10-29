#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

PREFIX=spring_2masses_smo10_steady
COMMON_ARGS=(
  --ssm_config spring_2masses_smo10_steady
  --nn_config multi_default_4d
  --override method_config.N_epochs=40000
  --override method_config.skip_last_potential_training=false
  --override method_config.use_checkpointing=False
  --override nn_config.Z_model_config.standardization.x_std=2
  --override nn_config.Z_model_config.standardization.y_std=2
  --override nn_config.V0_model_config.standardization.x_std=2
  --override nn_config.V0_model_config.standardization.y_std=2
)

python train.py --name ${PREFIX}_WLogVarFull    --method_config WLogVar      "${COMMON_ARGS[@]}" --override method_config.training_procedure=full &

wait
