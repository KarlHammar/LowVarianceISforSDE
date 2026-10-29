#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

PREFIX=spring_5masses_smo10_seed0
COMMON_ARGS=(
  --ssm_config spring_5masses_smo10
  --nn_config multi_default_10d
  --method_config WLogVar
  --override method_config.N_epochs=30000
  --override method_config.use_checkpointing=False
  --override method_config.M_obs_per_batch=100
  --override method_config.B_minibatch_per_obs=100
)

# All seven runs use all B=100 inner realizations for the controller loss.
# Only the inner K used to construct the potential/value-network target changes.
# They run concurrently on the same GPU.
python train.py --seed 0 --name ${PREFIX}_WLogVarK1   "${COMMON_ARGS[@]}" --override method_config.K_potential_target=1   &
python train.py --seed 0 --name ${PREFIX}_WLogVarK2   "${COMMON_ARGS[@]}" --override method_config.K_potential_target=2   &
python train.py --seed 0 --name ${PREFIX}_WLogVarK5   "${COMMON_ARGS[@]}" --override method_config.K_potential_target=5   &
python train.py --seed 0 --name ${PREFIX}_WLogVarK10  "${COMMON_ARGS[@]}" --override method_config.K_potential_target=10  &
python train.py --seed 0 --name ${PREFIX}_WLogVarK20  "${COMMON_ARGS[@]}" --override method_config.K_potential_target=20  &
python train.py --seed 0 --name ${PREFIX}_WLogVarK50  "${COMMON_ARGS[@]}" --override method_config.K_potential_target=50  &
python train.py --seed 0 --name ${PREFIX}_WLogVarK100 "${COMMON_ARGS[@]}" --override method_config.K_potential_target=100 &

wait
