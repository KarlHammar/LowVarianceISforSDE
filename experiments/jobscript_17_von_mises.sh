#!/usr/bin/env bash
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

PREFIX=ou_1d_von_mises_new
COMMON_ARGS=(
  --master_config default
  --ssm_config ou_1d_von_mises
  --override ssm_config.sde_config.Sigma[0][0]=0.3
  --override ssm_config.sde_config.A[0][0]=-1.0
  --override ssm_config.deltaT_meas=1.0
  --override method_config.N_epochs=30000
)

for seed in 0 1 2 3 4; do
  RUN_PREFIX=${PREFIX}_seed${seed}

  python train.py --name ${RUN_PREFIX}_RE       --method_config RE      --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_CE       --method_config CE      --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_Moment   --method_config Moment  --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_LogVar   --method_config LogVar  --seed ${seed} "${COMMON_ARGS[@]}" &
  python train.py --name ${RUN_PREFIX}_WLogVar  --method_config WLogVar --seed ${seed} "${COMMON_ARGS[@]}" &
  wait
done
wait
