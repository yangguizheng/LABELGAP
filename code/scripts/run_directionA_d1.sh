#!/usr/bin/env bash
# A-D1: Confident Learning (detection + retraining), GCE and co-teaching on the noisy labels.
# Hyper-parameters mirror configs/direction_a_d1_noise_robust.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
A="--common configs/directionA_common.yaml --label-condition noisy --skip-existing"
mkdir -p logs
$PY scripts/directionA_d1_cl.py 2>&1 | tee -a logs/directionA_d1.log
for seed in 42 52 62; do
  $PY scripts/train_baselines.py $A --seeds "$seed" --models xgboost \
      --exclude-train "results/directionA_d1/cl_issues_seed{seed}.npy" --base directionA/d1_cl 2>&1 | tee -a logs/directionA_d1.log
  $PY scripts/train_baselines.py $A --seeds "$seed" --models mlp transformer --fixed-epochs 40 \
      --exclude-train "results/directionA_d1/cl_issues_seed{seed}.npy" --base directionA/d1_cl 2>&1 | tee -a logs/directionA_d1.log
  $PY scripts/train_baselines.py $A --seeds "$seed" --models mlp transformer --fixed-epochs 40 --loss gce --gce-q 0.7 \
      --base directionA/d1_gce 2>&1 | tee -a logs/directionA_d1.log
  $PY scripts/train_baselines.py $A --seeds "$seed" --models mlp transformer --fixed-epochs 40 \
      --coteaching-forget-rate 0.2 --coteaching-tk 10 --base directionA/d1_coteaching 2>&1 | tee -a logs/directionA_d1.log
done
$PY scripts/directionA_d1_analyze.py 2>&1 | tee -a logs/directionA_d1.log
touch logs/DIRECTION_A_D1_DONE
