#!/usr/bin/env bash
# A-D1 on CSE-CIC-IDS2018: Confident Learning (detection + retraining), GCE and co-teaching on the
# noisy labels; same hyper-parameters and 40-epoch budget as the CICIDS2017 A-D1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
A="--common configs/directionA2018_common.yaml --label-condition noisy --skip-existing"
M="results/directionA2018_d1/cl_issues_seed{seed}.npy"
mkdir -p logs
$PY scripts/directionA_d1_cl.py --common configs/directionA2018_common.yaml --out results/directionA2018_d1 \
    2>&1 | tee -a logs/directionA2018_d1.log
for seed in 42 52 62; do
  $PY scripts/train_baselines.py $A --seeds "$seed" --models xgboost --exclude-train "$M" \
      --base directionA2018/d1_cl 2>&1 | tee -a logs/directionA2018_d1.log
  $PY scripts/train_baselines.py $A --seeds "$seed" --models mlp transformer --fixed-epochs 40 --exclude-train "$M" \
      --base directionA2018/d1_cl 2>&1 | tee -a logs/directionA2018_d1.log
  $PY scripts/train_baselines.py $A --seeds "$seed" --models mlp transformer --fixed-epochs 40 --loss gce --gce-q 0.7 \
      --base directionA2018/d1_gce 2>&1 | tee -a logs/directionA2018_d1.log
  $PY scripts/train_baselines.py $A --seeds "$seed" --models mlp transformer --fixed-epochs 40 \
      --coteaching-forget-rate 0.2 --coteaching-tk 10 --base directionA2018/d1_coteaching 2>&1 | tee -a logs/directionA2018_d1.log
done
$PY scripts/directionA_d1_analyze.py --common configs/directionA2018_common.yaml --runs directionA2018 \
    --out results/directionA2018_d1 --report reports/DIRECTION_A2018_D1.md \
    --interp reports/direction_a2018_d1_interpretation.md --focus Infilteration Benign --primary hidden \
    --title "CSE-CIC-IDS2018" 2>&1 | tee -a logs/directionA2018_d1.log
touch logs/DIRECTION_A2018_D1_DONE
