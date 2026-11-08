#!/usr/bin/env bash
# Direction A Stage I formal runs: 3 seeds x {noisy, clean} x 5 models, sequential (timing metrics).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
mkdir -p logs
for seed in 42 52 62; do
  for cond in noisy clean; do
    $PY scripts/train_baselines.py --common configs/directionA_common.yaml --label-condition "$cond" \
        --seeds "$seed" --base "directionA/$cond" --skip-existing 2>&1 | tee -a logs/directionA_run.log
  done
done
touch logs/DIRECTION_A_DONE
