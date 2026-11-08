#!/usr/bin/env bash
# Direction A on CSE-CIC-IDS2018: attempted-flow convention variants.
# Rules: configs/direction_a2018_attempted.yaml (committed before this script is run).
# K: retrain the clean-label models only (the noisy-label runs are the Stage-I runs); E: retrain both conditions.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
LOG=logs/directionA2018_attempted.log
mkdir -p logs
$PY scripts/directionA2018_attempted_variants.py 2>&1 | tee -a "$LOG"
for seed in 42 52 62; do
  $PY scripts/train_baselines.py --common configs/directionA2018_attK_common.yaml --label-condition clean \
      --seeds "$seed" --base directionA2018_attK/clean --skip-existing 2>&1 | tee -a "$LOG"
  for cond in noisy clean; do
    $PY scripts/train_baselines.py --common configs/directionA2018_attE_common.yaml --label-condition "$cond" \
        --seeds "$seed" --base "directionA2018_attE/$cond" --skip-existing 2>&1 | tee -a "$LOG"
  done
done
$PY scripts/directionA2018_attempted_analyze.py 2>&1 | tee -a "$LOG"
touch logs/DIRECTION_A2018_ATTEMPTED_DONE
