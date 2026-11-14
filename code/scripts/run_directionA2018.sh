#!/usr/bin/env bash
# Direction A on CSE-CIC-IDS2018: data preparation, then 3 seeds x {noisy, clean} x 5 models, then analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/srv/sda/ygz_projects/NIDS-Failure-First-Research-Protocol/env/bin/python}
mkdir -p logs
[ -f data/processed/cicids2018_labelnoise_v1.npz ] || $PY scripts/directionA2018_prepare.py 2>&1 | tee -a logs/directionA2018_run.log
for seed in 42 52 62; do
  for cond in noisy clean; do
    $PY scripts/train_baselines.py --common configs/directionA2018_common.yaml --label-condition "$cond" \
        --seeds "$seed" --base "directionA2018/$cond" --skip-existing 2>&1 | tee -a logs/directionA2018_run.log
  done
done
$PY scripts/directionA_analyze.py --common configs/directionA2018_common.yaml --runs directionA2018 \
    --results results/directionA2018 --report reports/DIRECTION_A2018_STAGE1.md \
    --interp reports/direction_a2018_interpretation.md \
    --title "CSE-CIC-IDS2018 original label vs Liu corrected label" 2>&1 | tee -a logs/directionA2018_run.log
touch logs/DIRECTION_A2018_DONE
