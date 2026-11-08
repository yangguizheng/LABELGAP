#!/usr/bin/env python
"""A-D1, Confident Learning step: out-of-fold XGBoost probabilities on the training
partition (noisy labels), cleanlab find_label_issues(prune_by_noise_rate), and
detection quality against the true affected rows (clean labels used ONLY here, for
scoring). Writes one boolean mask per seed for the retraining runs.
Outputs: results/directionA_d1/cl_issues_seed<k>.npy, cl_detection.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from cleanlab.filter import find_label_issues  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.utils import load_yaml  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", default="configs/directionA_common.yaml")
    ap.add_argument("--out", default="results/directionA_d1")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "configs/direction_a_d1_noise_robust.yaml")
    common = load_yaml(ROOT / args.common)
    params = load_yaml(ROOT / "configs/xgboost.yaml")["params"]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / common["dataset"]["processed_file"], allow_pickle=False)
    names = [str(c) for c in d["class_names"]]
    x, yn, yc = d["X_train"], d["y_train_noisy"], d["y_train_clean"]
    xv, yvn = d["X_val"], d["y_val_noisy"]
    affected = yn != yc
    groups = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
    rows = []
    for seed in cfg["seeds"]:
        t0 = time.perf_counter()
        probs = np.zeros((len(yn), len(names)))
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(cv.split(x, yn, groups)):
            assert np.unique(yn[tr]).size == len(names), f"fold {k} misses a class"
            clf = xgb.XGBClassifier(**params, random_state=seed)
            clf.fit(x[tr], yn[tr], eval_set=[(xv, yvn)], verbose=False)
            probs[te] = clf.predict_proba(x[te], iteration_range=(0, clf.best_iteration + 1))
            print(f"seed {seed} fold {k}: best_iteration {clf.best_iteration}", flush=True)
        issues = find_label_issues(yn, probs, filter_by=cfg["methods"]["confident_learning"]["filter_by"])
        np.save(out / f"cl_issues_seed{seed}.npy", issues)
        np.save(out / f"cl_pred_probs_seed{seed}.npy", probs.astype(np.float32))
        tp = int((issues & affected).sum())
        row = {"seed": seed, "flagged": int(issues.sum()), "true_affected": int(affected.sum()),
               "true_positives": tp, "precision": tp / max(int(issues.sum()), 1),
               "recall": tp / int(affected.sum()), "minutes": (time.perf_counter() - t0) / 60}
        for c, name in enumerate(names):   # recall of the hidden rows, by their clean class
            hid = affected & (yc == c)
            if hid.sum() >= 30:
                row[f"recall_hidden_{name}"] = float(issues[hid].mean())
        for c, name in enumerate(names):   # where the false flags come from (noisy label)
            fp = issues & ~affected & (yn == c)
            if fp.sum():
                row[f"false_flags_labelled_{name}"] = int(fp.sum())
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "cl_detection.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
