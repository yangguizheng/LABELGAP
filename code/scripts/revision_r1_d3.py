#!/usr/bin/env python
"""LABELGAP revision, round 1 (descriptive): reruns A-D3 (group-held-out Confident Learning) with the functions of
scripts/directionA_d3_grouped_cv.py and the same seeds, saving the per-flow flags, then reports the
omission-specific precision (both datasets) and the CSE-CIC-IDS2018 precision and recall under convention K.
The rerun is compared with the saved Stage-I results (results/directionA_d3/grouped_cv.csv); any difference is
reported, never hidden.
Plan: configs/revision_round1.yaml (committed before this script is run).
Outputs: results/revision_round1/d3_*.npy, d3_grouped_cv.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cleanlab.filter import find_label_issues  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402

from directionA_d3_grouped_cv import oof_probs, time_groups  # noqa: E402
from src.utils import load_yaml  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    plan = load_yaml(ROOT / "configs/revision_round1.yaml")
    out = ROOT / plan["outputs"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(ROOT / "configs/direction_a_d3_grouped_cv.yaml")
    params = load_yaml(ROOT / "configs/xgboost.yaml")["params"]
    ref = pd.read_csv(ROOT / "results/directionA_d3/grouped_cv.csv")
    km = cfg["kmeans"]
    t0 = time.perf_counter()
    rows = []
    for ds, common_path in cfg["datasets"].items():
        common = load_yaml(ROOT / common_path)
        d = np.load(ROOT / common["dataset"]["processed_file"], allow_pickle=False)
        names = [str(c) for c in d["class_names"]]
        x, yn, yc, ids = d["X_train"], d["y_train_noisy"], d["y_train_clean"], d["id_train"]
        xv, yvn = d["X_val"], d["y_val_noisy"]
        benign = names.index(next(n for n in ("BENIGN", "Benign") if n in names))
        labels = {"stage1": yc}
        if ds == "cicids2018":
            dk = np.load(ROOT / "data/processed/cicids2018_labelnoise_attK.npz", allow_pickle=False)
            assert (dk["id_train"] == ids).all() and (dk["y_train_noisy"] == yn).all()
            labels["K"] = dk["y_train_clean"]
        groupings = {"exact": pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()}
        for k in (64, 256, 1024):
            groupings[f"kmeans_{k}"] = MiniBatchKMeans(n_clusters=k, random_state=km["random_state"],
                                                       batch_size=km["batch_size"], n_init=km["n_init"]).fit_predict(x)
        groupings["time_60min"] = time_groups(ds, common, ids)
        print(f"[{ds}] groupings ready ({time.perf_counter() - t0:.0f}s)", flush=True)
        for gname, g in groupings.items():
            probs = oof_probs(x, yn, g, xv, yvn, params, len(names), 2026)
            try:
                issues = find_label_issues(yn, probs, filter_by="prune_by_noise_rate")
            except Exception as e:                                        # noqa: BLE001
                print(f"  cleanlab failed for {gname}: {e}", flush=True)
                issues = np.zeros(len(yn), bool)
            np.save(out / f"d3_issues_{ds}_{gname}.npy", issues)
            q = ref[(ref["dataset"] == ds) & (ref["grouping"] == gname)]
            for conv, ycc in labels.items():
                hidden = (yn == benign) & (ycc != benign)
                affected = yn != ycc
                r = {"dataset": ds, "grouping": gname, "convention": conv, "flagged": int(issues.sum()),
                     "precision_either_direction": float((issues & affected).sum() / max(issues.sum(), 1)),
                     "precision_omitted": float((issues & hidden).sum() / max(issues.sum(), 1)),
                     "recall_omitted": float(issues[hidden].mean()), "omitted_rows": int(hidden.sum())}
                if conv == "stage1" and len(q) == 1:
                    r["saved_flagged"] = int(q["flagged"].iloc[0])
                    r["saved_precision"] = float(q["precision"].iloc[0])
                    r["saved_recall_hidden"] = float(q["recall_hidden"].iloc[0])
                    r["reproduced"] = bool(r["saved_flagged"] == r["flagged"]
                                           and abs(r["saved_precision"] - r["precision_either_direction"]) < 1e-6
                                           and abs(r["saved_recall_hidden"] - r["recall_omitted"]) < 1e-6)
                rows.append(r)
                print(f"  {gname} [{conv}]: flagged {r['flagged']:,} precision {r['precision_either_direction']:.4f} "
                      f"omission-precision {r['precision_omitted']:.4f} recall {r['recall_omitted']:.4f} "
                      f"reproduced {r.get('reproduced', '-')} ({time.perf_counter() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out / "d3_grouped_cv.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
