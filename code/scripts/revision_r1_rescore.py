#!/usr/bin/env python
"""LABELGAP revision, round 1 (descriptive): re-scores the saved test predictions of the controlled comparison and
of the noise-robust runs, per seed, for CIC-IDS2017, CSE-CIC-IDS2018 under convention K (primary) and under the
Stage-I convention; scores the saved Confident Learning training flags against the training labels of each
convention; writes the data behind Fig. 1 and Fig. 2.
Plan: configs/revision_round1.yaml (committed before this script is run).
Outputs: results/revision_round1/rescore_*.csv, rescore_summary.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.utils import MODEL_ORDER, load_yaml, save_json  # noqa: E402

SETS = {
    "cicids2017": {"npz": "data/processed/cicids2017_labelnoise_v1.npz", "cl": "results/directionA_d1",
                   "runs": {"CE, original labels": "directionA/noisy", "GCE": "directionA/d1_gce",
                            "Co-teaching": "directionA/d1_coteaching", "Confident Learning": "directionA/d1_cl",
                            "CE, corrected labels": "directionA/clean"}},
    "cicids2018_K": {"npz": "data/processed/cicids2018_labelnoise_attK.npz", "cl": "results/directionA2018_d1",
                     "runs": {"CE, original labels": "directionA2018/noisy", "GCE": "directionA2018/d1_gce",
                              "Co-teaching": "directionA2018/d1_coteaching",
                              "Confident Learning": "directionA2018/d1_cl",
                              "CE, corrected labels": "directionA2018_attK/clean"}},
    "cicids2018_stage1": {"npz": "data/processed/cicids2018_labelnoise_v1.npz", "cl": "results/directionA2018_d1",
                          "runs": {"CE, original labels": "directionA2018/noisy", "GCE": "directionA2018/d1_gce",
                                   "Co-teaching": "directionA2018/d1_coteaching",
                                   "Confident Learning": "directionA2018/d1_cl",
                                   "CE, corrected labels": "directionA2018/clean"}},
}


def main() -> None:
    plan = load_yaml(ROOT / "configs/revision_round1.yaml")
    out = ROOT / plan["outputs"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    rows, cl_rows = [], []
    for ds, spec in SETS.items():
        d = np.load(ROOT / spec["npz"], allow_pickle=False)
        names = [str(c) for c in d["class_names"]]
        ben = names.index(next(n for n in ("BENIGN", "Benign") if n in names))
        yc, yn, ids = d["y_test_clean"], d["y_test_noisy"], d["id_test"]
        lab_c = [c for c in range(len(names)) if (yc == c).any()]
        lab_n = [c for c in range(len(names)) if (yn == c).any()]
        omitted = (yn == ben) & (yc != ben)
        agree = yc == yn
        for method, runs in spec["runs"].items():
            for m in MODEL_ORDER:
                for p in sorted((ROOT / runs / "results/raw" / m).glob("seed*/test_predictions.parquet")):
                    q = pd.read_parquet(p, columns=["sample_id", "pred_idx"])
                    assert (q["sample_id"].to_numpy() == ids).all(), p
                    pred = q["pred_idx"].to_numpy()
                    rows.append({"dataset": ds, "method": method, "model": m, "seed": int(p.parent.name[4:]),
                                 "recovery": float((pred[omitted] == yc[omitted]).mean()),
                                 "macro_f1_corrected": f1_score(yc, pred, labels=lab_c, average="macro",
                                                                zero_division=0),
                                 "macro_f1_original": f1_score(yn, pred, labels=lab_n, average="macro",
                                                               zero_division=0),
                                 "false_alarm": float((pred[yc == ben] != ben).mean()),
                                 "attacks_as_benign": float((pred[yc != ben] == ben).mean()),
                                 "accuracy_agreeing": float((pred[agree] == yc[agree]).mean()),
                                 "n_omitted_test": int(omitted.sum())})
        # Confident Learning training flags (one set per seed, XGBoost out-of-fold probabilities)
        ytr_c, ytr_n = d["y_train_clean"], d["y_train_noisy"]
        mis = ytr_c != ytr_n
        om_tr = (ytr_n == ben) & (ytr_c != ben)
        for f in sorted((ROOT / spec["cl"]).glob("cl_issues_seed*.npy")):
            iss = np.load(f)
            assert iss.shape == ytr_c.shape, f
            cl_rows.append({"dataset": ds, "seed": int(f.stem.split("seed")[1]), "flagged": int(iss.sum()),
                            "mislabeled_training_flows": int(mis.sum()),
                            "precision": float((iss & mis).sum() / max(iss.sum(), 1)),
                            "recall_mislabeled": float(iss[mis].mean()),
                            "recall_omitted": float(iss[om_tr].mean()), "omitted_training_flows": int(om_tr.sum())})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "rescore_per_run.csv", index=False, float_format="%.6f")
    counts = per_run.groupby(["dataset", "method", "model"]).size()
    assert (counts == 3).all(), counts[counts != 3]
    summ = per_run.groupby(["dataset", "method", "model"]).agg(["mean", "std"]).drop(columns="seed", level=0)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(out / "rescore_summary.csv", index=False, float_format="%.6f")
    cl = pd.DataFrame(cl_rows)
    cl.to_csv(out / "rescore_cl_flags.csv", index=False, float_format="%.6f")
    # figure data
    fig1 = per_run[per_run["method"].isin(["CE, original labels", "CE, corrected labels"])].groupby(
        ["dataset", "method", "model"])[["macro_f1_corrected", "macro_f1_original"]].mean().reset_index()
    fig1.to_csv(out / "fig1_data.csv", index=False, float_format="%.6f")
    fig2 = per_run[per_run["dataset"].isin(["cicids2017", "cicids2018_K"])].groupby(
        ["dataset", "method", "model"])["recovery"].agg(["mean", "std"]).reset_index()
    fig2.to_csv(out / "fig2_data.csv", index=False, float_format="%.6f")
    rng = {}
    for (ds, method), g in summ.reset_index().groupby(["dataset", "method"]):
        rng[f"{ds} | {method}"] = {k: [float(g[f"{k}_mean"].min()), float(g[f"{k}_mean"].max())]
                                   for k in ("recovery", "macro_f1_corrected", "macro_f1_original", "false_alarm",
                                             "attacks_as_benign", "accuracy_agreeing")}
    clg = cl.groupby("dataset")[["precision", "recall_mislabeled", "recall_omitted"]].agg(["min", "max"]).round(4)
    clg.columns = [f"{a}_{b}" for a, b in clg.columns]
    save_json({"ranges_over_models": rng, "cl_flags": clg.to_dict(orient="index")}, out / "rescore_summary.json")
    pd.set_option("display.width", 250)
    print(summ.reset_index()[["dataset", "method", "model", "recovery_mean", "macro_f1_corrected_mean",
                              "macro_f1_original_mean", "false_alarm_mean"]].to_string(index=False))
    print(cl.to_string(index=False))


if __name__ == "__main__":
    main()
