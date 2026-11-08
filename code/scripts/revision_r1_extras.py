#!/usr/bin/env python
"""LABELGAP revision, round 1 (descriptive, added with the amendment of 2026-09-28): the remaining numbers for the
text. (1) CSE-CIC-IDS2018 under convention K: recall of the infiltration class and recovery of the omitted test
flows on tier-A (unique-key) flows only, per model and training condition; (2) audit effort of the
exposure rule (60 min) and of the cleaned rule in analyst units: flagged hosts and host-hours, both datasets.
Outputs: results/revision_round1/extras_*.csv, extras_summary.json
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from directionA_d4_context import flag  # noqa: E402
from directionA_d5_evidence_cleaning import flag_exposed, host_trust  # noqa: E402
from directionA_prepare import load_original  # noqa: E402
from revision_r1_context import D5, frame2018  # noqa: E402
from src.utils import MODEL_ORDER, load_yaml, save_json  # noqa: E402

RUNS = {"original labels": "directionA2018/noisy", "corrected labels": "directionA2018_attK/clean"}


def units(fr: pd.DataFrame, flagged: np.ndarray) -> dict:
    f = fr.loc[flagged, ["day", "src", "t"]]
    return {"flagged_flows": int(flagged.sum()), "hosts": int(f[["day", "src"]].drop_duplicates().shape[0]),
            "host_hours": int(f.assign(h=f["t"].dt.floor("60min"))[["day", "src", "h"]].drop_duplicates().shape[0])}


def main() -> None:
    plan = load_yaml(ROOT / "configs/revision_round1.yaml")
    out = ROOT / plan["outputs"]["results"]
    S: dict = {}

    # ---- (1) learning results on CSE-CIC-IDS2018, convention K ---------------------------------------------
    d = np.load(ROOT / "data/processed/cicids2018_labelnoise_attK.npz", allow_pickle=False)
    names = [str(c) for c in d["class_names"]]
    ben, inf = names.index("Benign"), names.index("Infilteration")
    yc, yn, ids, tier = d["y_test_clean"], d["y_test_noisy"], d["id_test"], d["tier_test"]
    omitted = (yn == ben) & (yc != ben)
    rows = []
    for cond, path in RUNS.items():
        for m in MODEL_ORDER:
            for p in sorted((ROOT / path / "results/raw" / m).glob("seed*/test_predictions.parquet")):
                q = pd.read_parquet(p, columns=["sample_id", "pred_idx"])
                assert (q["sample_id"].to_numpy() == ids).all()
                pred = q["pred_idx"].to_numpy()
                rows.append({"training": cond, "model": m, "seed": int(p.parent.name[4:]),
                             "infiltration_recall": float((pred[yc == inf] == inf).mean()),
                             "recovery_all": float((pred[omitted] == yc[omitted]).mean()),
                             "recovery_tier_A": float((pred[omitted & (tier == 1)] == yc[omitted & (tier == 1)]).mean()),
                             "omitted_test": int(omitted.sum()), "omitted_test_tier_A": int((omitted & (tier == 1)).sum())})
    r = pd.DataFrame(rows)
    r.to_csv(out / "extras_2018_learning_per_run.csv", index=False, float_format="%.6f")
    g = r.groupby(["training", "model"])[["infiltration_recall", "recovery_all", "recovery_tier_A"]].mean()
    S["learning_2018_K"] = {f"{a} | {b}": v.round(4).to_dict() for (a, b), v in g.iterrows()}
    S["omitted_test"] = int(omitted.sum())
    S["omitted_test_tier_A"] = int((omitted & (tier == 1)).sum())
    print(g.to_string(float_format=lambda v: f"{v:.4f}"), flush=True)

    # ---- (2) audit effort in analyst units -------------------------------------------------------------------
    win, min_share = D5["window_minutes"], D5["min_surviving_share"]
    fr = frame2018()
    issues = np.load(ROOT / "results/directionA_d5/cl_issues_cicids2018.npy")
    cand = fr["noisy"].str.lower().eq("benign").to_numpy()
    attack = ~cand
    cidx = np.where(cand)[0]
    surviving = attack & ~issues
    S["cicids2018"] = {"exposure_60": units(fr, flag(fr, cidx, attack, "V2", win) & cand),
                       "cleaned": units(fr, flag_exposed(fr, cidx, surviving, host_trust(fr, attack, surviving),
                                                         win, min_share) & cand),
                       "candidate_hosts": int(fr.loc[cand, ["day", "src"]].drop_duplicates().shape[0])}
    del fr
    ca = load_yaml(ROOT / "configs/directionA_common.yaml")
    orig = load_original(ca)
    mask = pd.read_parquet(ca["dataset"]["noise_mask"])
    ts = pd.to_datetime(orig["Timestamp"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    ts = ts.where(ts.dt.hour >= 8, ts + pd.Timedelta(hours=12))
    fr17 = pd.DataFrame({"t": ts, "src": orig["Source IP"].astype(str).str.strip(),
                         "dst": orig["Destination IP"].astype(str).str.strip(), "day": orig["day"].to_numpy()})
    del orig
    noisy = mask["base_orig"].astype(str).to_numpy()
    tier17 = mask["tier_liu"].to_numpy()
    covered = (tier17 > 0) & (noisy != "UNLABELED")
    cand17 = covered & (noisy == "BENIGN")
    attack17 = (noisy != "UNLABELED") & (noisy != "BENIGN")
    issues17 = np.load(ROOT / "results/directionA_d5/cl_issues_cicids2017.npy")
    c17 = np.where(cand17)[0]
    surv17 = attack17 & ~issues17
    f_exp = flag(fr17, c17, attack17, "V2", win) & cand17
    f_cln = flag_exposed(fr17, c17, surv17, host_trust(fr17, attack17, surv17), win, min_share) & cand17
    t1 = tier17 == 1
    S["cicids2017"] = {"exposure_60_tier1": units(fr17, f_exp & t1), "cleaned_tier1": units(fr17, f_cln & t1),
                       "exposure_60_all_aligned": units(fr17, f_exp), "cleaned_all_aligned": units(fr17, f_cln),
                       "candidate_hosts_tier1": int(fr17.loc[cand17 & t1, ["day", "src"]].drop_duplicates().shape[0])}
    save_json(S, out / "extras_summary.json")
    print({k: v for k, v in S.items() if k != "learning_2018_K"})


if __name__ == "__main__":
    main()
