#!/usr/bin/env python
"""A-D2 post-hoc (NOT pre-registered): does the Stage-I direction-A failure survive
when the "clean" labels are restricted to their most reliable part?

A-D2 showed that most hidden rows other than the Infiltration port scans carry a
tier-2 (propagated) Liu label, and that the tier-1 hidden Bot rows are all
"Attempted" flows (the dataset authors recommend treating those as benign).
Re-evaluates the saved Stage-I predictions (no training) on
  R1: tier-1 test rows only (exact flow match with Liu's data)
  R2: R1 with Liu "Attempted" flows relabelled BENIGN (authors' recommendation)
Output: results/directionA_d2/label_robustness.csv (+ report section in DIRECTION_A_D2.md)
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import MODEL_ORDER, load_yaml  # noqa: E402


def main() -> None:
    ca = load_yaml(ROOT / "configs/directionA_common.yaml")
    d = np.load(ROOT / ca["dataset"]["processed_file"], allow_pickle=False)
    names = [str(c) for c in d["class_names"]]
    benign = names.index("BENIGN")
    mask = pd.read_parquet(ca["dataset"]["noise_mask"])
    keep = (mask["tier_liu"].to_numpy() > 0) & (mask["base_orig"].to_numpy() != "UNLABELED")
    mk = mask.loc[keep].reset_index(drop=True)
    sys.path.insert(0, str(ROOT / "alignment_cicids2017"))
    diff = import_module("10_label_diff")
    liu = diff.load_version(str(Path(ca["dataset"]["original_parquet_dir"]).parent / "liu"), diff.LIU_DAY_FILES, "liu")
    key = ["day", "Flow ID", "dur", "occ"]
    att_all = mk[key].merge(liu[key + ["att_liu"]], on=key, how="left")["att_liu"].astype(float).to_numpy()

    ids = d["id_test"]
    tier = d["tier_test"]
    att = att_all[ids]
    yc, yn = d["y_test_clean"].copy(), d["y_test_noisy"]
    t1 = tier == 1
    yc_r2 = yc.copy()
    yc_r2[t1 & (att == 1) & (yc != benign)] = benign        # Attempted -> BENIGN (authors' advice)

    variants = {"all covered rows (Stage I)": (np.ones(len(yc), bool), yc),
                "R1 tier-1 only": (t1, yc),
                "R2 tier-1, Attempted -> BENIGN": (t1, yc_r2)}
    rows = []
    for cond in ("noisy", "clean"):
        for m in MODEL_ORDER:
            for p in sorted((ROOT / "directionA" / cond / "results/raw" / m).glob("seed*/test_predictions.parquet")):
                pred = pd.read_parquet(p, columns=["pred_idx"])["pred_idx"].to_numpy()
                for vname, (sel, y) in variants.items():
                    aff = sel & (y != yn) & (y != benign) & (yn == benign)   # hidden attacks
                    att_rows = sel & (y != benign)
                    rows.append({"variant": vname, "training_labels": cond, "model": m, "seed": int(p.parent.name[4:]),
                                 "rows": int(sel.sum()), "attack_rows": int(att_rows.sum()),
                                 "hidden_attack_rows": int(aff.sum()),
                                 "hidden_share_of_attacks": float(aff.sum() / max(att_rows.sum(), 1)),
                                 "recovery_of_hidden": float((pred[aff] == y[aff]).mean()) if aff.any() else np.nan,
                                 "attacks_missed_as_benign": float((pred[att_rows] == benign).mean()),
                                 "benign_false_alarm": float((pred[sel & (y == benign)] != benign).mean())})
    df = pd.DataFrame(rows)
    out = ROOT / "results/directionA_d2"
    df.to_csv(out / "label_robustness.csv", index=False, float_format="%.6f")
    g = df.groupby(["variant", "training_labels"], sort=False)[
        ["rows", "attack_rows", "hidden_attack_rows", "hidden_share_of_attacks", "recovery_of_hidden",
         "attacks_missed_as_benign", "benign_false_alarm"]].agg(["mean", "min", "max"])
    g.to_csv(out / "label_robustness_summary.csv", float_format="%.6f")
    # hidden rows by clean class under each variant (counts only)
    by_class = []
    for vname, (sel, y) in variants.items():
        aff = sel & (y != yn) & (y != benign) & (yn == benign)
        vc = pd.Series([names[i] for i in y[aff]]).value_counts()
        by_class.append({"variant": vname, **vc.to_dict()})
    pd.DataFrame(by_class).fillna(0).to_csv(out / "label_robustness_hidden_by_class.csv", index=False)
    print(g.round(4).to_string())
    print(pd.DataFrame(by_class).fillna(0).to_string(index=False))
    print("tier-1 test rows:", int(t1.sum()), "of", len(t1), "| attempted among tier-1 attack rows:",
          float(np.nanmean(att[t1 & (yc != benign)])))


if __name__ == "__main__":
    main()
