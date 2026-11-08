#!/usr/bin/env python
"""Recompute Table 5, the reversal counts of Section 5.2, and Table 14 from derived_data/ alone.

Usage: python code/verify_from_predictions.py   (run from the package root)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DD = Path(__file__).resolve().parents[1] / "derived_data"
MODELS = ["xgboost", "mlp", "cnn", "bigru", "transformer"]
SEEDS = ["seed42", "seed52", "seed62"]


def load(name):
    t = pd.read_csv(DD / f"test_predictions_{name}.csv.gz")
    names = json.loads((DD / f"class_names_{name}.json").read_text())
    ben = names.index("BENIGN" if "BENIGN" in names else "Benign")
    return t, ben


def mf1(ref, pred):
    present = np.unique(ref)
    return f1_score(ref, pred, labels=present, average="macro", zero_division=0)


def compare(t, ben, y_orig, y_corr, h_o, h_c, mask=None, title=""):
    mask = np.ones(len(t), bool) if mask is None else mask
    yo, yc = t[y_orig].to_numpy()[mask], t[y_corr].to_numpy()[mask]
    om = (yo == ben) & (yc != ben)
    refs = {"original": yo, "omission repaired": np.where(om, yc, yo),
            "other errors repaired": np.where(om, ben, yc), "corrected": yc}
    print(f"\n== {title}  ({mask.sum():,} test flows, {om.sum():,} omitted)")
    rows, rev = [], 0
    for m in MODELS:
        r = {"model": m}
        for s in SEEDS:
            po, pc = t[f"{h_o}__{m}__{s}"].to_numpy()[mask], t[f"{h_c}__{m}__{s}"].to_numpy()[mask]
            d_o = mf1(refs["original"], pc) - mf1(refs["original"], po)
            d_c = mf1(refs["corrected"], pc) - mf1(refs["corrected"], po)
            rev += int(d_o < 0 < d_c)
        for k, ref in refs.items():
            vals_o = [mf1(ref, t[f"{h_o}__{m}__{s}"].to_numpy()[mask]) for s in SEEDS]
            vals_c = [mf1(ref, t[f"{h_c}__{m}__{s}"].to_numpy()[mask]) for s in SEEDS]
            r[k] = np.mean(vals_c) - np.mean(vals_o)
        r["recovery h_o"] = np.mean([(t[f"{h_o}__{m}__{s}"].to_numpy()[mask][om] == yc[om]).mean() for s in SEEDS])
        r["recovery h_c"] = np.mean([(t[f"{h_c}__{m}__{s}"].to_numpy()[mask][om] == yc[om]).mean() for s in SEEDS])
        r["MF1 corrected h_o"] = np.mean([mf1(yc, t[f"{h_o}__{m}__{s}"].to_numpy()[mask]) for s in SEEDS])
        r["MF1 corrected h_c"] = np.mean([mf1(yc, t[f"{h_c}__{m}__{s}"].to_numpy()[mask]) for s in SEEDS])
        r["MF1 original h_o"] = np.mean([mf1(yo, t[f"{h_o}__{m}__{s}"].to_numpy()[mask]) for s in SEEDS])
        r["MF1 original h_c"] = np.mean([mf1(yo, t[f"{h_c}__{m}__{s}"].to_numpy()[mask]) for s in SEEDS])
        rows.append(r)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(rows).set_index("model").to_string())
    print(f"model-seed pairs reversed between the original and the corrected test labels: {rev}/15")


t, ben = load("cicids2017")
compare(t, ben, "y_original", "y_corrected", "ce_original", "ce_corrected", title="CIC-IDS2017, all aligned test flows")
compare(t, ben, "y_original", "y_corrected", "ce_original", "ce_corrected", mask=t["tier"].to_numpy() == 1,
        title="CIC-IDS2017, exact-match (tier 1) test flows")
t, ben = load("cicids2018")
compare(t, ben, "y_original", "y_corrected_attempted_attack_class", "ce_original", "ce_corrected_attempted_attack_class",
        title="CSE-CIC-IDS2018, attempted flows in their attack class")
compare(t, ben, "y_original", "y_corrected_attempted_benign", "ce_original", "ce_corrected_attempted_benign",
        title="CSE-CIC-IDS2018, attempted flows benign")
t, ben = load("cicids2018_attempted_removed")
compare(t, ben, "y_original", "y_corrected", "ce_original", "ce_corrected",
        title="CSE-CIC-IDS2018, attempted flows removed")
