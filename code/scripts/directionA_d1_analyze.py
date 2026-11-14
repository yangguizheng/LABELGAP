#!/usr/bin/env python
"""A-D1 analysis: do off-the-shelf noise-robust methods recover the systematically
hidden attacks without clean labels? Rules: configs/direction_a_d1_noise_robust.yaml.

Defaults reproduce the CICIDS2017 analysis. For CSE-CIC-IDS2018:
  --common configs/directionA2018_common.yaml --runs directionA2018 --out results/directionA2018_d1
  --report reports/DIRECTION_A2018_D1.md --interp reports/direction_a2018_d1_interpretation.md
  --focus Infilteration Benign --primary hidden
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from build_failure_report import md_table  # noqa: E402
from src.utils import MODEL_DISPLAY, git_commit, load_yaml  # noqa: E402

METHODS = [  # (label, sub-directory under --runs, is_reference)
    ("CE on noisy labels (Stage I)", "noisy", True),
    ("GCE", "d1_gce", False),
    ("Co-teaching", "d1_coteaching", False),
    ("Confident Learning", "d1_cl", False),
    ("Clean-label oracle (Stage I)", "clean", True),
]
FOCUS_2017 = ["PortScan", "DDoS", "FTP-Patator", "Bot", "Web Attack - Brute Force"]


def pm(v: pd.Series) -> str:
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", default="configs/directionA_common.yaml")
    ap.add_argument("--runs", default="directionA")
    ap.add_argument("--out", default="results/directionA_d1")
    ap.add_argument("--report", default="reports/DIRECTION_A_D1.md")
    ap.add_argument("--interp", default="reports/direction_a_d1_interpretation.md")
    ap.add_argument("--focus", nargs="+", default=FOCUS_2017)
    ap.add_argument("--primary", choices=["all", "hidden"], default="all",
                    help="rows used for the verdict: all affected rows (2017) or hidden attacks only (2018)")
    ap.add_argument("--title", default="CICIDS2017")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / "configs/direction_a_d1_noise_robust.yaml")
    common = load_yaml(ROOT / args.common)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / common["dataset"]["processed_file"], allow_pickle=False)
    names = [str(c) for c in d["class_names"]]
    yc, yn = d["y_test_clean"], d["y_test_noisy"]
    aff = yc != yn
    benign = names.index(next(n for n in ("BENIGN", "Benign") if n in names))
    hidden = (yn == benign) & (yc != benign)
    labels_present = [c for c in range(len(names)) if (yc == c).any()]
    rec_col = "recovery" if args.primary == "all" else "recovery_hidden"

    rows = []
    for label, sub, ref in METHODS:
        for m in ("xgboost", "mlp", "transformer"):
            for s in cfg["seeds"]:
                p = ROOT / args.runs / sub / "results/raw" / m / f"seed{s}" / "test_predictions.parquet"
                if not p.exists():
                    continue
                pred = pd.read_parquet(p, columns=["pred_idx"])["pred_idx"].to_numpy()
                r = {"method": label, "reference": ref, "model": m, "seed": s,
                     "recovery": float((pred[aff] == yc[aff]).mean()),
                     "recovery_hidden": float((pred[hidden] == yc[hidden]).mean()),
                     "affected_pred_benign": float((pred[aff] == benign).mean()),
                     "hidden_pred_benign": float((pred[hidden] == benign).mean()),
                     "unaffected_acc": float((pred[~aff] == yc[~aff]).mean()),
                     "benign_false_alarm": float((pred[yc == benign] != benign).mean()),
                     "attacks_missed_as_benign": float((pred[yc != benign] == benign).mean()),
                     "macro_f1_clean": f1_score(yc, pred, labels=labels_present, average="macro", zero_division=0)}
                for c in args.focus:
                    k = names.index(c)
                    r[f"recall_{c}"] = float((pred[yc == k] == k).mean())
                    hid = aff & (yc == k)
                    r[f"recovery_{c}"] = float((pred[hid] == k).mean()) if hid.any() else np.nan
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "recovery_per_run.csv", index=False, float_format="%.6f")
    num = [c for c in df.columns if c not in ("method", "reference", "model", "seed")]
    summ = df.groupby(["method", "model"], sort=False)[num].mean()
    summ.to_csv(out / "recovery_summary.csv", float_format="%.6f")

    v = cfg["verdict"]
    verdicts = []
    for (label, m), r in summ.iterrows():
        if label.startswith(("CE on noisy", "Clean-label")):
            continue
        rec = v["recovered"]
        if r[rec_col] >= rec["min_recovery"] and r["unaffected_acc"] >= rec["min_unaffected_accuracy"] \
                and r["benign_false_alarm"] <= rec["max_benign_false_alarm"]:
            verdict = "recovered"
        elif r[rec_col] >= v["partial"]["min_recovery"]:
            verdict = "partial"
        else:
            verdict = "fails"
        verdicts.append({"method": label, "model": m, "recovery": r[rec_col],
                         "unaffected_acc": r["unaffected_acc"], "benign_false_alarm": r["benign_false_alarm"],
                         "verdict": verdict})
    verdicts = pd.DataFrame(verdicts)
    verdicts.to_csv(out / "verdicts.csv", index=False, float_format="%.6f")
    if (verdicts["verdict"] == "recovered").any():
        direction = v["direction"]["any_recovered"]
    elif (verdicts["verdict"] == "partial").any():
        direction = v["direction"]["any_partial"]
    else:
        direction = v["direction"]["all_fail"]

    # ---- report ------------------------------------------------------------------------------------
    L = []
    w = L.append
    w(f"# DIRECTION A — Diagnostics A-D1 ({args.title}): Can off-the-shelf anti-noise methods recover missed attacks when there are no clean labels? \n")
    w(f">Generation time{time.strftime('%Y-%m-%d %H:%M')};Code commit `{git_commit()[:7]}`. Design and interpretation standards"
      "`configs/direction_a_d1_noise_robust.yaml` (2018 see also `configs/direction_a2018_d1.yaml`) is committed before running."
      "Only the original (with errors) labels are used for training and verification; the corrected labels are only used for test evaluation and scoring of Confident Learning’s error checking results."
      "In the first stage, the CE (with wrong label) of the same seed and the upper limit of the corrected label are used as a comparison without retraining. \n")
    interp = ROOT / args.interp
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    w(f"## D1-a Recovery and collateral effects (test set, mean ± standard deviation over {len(cfg['seeds'])} seeds)\n")
    w(f"Recovery is the share of mislabeled test flows ({int(aff.sum()):,}) predicted as their corrected class. "
      f"Omitted attacks are flows labeled benign in the original release but attack in the corrected release ({int(hidden.sum()):,} flows). "
      f"**Primary criterion: {'all mislabeled flows' if args.primary == 'all' else 'omitted attacks'}**.\n")
    t = []
    for (label, m), g in df.groupby(["method", "model"], sort=False):
        row = {"Method": label, "Model": MODEL_DISPLAY[m], "Recovery (all affected)": pm(g["recovery"])}
        if args.primary == "hidden":
            row["Recovery (hidden attacks)"] = pm(g["recovery_hidden"])
            row["Hidden → Benign"] = pm(g["hidden_pred_benign"])
        else:
            row["Affected → BENIGN"] = pm(g["affected_pred_benign"])
        row.update({"Unaffected accuracy": pm(g["unaffected_acc"]), "Benign false alarm": pm(g["benign_false_alarm"]),
                    "Attacks missed as benign": pm(g["attacks_missed_as_benign"]),
                    "Macro-F1 (clean test)": pm(g["macro_f1_clean"])})
        t.append(row)
    w(md_table(pd.DataFrame(t)))
    w("\n## D1-b Interpretation (pre-registration) \n")
    rr = v["recovered"]
    w(f"recovered: recovery rate ≥{rr['min_recovery']}, and the accuracy of not mismarking rows is ≥{rr['min_unaffected_accuracy']}, normal traffic false alarm rate ≤"
      f"{rr['max_benign_false_alarm']};partial: retrieval rate ≥{v['partial']['min_recovery']};The rest fails."
      f"**Direction determination:{direction}**.\n")
    vv = verdicts.copy()
    vv["model"] = vv["model"].map(MODEL_DISPLAY)
    w(md_table(vv.rename(columns={"method": "Method", "model": "Model", "recovery": "Recovery (verdict rows)",
                                  "unaffected_acc": "Unaffected acc.", "benign_false_alarm": "Benign FAR",
                                  "verdict": "Verdict"})))
    w("\n## D1-c Retrieval rate by category (proportion of mislabeled rows predicted to be true categories, seed mean) \n")
    t = pd.DataFrame({"Method": [a for a, _ in summ.index], "Model": [MODEL_DISPLAY[b] for _, b in summ.index]})
    for c in args.focus:
        t[c] = summ[f"recovery_{c}"].to_numpy()
    w(md_table(t))
    cl = out / "cl_detection.csv"
    if cl.exists():
        w("\n## D1-d Error checking quality of Confident Learning (training set; correction labels are only used for scoring) \n")
        c = pd.read_csv(cl)
        base = ["seed", "flagged", "true_affected", "true_positives", "precision", "recall"]
        w(md_table(c[base].rename(columns={"seed": "Seed", "flagged": "Flagged rows", "true_affected": "True affected",
                                           "true_positives": "Correctly flagged", "precision": "Precision",
                                           "recall": "Recall"})))
        hid = [k for k in c.columns if k.startswith("recall_hidden_")]
        w("\n*The proportion of various types of wrongly marked rows marked by CL (by correction category, mean seed value):*" +
          ";".join(f"{k.replace('recall_hidden_', '')} {c[k].mean():.1%}" for k in hid))
        ff = [k for k in c.columns if k.startswith("false_flags_labelled_")]
        w("\n* Mislabeled rows (the original labels are correct) are distributed according to the original labels (seed mean): *" +
          ";".join(f"{k.replace('false_flags_labelled_', '')} {c[k].mean():,.0f}" for k in ff
                   if c[k].mean() >= 1))
    mon = []
    for label, sub, ref in METHODS:
        if ref:
            continue
        for m in ("mlp", "transformer"):
            for s in cfg["seeds"]:
                h = ROOT / args.runs / sub / "logs" / m / f"seed{s}_history.csv"
                if not h.exists():
                    continue
                hist = pd.read_csv(h)
                if "val_recovery_clean" not in hist:
                    continue
                i = int(hist["val_recovery_clean"].idxmax())
                mon.append({"method": label, "model": m, "seed": s,
                            "max_val_recovery": float(hist["val_recovery_clean"].max()),
                            "epoch_of_max": int(hist.loc[i, "epoch"]),
                            "last_val_recovery": float(hist["val_recovery_clean"].iloc[-1])})
    if mon:
        mon = pd.DataFrame(mon)
        mon.to_csv(out / "monitoring_val_recovery.csv", index=False, float_format="%.6f")
        w("\n## D1-e Retrieval rate of the validation set during training (only observation, not used for model selection) \n")
        w("In each round, the correction labels are used to calculate the retrieval rate of mislabeled rows (in both directions) in the validation set; \"highest value\" ≥ 0.5 is recorded as the presence of early learning signals. \n")
        t = []
        for (label, m), g in mon.groupby(["method", "model"], sort=False):
            t.append({"Method": label, "Model": MODEL_DISPLAY[m],
                      "Max over epochs": pm(g["max_val_recovery"]),
                      "Epoch of max": ", ".join(str(e) for e in g["epoch_of_max"]),
                      "Last epoch": pm(g["last_val_recovery"]),
                      "Early-learning signal": bool((g["max_val_recovery"] >= 0.5).mean() >= 0.5)})
        w(md_table(pd.DataFrame(t)))
    w("\n## file \n")
    w(f"`{args.out}/`:recovery_per_run.csv、recovery_summary.csv、verdicts.csv、cl_detection.csv、"
      f"cl_issues_seed*.npy. Running result: `{args.runs}/d1_{{gce,coteaching,cl}}/`. Code: `src/training/noise_robust.py`,"
      "`scripts/directionA_d1_cl.py`、`scripts/directionA_d1_analyze.py`.")
    (ROOT / args.report).write_text("\n".join(L) + "\n")
    print(pd.DataFrame(verdicts).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("direction verdict:", direction)


if __name__ == "__main__":
    main()
