#!/usr/bin/env python
"""Direction A, Steps 14-20: failure analysis of training under natural label
noise (CICIDS2017 original vs Liu-corrected labels) against the clean-label
oracle. Rules: configs/direction_a_label_noise.yaml (committed before training).
Outputs: results/directionA/*.csv, reports/DIRECTION_A_STAGE1.md
(interpretation spliced from reports/direction_a_interpretation.md).
"""
from __future__ import annotations

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
from src.utils import MODEL_DISPLAY, MODEL_ORDER, git_commit, load_json, load_yaml  # noqa: E402

CONDS = ("noisy", "clean")


def pm(m: float, s: float, nd: int = 3) -> str:
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", default="configs/directionA_common.yaml")
    ap.add_argument("--runs", default="directionA", help="root holding <cond>/results/raw")
    ap.add_argument("--results", default="results/directionA")
    ap.add_argument("--report", default="reports/DIRECTION_A_STAGE1.md")
    ap.add_argument("--interp", default="reports/direction_a_interpretation.md")
    ap.add_argument("--title", default="CICIDS2017 original label vs Liu's revised label")
    args = ap.parse_args()
    common = load_yaml(ROOT / args.common)
    rules = load_yaml(ROOT / "configs/direction_a_label_noise.yaml")
    r = rules["rules"]
    kmin, min_sup = rules["min_models"], rules["min_test_support"]
    rec_ok = rules["headroom"]["clean_trained_recovers_at_least"]
    out = ROOT / args.results
    d = np.load(ROOT / common["dataset"]["processed_file"], allow_pickle=False)
    names = [str(c) for c in d["class_names"]]
    C = len(names)
    yc, yn = d["y_test_clean"], d["y_test_noisy"]
    affected = yc != yn
    benign = names.index(next(n for n in ("BENIGN", "Benign") if n in names))
    lab_c = [c for c in range(C) if (yc == c).any()]   # macro averages over classes present
    lab_n = [c for c in range(C) if (yn == c).any()]   # (all of them for CICIDS2017)
    prof = pd.read_csv(out / "data_profile.csv")
    limited = set(prof.loc[prof["separability_limited"], "class"])

    runs = []
    for cond in CONDS:
        for m in MODEL_ORDER:
            for p in sorted((ROOT / args.runs / cond / "results/raw" / m).glob("seed*/test_predictions.parquet")):
                pred = pd.read_parquet(p, columns=["pred_idx"])["pred_idx"].to_numpy()
                runs.append({"cond": cond, "model": m, "seed": int(p.parent.name[4:]), "pred": pred})
    models = [m for m in MODEL_ORDER if any(x["model"] == m for x in runs)]
    seeds = sorted({x["seed"] for x in runs})

    # ---- per-run metrics ---------------------------------------------------------------------------
    rows, pcr = [], []
    for x in runs:
        pred = x["pred"]
        mf1_clean = f1_score(yc, pred, labels=lab_c, average="macro", zero_division=0)
        mf1_noisy = f1_score(yn, pred, labels=lab_n, average="macro", zero_division=0)
        att = yc != benign
        rows.append({"cond": x["cond"], "model": x["model"], "seed": x["seed"],
                     "acc_clean": float((pred == yc).mean()), "acc_noisy": float((pred == yn).mean()),
                     "macro_f1_clean": mf1_clean, "macro_f1_noisy": mf1_noisy,
                     "attack_missed_as_benign_clean": float((pred[att] == benign).mean()),
                     "benign_false_alarm_clean": float((pred[~att] != benign).mean()),
                     "affected_correct": float((pred[affected] == yc[affected]).mean()),
                     "affected_pred_benign": float((pred[affected] == benign).mean()),
                     "affected_detected_as_attack": float((pred[affected] != benign).mean()),
                     "unaffected_acc": float((pred[~affected] == yc[~affected]).mean())})
        for c, name in enumerate(names):
            sel = yc == c
            aff = sel & affected
            pcr.append({"cond": x["cond"], "model": x["model"], "seed": x["seed"], "class": name,
                        "n_test": int(sel.sum()), "n_affected": int(aff.sum()),
                        "recall": float((pred[sel] == c).mean()),
                        "recall_affected": float((pred[aff] == c).mean()) if aff.any() else np.nan,
                        "recall_unaffected": float((pred[sel & ~affected] == c).mean())
                        if (sel & ~affected).any() else np.nan})
    ov, pc = pd.DataFrame(rows), pd.DataFrame(pcr)
    ov.to_csv(out / "overall_per_run.csv", index=False, float_format="%.6f")
    pc.to_csv(out / "per_class_recall_per_run.csv", index=False, float_format="%.6f")

    # ---- A1: noise-induced class failure (paired by seed) --------------------------------------------
    a1 = []
    for name in names:
        n_test = int((yc == names.index(name)).sum())
        row = {"class": name, "n_test": n_test, "n_affected": int(((yc == names.index(name)) & affected).sum()),
               "separability_limited": name in limited}
        failing, recovering = [], []
        for m in models:
            q = (pc[(pc["class"] == name) & (pc["model"] == m)]
                 .pivot(index="seed", columns="cond", values="recall").reindex(columns=list(CONDS)))
            drop = (q["clean"] - q["noisy"]).dropna()
            row[f"{m}_noisy"], row[f"{m}_clean"], row[f"{m}_drop"] = q["noisy"].mean(), q["clean"].mean(), drop.mean()
            if n_test >= min_sup and drop.mean() >= r["A1_noise_induced_class_failure"]["min_recall_drop"] \
                    and (drop > 0).all():
                failing.append(m)
                if q["clean"].mean() >= rec_ok:
                    recovering.append(m)
        row["models_failing"] = ", ".join(MODEL_DISPLAY[m] for m in failing)
        row["flagged"] = len(failing) >= kmin
        row["oracle_recovers"] = len(recovering) >= kmin
        row["method_addressable"] = bool(row["flagged"] and row["oracle_recovers"] and not row["separability_limited"])
        a1.append(row)
    a1 = pd.DataFrame(a1)
    a1.to_csv(out / "A1_class_failures.csv", index=False, float_format="%.6f")

    # ---- A2: hidden-attack failure on affected rows, per clean class ---------------------------------
    a2 = []
    for name in ["ALL"] + names:
        if name == "ALL":
            sub = ov.groupby(["cond", "model"])["affected_correct"].mean()
            n_aff = int(affected.sum())
        else:
            n_aff = int(((yc == names.index(name)) & affected).sum())
            if n_aff < min_sup:
                continue
            sub = pc[pc["class"] == name].groupby(["cond", "model"])["recall_affected"].mean()
        row = {"class": name, "n_affected_test": n_aff}
        for m in models:
            row[f"{m}_noisy"], row[f"{m}_clean"] = sub.get(("noisy", m), np.nan), sub.get(("clean", m), np.nan)
        n_fail = sum(row[f"{m}_noisy"] <= r["A2_hidden_attack_failure"]["noisy_trained_at_most"] for m in models)
        n_rec = sum(row[f"{m}_clean"] >= rec_ok for m in models)
        row.update({"models_failing": int(n_fail), "models_oracle_recovers": int(n_rec),
                    "flagged": n_fail >= kmin, "oracle_recovers": n_rec >= kmin,
                    "method_addressable": bool(n_fail >= kmin and n_rec >= kmin and name not in limited)})
        a2.append(row)
    a2 = pd.DataFrame(a2)
    a2.to_csv(out / "A2_hidden_attack_failures.csv", index=False, float_format="%.6f")

    # ---- A3: evaluation masking -------------------------------------------------------------------------
    g = ov.groupby(["cond", "model"])[["macro_f1_clean", "macro_f1_noisy", "acc_clean", "acc_noisy"]].mean()
    over = (g.loc["noisy", "macro_f1_noisy"] - g.loc["noisy", "macro_f1_clean"])
    a3_flag = int((over >= r["A3_evaluation_masking"]["min_overstatement"]).sum()) >= kmin

    # ---- report -------------------------------------------------------------------------------------------
    L = []
    w = L.append
    w(f"# DIRECTION A — Stage I: The consequences of training with wrong labels ({args.title})\n")
    w(f">Researcher 2026-09-25 selected (parallel with direction B). Generation time{time.strftime('%Y-%m-%d %H:%M')};Code commit"
      f"`{git_commit()[:7]}`. Interpretation rules `configs/direction_a_label_noise.yaml` is submitted before training. The same batch of rows, the same division,"
      f"Only change training/validation labels: noisy = original label, clean = Liu modified label (upper limit control). Tests are always evaluated according to the revision label."
      f"{len(models)}models ×{len(seeds)}seeds × 2 conditions. \n")
    interp = ROOT / args.interp
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    s = load_json(out / "data_summary.json")
    aff_test_n = int(affected.sum())
    w("## A-0 Data and Step 4 Data Physical Examination \n")
    w(f"overwrite row{s['rows']:,}(train{s['train']:,}/ verify{s['val']:,}/ test{s['test']:,}),{s['n_features']}characteristics;"
      f"Lines with inconsistent labels between the two versions{s['affected_rows_total']:,}({s['global_noise_rate']:.2%}), where the test set"
      f"{s['affected_rows_test']:,}. Among them, the proportion of \"attacks marked as normal by the original version\""
      f"{((yn == benign) & (yc != benign)).sum() / max(aff_test_n, 1):.1%}(test set), see label_transitions.csv for other directions. \n")
    p = prof.copy()
    p.loc[p["class"] == names[benign], "hidden_as_benign"] = np.nan
    w(md_table(p[["class", "rows_clean", "rows_noisy", "hidden_as_benign", "noise_rate_clean_class", "test_clean",
                  "test_affected", "train_conflict_rate_clean_labels", "separability_limited"]].rename(columns={
        "class": "Class (clean)", "rows_clean": "Rows (clean)", "rows_noisy": "Rows (noisy)",
        "hidden_as_benign": "Hidden as BENIGN", "noise_rate_clean_class": "Noise rate",
        "test_clean": "Test rows", "test_affected": "Test affected",
        "train_conflict_rate_clean_labels": "Train conflict rate (clean)",
        "separability_limited": "Separability-limited"})))
    w("\n## A-1 Overall results (test set evaluated by revised label; 3 seeds mean ± std) \n")
    tab = []
    for cond in CONDS:
        for m in models:
            q = ov[(ov["cond"] == cond) & (ov["model"] == m)]
            tab.append({"Training labels": cond, "Model": MODEL_DISPLAY[m],
                        "Macro-F1 (clean test)": pm(q["macro_f1_clean"].mean(), q["macro_f1_clean"].std(ddof=1)),
                        "Macro-F1 (noisy test)": pm(q["macro_f1_noisy"].mean(), q["macro_f1_noisy"].std(ddof=1)),
                        "Accuracy (clean test)": pm(q["acc_clean"].mean(), q["acc_clean"].std(ddof=1)),
                        "Attacks missed as BENIGN": pm(q["attack_missed_as_benign_clean"].mean(),
                                                       q["attack_missed_as_benign_clean"].std(ddof=1)),
                        "Affected rows correct": pm(q["affected_correct"].mean(), q["affected_correct"].std(ddof=1)),
                        "Unaffected rows accuracy": pm(q["unaffected_acc"].mean(), q["unaffected_acc"].std(ddof=1))})
    w(md_table(pd.DataFrame(tab)))
    w(f"\n ## A-2　A1: Reduced category recall caused by incorrect labels (paired seeds; test row ≥{min_sup}category) \n")
    thr = r["A1_noise_induced_class_failure"]["min_recall_drop"]
    w(f"Rule: Recall (clean training) − Recall (noisy training) ≥{thr}, and the differences of all seeds are positive, at ≥{kmin}Mark the success in each model immediately;"
      f"Upper limit comparison: clean training recall ≥{rec_ok}.\n")
    cols = ["class", "n_test", "n_affected"] + [f"{m}_{k}" for m in models for k in ("noisy", "clean")] + \
           ["models_failing", "flagged", "oracle_recovers", "method_addressable"]
    show = a1[cols].copy()
    show.columns = ["Class", "Test n", "Affected n"] + [f"{MODEL_DISPLAY[m]} {k}" for m in models for k in ("noisy", "clean")] + \
                   ["Models failing", "Flagged", "Oracle recovers", "Method-addressable"]
    w(md_table(show))
    w(f"\n ## A-3　A2: On the wrongly labeled test row, can the model trained with the wrong label recognize the real category \n")
    w(f"Values ​​are the proportion of mislabeled rows that are predicted to be true (corrected) classes (average of 3 seeds). Rule: noisy training ≤"
      f"{r['A2_hidden_attack_failure']['noisy_trained_at_most']}model ≥{kmin}Each is the mark; upper limit comparison: clean training ≥{rec_ok}.\n")
    cols = ["class", "n_affected_test"] + [f"{m}_{k}" for m in models for k in ("noisy", "clean")] + \
           ["models_failing", "models_oracle_recovers", "flagged", "oracle_recovers", "method_addressable"]
    show = a2[cols].copy()
    show.columns = ["Class", "Affected test n"] + [f"{MODEL_DISPLAY[m]} {k}" for m in models for k in ("noisy", "clean")] + \
                   ["Models failing", "Models oracle ok", "Flagged", "Oracle recovers", "Method-addressable"]
    w(md_table(show))
    w(f"\n ## A-4　A3: Does the regular evaluation (according to the original test label) cover up the problem? \n")
    w(f"noisy training model: Macro-F1 (original test label) − Macro-F1 (corrected test label); ≥{r['A3_evaluation_masking']['min_overstatement']} "
      f"model ≥{kmin}One is the mark. **determination:{'mark' if a3_flag else 'Not marked'}**.\n")
    w(md_table(pd.DataFrame({"Model": [MODEL_DISPLAY[m] for m in models],
                             "Macro-F1 noisy test": [g.loc[("noisy", m), "macro_f1_noisy"] for m in models],
                             "Macro-F1 clean test": [g.loc[("noisy", m), "macro_f1_clean"] for m in models],
                             "Overstatement": [over[m] for m in models],
                             "Accuracy noisy test": [g.loc[("noisy", m), "acc_noisy"] for m in models],
                             "Accuracy clean test": [g.loc[("noisy", m), "acc_clean"] for m in models]})))
    w("\n## file \n")
    w(f"`{args.results}/`:data_profile.csv、data_summary.json、overall_per_run.csv、per_class_recall_per_run.csv、"
      f"A1_class_failures.csv, A2_hidden_attack_failures.csv. Running result: `{args.runs}/{{noisy,clean}}/`."
      "Code: `scripts/directionA_analyze.py` (for data preparation and running scripts, see the prepare/run script of the corresponding data set).")
    (ROOT / args.report).write_text("\n".join(L) + "\n")
    print(pd.DataFrame(tab).to_string(index=False))
    print(a1[["class", "n_test", "models_failing", "flagged", "oracle_recovers", "method_addressable"]].to_string(index=False))
    print(a2[["class", "n_affected_test", "models_failing", "models_oracle_recovers", "flagged", "method_addressable"]]
          .to_string(index=False))
    print("A3 overstatement:", over.round(4).to_dict(), "flag:", a3_flag)


if __name__ == "__main__":
    main()
