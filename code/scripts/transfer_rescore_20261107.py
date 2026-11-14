#!/usr/bin/env python
"""LABELGAP transfer revision, 2026-11-07 (descriptive, no training).

Re-scores the saved test predictions of the controlled comparison under four test-label references (original,
omission fixed only, every other disagreement fixed only, corrected), on all test flows and on exactly matched
flows, with macro-F1 over the classes present in each reference, macro-F1 over the classes common to both label
versions, binary attack metrics and the F1 of the infiltration class.

Plan: configs/transfer_rescore_20261107.yaml (committed before this script is run).
Outputs: results/transfer_rescore_20261107/*, reports/TRANSFER_RESCORE_20261107.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.utils import MODEL_ORDER, git_commit, load_yaml, save_json  # noqa: E402

REFS = ["original", "omission_fixed", "other_fixed", "corrected"]
METRICS = ["macro_f1", "macro_f1_common", "attack_recall", "attack_precision", "attack_f1", "infiltration_f1"]


def binary(ref: np.ndarray, pred: np.ndarray, ben: int) -> tuple[float, float, float]:
    a_ref, a_pred = ref != ben, pred != ben
    tp = int((a_ref & a_pred).sum())
    fn = int((a_ref & ~a_pred).sum())
    fp = int((~a_ref & a_pred).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return rec, prec, (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)


def load_preds(spec: dict, ids: np.ndarray, ds: str) -> dict:
    preds = {}
    for cond in ("original", "corrected"):
        for m in MODEL_ORDER:
            files = sorted((ROOT / spec[cond] / "results/raw" / m).glob("seed*/test_predictions.parquet"))
            assert len(files) == 3, (ds, cond, m, files)
            for p in files:
                q = pd.read_parquet(p, columns=["sample_id", "pred_idx"])
                assert (q["sample_id"].to_numpy() == ids).all(), p
                preds[(cond, m, int(p.parent.name[4:]))] = q["pred_idx"].to_numpy()
    return preds


def sanity(ds: str, spec: dict, per_run: pd.DataFrame) -> float:
    """Max abs difference between the recomputed and the saved macro-F1 (subset all, original/corrected refs)."""
    s = spec["sanity"]
    saved = pd.read_csv(ROOT / s["file"])
    mine = per_run[(per_run["dataset"] == ds) & (per_run["subset"] == "all")
                   & per_run["reference"].isin(["original", "corrected"])]
    worst = 0.0
    for _, r in mine.iterrows():
        if "conv" in s:
            cond = "noisy" if r["trained_on"] == "original" else "clean"
            q = saved[(saved["conv"] == s["conv"]) & (saved["cond"] == cond) & (saved["model"] == r["model"])
                      & (saved["seed"] == r["seed"])]
        else:
            method = "CE, original labels" if r["trained_on"] == "original" else "CE, corrected labels"
            q = saved[(saved["dataset"] == s["dataset"]) & (saved["method"] == method)
                      & (saved["model"] == r["model"]) & (saved["seed"] == r["seed"])]
        assert len(q) == 1, (ds, r["trained_on"], r["model"], r["seed"], len(q))
        col = "macro_f1_original" if r["reference"] == "original" else "macro_f1_corrected"
        worst = max(worst, abs(float(q[col].iloc[0]) - float(r["macro_f1"])))
    return worst


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}" if np.isfinite(x) else "n/a"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main() -> None:
    t0 = time.perf_counter()
    plan = load_yaml(ROOT / "configs/transfer_rescore_20261107.yaml")
    out = ROOT / plan["outputs"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    rows, support_rows, sanity_res = [], [], {}
    for ds, spec in plan["comparisons"].items():
        d = np.load(ROOT / spec["npz"], allow_pickle=False)
        names = [str(c) for c in d["class_names"]]
        ben = names.index(next(n for n in ("BENIGN", "Benign") if n in names))
        inf = names.index(next(n for n in ("Infiltration", "Infilteration") if n in names))
        yn, yc, ids, tier = d["y_test_noisy"], d["y_test_clean"], d["id_test"], d["tier_test"]
        omitted = (yn == ben) & (yc != ben)
        refs = {"original": yn, "omission_fixed": np.where(omitted, yc, yn),
                "other_fixed": np.where(omitted, ben, yc), "corrected": yc}
        assert ((refs["omission_fixed"] != yn) == omitted).all()
        assert ((refs["other_fixed"] != yc) == omitted).all()
        subsets = {"all": np.ones(len(yn), bool), "tier1": tier == 1}
        if ds == "cicids2017":
            subsets["tier2"] = tier == 2
        preds = load_preds(spec, ids, ds)
        for sub, mask in subsets.items():
            common = [c for c in range(len(names)) if (yn[mask] == c).any() and (yc[mask] == c).any()]
            for ref_name in REFS:
                r = refs[ref_name][mask]
                present = [c for c in range(len(names)) if (r == c).any()]
                for c in present:
                    support_rows.append({"dataset": ds, "subset": sub, "reference": ref_name, "class": names[c],
                                         "support": int((r == c).sum()), "in_common_set": c in common})
                for (cond, m, seed), pred in preds.items():
                    p = pred[mask]
                    rec, prec, bf1 = binary(r, p, ben)
                    rows.append({"dataset": ds, "subset": sub, "reference": ref_name, "trained_on": cond,
                                 "model": m, "seed": seed, "n": int(mask.sum()),
                                 "n_omitted": int(omitted[mask].sum()),
                                 "n_disagree": int((yn[mask] != yc[mask]).sum()),
                                 "n_classes": len(present), "n_classes_common": len(common),
                                 "macro_f1": f1_score(r, p, labels=present, average="macro", zero_division=0),
                                 "macro_f1_common": f1_score(r, p, labels=common, average="macro", zero_division=0),
                                 "attack_recall": rec, "attack_precision": prec, "attack_f1": bf1,
                                 "infiltration_f1": f1_score(r == inf, p == inf, zero_division=0),
                                 "infiltration_support": int((r == inf).sum())})
        per_ds = pd.DataFrame([x for x in rows if x["dataset"] == ds])
        sanity_res[ds] = sanity(ds, spec, per_ds)
        assert sanity_res[ds] < 1e-6, (ds, sanity_res[ds])
        print(f"[{ds}] done, sanity max diff {sanity_res[ds]:.2e} ({time.perf_counter() - t0:.0f}s)", flush=True)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False, float_format="%.6f")
    pd.DataFrame(support_rows).to_csv(out / "class_support.csv", index=False)

    # paired differences h^c - h^o per model and seed
    key = ["dataset", "subset", "reference", "model", "seed"]
    wide = per_run.pivot_table(index=key, columns="trained_on", values=METRICS)
    deltas = pd.DataFrame({m: wide[(m, "corrected")] - wide[(m, "original")] for m in METRICS}).reset_index()
    deltas.to_csv(out / "deltas_per_pair.csv", index=False, float_format="%.6f")
    dmean = deltas.groupby(["dataset", "subset", "reference", "model"])[METRICS].mean().reset_index()
    dmean.to_csv(out / "deltas_model_mean.csv", index=False, float_format="%.6f")
    summ = per_run.groupby(["dataset", "subset", "reference", "trained_on", "model"])[METRICS].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(out / "summary_by_model.csv", index=False, float_format="%.6f")

    def mean_delta(ds, sub, ref, metric="macro_f1"):
        g = dmean[(dmean["dataset"] == ds) & (dmean["subset"] == sub) & (dmean["reference"] == ref)]
        return g.set_index("model")[metric].reindex(MODEL_ORDER)

    def pair_reversals(ds, sub, metric="macro_f1"):
        g = deltas[(deltas["dataset"] == ds) & (deltas["subset"] == sub)]
        o = g[g["reference"] == "original"].set_index(["model", "seed"])[metric]
        c = g[g["reference"] == "corrected"].set_index(["model", "seed"])[metric]
        return int(((o < 0) & (c > 0)).sum()), int(len(o))

    verdicts: dict = {"sanity_max_abs_diff": sanity_res}
    # decision 1: tier-1 reversal on CIC-IDS2017
    o1, c1 = mean_delta("cicids2017", "tier1", "original"), mean_delta("cicids2017", "tier1", "corrected")
    n_rev, n_pairs = pair_reversals("cicids2017", "tier1")
    verdicts["tier1_reversal"] = {"holds": bool(((o1 < 0) & (c1 > 0)).all()),
                                  "models_reversing": int(((o1 < 0) & (c1 > 0)).sum()),
                                  "pairs_reversing": n_rev, "pairs": n_pairs}
    # decision 2: omission attribution on CIC-IDS2017
    attr = {}
    for sub in ("all", "tier1"):
        om, ot = mean_delta("cicids2017", sub, "omission_fixed"), mean_delta("cicids2017", sub, "other_fixed")
        attr[sub] = {"hc_ahead_under_omission_fixed": int((om > 0).sum()),
                     "ho_ahead_under_other_fixed": int((ot < 0).sum()),
                     "holds": bool(((om > 0) & (ot < 0)).all())}
    verdicts["omission_attribution"] = {"by_subset": attr,
                                        "verdict": "omission" if all(v["holds"] for v in attr.values()) else "joint"}
    # descriptive: pairs reversing on every dataset and subset, both macro-F1 definitions
    rev = {}
    for (ds, sub), _ in deltas.groupby(["dataset", "subset"]):
        rev[f"{ds}|{sub}"] = {}
        for metric in ("macro_f1", "macro_f1_common"):
            n, tot = pair_reversals(ds, sub, metric)
            rev[f"{ds}|{sub}"][metric] = {"pairs_reversing": n, "pairs": tot}
            for ref in REFS:
                md = mean_delta(ds, sub, ref, metric)
                rev[f"{ds}|{sub}"][metric][f"models_ho_ahead_{ref}"] = int((md < 0).sum())
    verdicts["reversals"] = rev
    save_json(verdicts, out / "verdicts.json")

    # report
    lines = ["# LABELGAP Transfer revision: Saved predictions re-scored (2026-11-07)", "",
             f">Generation time{time.strftime('%Y-%m-%d %H:%M')};Code commit `{git_commit()[:7]}`."
             "The interpretation standard `configs/transfer_rescore_20261107.yaml` is submitted before running. Only the saved test predictions are recalculated, not training."
             "All previous results of this project are post hoc analyses.", "",
             "## Consistency check", "",
             "The maximum absolute difference between the original/corrected reference, Macro-F1 on all test traffic and the saved run-by-run results:"
             + ",".join(f"{k} {v:.1e}" for k, v in sanity_res.items()) + ".", "",
             "## Predetermined two interpretations (CIC-IDS2017)", "",
             f"- Tier 1 inversion:{'established' if verdicts['tier1_reversal']['holds'] else 'Not established'}"
             f"(Mean by seed{verdicts['tier1_reversal']['models_reversing']}/5 model inversion;"
             f"pair by pair{n_rev}/{n_pairs}).",
             f"- Attribution of missing bids:{verdicts['omission_attribution']['verdict']}"
             f"(All test traffic and Tier 1, the number of leading models in corrected training when only missing labels are corrected, and the number of leading models in original training when only other mislabelings are corrected:"
             + ";".join(f"{s} {v['hc_ahead_under_omission_fixed']}/5、{v['ho_ahead_under_other_fixed']}/5"
                         for s, v in attr.items()) + ").", "",
             "## Macro-F1 difference under each reference (corrected training minus original training, mean by seed)", ""]
    for metric, title in (("macro_f1", "Macro-F1 (category appearing in reference)"), ("macro_f1_common", "Macro-F1 (common category for both versions)"),
                          ("attack_f1", "Two-class attack F1"), ("infiltration_f1", "Penetration class F1")):
        lines += [f"### {title}", ""]
        t = dmean.pivot_table(index=["dataset", "subset", "model"], columns="reference", values=metric)[REFS]
        t = t.reset_index()
        t["model"] = pd.Categorical(t["model"], MODEL_ORDER)
        t = t.sort_values(["dataset", "subset", "model"])
        for ref in REFS:
            t[ref] = t[ref].map(fmt)
        lines += [md_table(t), ""]
    lines += ["## Pair-by-pair reversal counting (original training leads under the original reference and corrected training leads under the revised reference)", ""]
    rr = [{"dataset|subset": k, "Macro-F1": f"{v['macro_f1']['pairs_reversing']}/{v['macro_f1']['pairs']}",
           "Macro-F1 common category": f"{v['macro_f1_common']['pairs_reversing']}/{v['macro_f1_common']['pairs']}"}
          for k, v in rev.items()]
    lines += [md_table(pd.DataFrame(rr)), "",
              "## Number of test samples", ""]
    ns = per_run.groupby(["dataset", "subset"])[["n", "n_omitted", "n_disagree", "n_classes_common"]].first()
    lines += [md_table(ns.reset_index()), "",
              "For the number of samples per class, see `class_support.csv`; for the run-by-run indicator `per_run.csv`; for the pair-wise difference `deltas_per_pair.csv`;"
              "Interpret `verdicts.json`."]
    rep = ROOT / plan["outputs"]["report"]
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {time.perf_counter() - t0:.0f}s; report {rep}")
    print(verdicts["tier1_reversal"], verdicts["omission_attribution"])


if __name__ == "__main__":
    main()
