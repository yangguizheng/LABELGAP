#!/usr/bin/env python
"""A-D2: mechanism diagnostic for the systematically hidden attacks (no training).
Feature-space position (vs correctly labelled attacks of the same class and vs
benign traffic), self-clustering, time/host context and "Attempted" share.
Readings: configs/direction_a_d2_mechanism.yaml. Outputs: results/directionA_d2/*,
reports/DIRECTION_A_D2.md (interpretation from reports/direction_a_d2_interpretation.md).
"""
from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from build_failure_report import md_table  # noqa: E402
from directionA_prepare import load_original  # noqa: E402
from src.analysis.diagnostics import knn_label_counts  # noqa: E402
from src.utils import git_commit, load_yaml, save_json  # noqa: E402


def nn_dist(q: np.ndarray, ref: np.ndarray, self_idx: np.ndarray | None = None,
            chunk: int = 256) -> np.ndarray:
    """Nearest-neighbour Euclidean distance (float64, GPU); self_idx[i] >= 0 excludes
    that exact reference row for query i."""
    r = torch.from_numpy(ref).to("cuda", torch.float64)
    rn = (r * r).sum(1)
    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        x = torch.from_numpy(q[s:s + chunk]).to("cuda", torch.float64)
        d2 = ((x * x).sum(1, keepdim=True) + rn[None, :] - 2.0 * x @ r.T).clamp_(min=0)
        if self_idx is not None:
            ex = torch.from_numpy(self_idx[s:s + chunk]).to("cuda")
            rows = torch.nonzero(ex >= 0).squeeze(1)
            d2[rows, ex[rows]] = float("inf")
        out[s:s + chunk] = d2.min(1).values.sqrt().cpu().numpy()
    return out


def densest_window_share(ts: pd.Series, minutes: int) -> float:
    t = np.sort(ts.dropna().to_numpy().astype("datetime64[s]").astype(np.int64))
    if len(t) == 0:
        return float("nan")
    j = np.searchsorted(t, t + minutes * 60, side="left")
    return float((j - np.arange(len(t))).max() / len(t))


def main() -> None:
    cfg = load_yaml(ROOT / "configs/direction_a_d2_mechanism.yaml")
    ca = load_yaml(ROOT / "configs/directionA_common.yaml")
    out = ROOT / "results/directionA_d2"
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / ca["dataset"]["processed_file"], allow_pickle=False)
    names = [str(c) for c in d["class_names"]]
    C, benign = len(names), names.index("BENIGN")
    xt, ytn, ytc = d["X_train"], d["y_train_noisy"], d["y_train_clean"]
    hid_t = (ytn == benign) & (ytc != benign)
    lab_t = (ytn == ytc)
    k = cfg["feature_space"]["neighbourhood_k"]
    rng = np.random.default_rng(2026)

    # category per training row: clean class for correctly labelled rows, C = hidden
    cat = np.where(hid_t, C, ytc)
    cat[(~hid_t) & (~lab_t)] = C + 1                       # other disagreements (rare): own bin
    classes = [c for c in range(C) if c != benign and (hid_t & (ytc == c)).sum() >= 100]
    t0 = time.perf_counter()
    feat_rows, per_row = [], []
    idx_all = np.arange(len(ytc))
    ben_ref = np.where((ytc == benign) & lab_t)[0]
    hid_ref = np.where(hid_t)[0]
    for c in classes:
        q_hid = np.where(hid_t & (ytc == c))[0]
        lab_c = np.where(lab_t & (ytc == c))[0]
        q_base = np.sort(rng.choice(lab_c, size=min(cfg["rows"]["baseline_queries_per_class"], len(lab_c)),
                                    replace=False))
        res = {}
        for tag, q in (("hidden", q_hid), ("labelled", q_base)):
            pos_in_lab = np.full(len(q), -1)
            if tag == "labelled":
                pos_in_lab = np.searchsorted(lab_c, q)
            pos_in_hid = np.full(len(q), -1)
            if tag == "hidden":
                pos_in_hid = np.searchsorted(hid_ref, q)
            d_lab = nn_dist(xt[q], xt[lab_c], pos_in_lab)
            d_ben = nn_dist(xt[q], xt[ben_ref])
            d_hid = nn_dist(xt[q], xt[hid_ref], pos_in_hid)
            counts = knn_label_counts(xt[q], xt, cat, [k], C + 2, chunk=256)[0][k]
            self_cat = C if tag == "hidden" else c
            counts[np.arange(len(q)), self_cat] -= 1           # drop the query itself
            tot = counts.sum(1)
            res[tag] = pd.DataFrame({"row": q, "d_labelled_c": d_lab, "d_benign": d_ben, "d_hidden": d_hid,
                                     "ratio": d_lab / np.maximum(d_ben, 1e-12),
                                     "share_labelled_c": counts[:, c] / tot,
                                     "share_benign": counts[:, benign] / tot,
                                     "share_hidden": counts[:, C] / tot})
        h, b = res["hidden"], res["labelled"]
        feat_rows.append({
            "class": names[c], "hidden_train": len(q_hid), "labelled_train": len(lab_c),
            "hidden_median_d_labelled_c": h["d_labelled_c"].median(), "hidden_median_d_benign": h["d_benign"].median(),
            "hidden_median_d_hidden": h["d_hidden"].median(), "hidden_median_ratio": h["ratio"].median(),
            "hidden_share_ratio_lt_1": float((h["ratio"] < 1).mean()),
            "labelled_median_ratio": b["ratio"].median(),
            "hidden_median_share_labelled_c": h["share_labelled_c"].median(),
            "hidden_median_share_benign": h["share_benign"].median(),
            "hidden_median_share_hidden": h["share_hidden"].median(),
            "hidden_mean_share_labelled_c": h["share_labelled_c"].mean(),
            "hidden_mean_share_benign": h["share_benign"].mean(),
            "hidden_mean_share_hidden": h["share_hidden"].mean(),
            "labelled_mean_share_labelled_c": b["share_labelled_c"].mean(),
            "proximity_signal": bool(h["ratio"].median() < 1),
            "self_cluster": bool(h["share_hidden"].median() >= 0.5)})
        per_row.append(h.assign(clean_class=names[c]))
        print(f"[ok] feature space {names[c]}: {len(q_hid)} hidden rows ({time.perf_counter() - t0:.0f}s)", flush=True)
    feat = pd.DataFrame(feat_rows)
    feat.to_csv(out / "feature_space.csv", index=False, float_format="%.6f")
    pd.concat(per_row).to_parquet(out / "hidden_rows_feature_space.parquet", index=False)

    # ---- context: all covered rows (train + val + test) ----------------------------------------
    orig = load_original(ca)
    mask = pd.read_parquet(ca["dataset"]["noise_mask"])
    keep = (mask["tier_liu"].to_numpy() > 0) & (mask["base_orig"].to_numpy() != "UNLABELED")
    meta = orig.loc[keep, ["day", "Source IP", "Destination IP", "Destination Port", "Timestamp"]].reset_index(drop=True)
    mk = mask.loc[keep].reset_index(drop=True)
    y_c, y_n = np.empty(len(meta), int), np.empty(len(meta), int)
    for s in ("train", "val", "test"):
        y_c[d[f"id_{s}"]], y_n[d[f"id_{s}"]] = d[f"y_{s}_clean"], d[f"y_{s}_noisy"]
    ts = pd.to_datetime(meta["Timestamp"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    ts = ts.where(ts.dt.hour >= 8, ts + pd.Timedelta(hours=12))   # 12-h clock without AM/PM
    parsed = float(ts.notna().mean())

    # "Attempted" flag for tier-1 matches (Liu labels), via the nids_audit loaders
    sys.path.insert(0, ca["dataset"]["mask_code_dir"])
    diff = import_module("10_label_diff")
    liu = diff.load_version(str(Path(ca["dataset"]["original_parquet_dir"]).parent / "liu"), diff.LIU_DAY_FILES, "liu")
    key = ["day", "Flow ID", "dur", "occ"]
    att = mk[key].merge(liu[key + ["att_liu"]], on=key, how="left")["att_liu"].astype(float)
    att = att.where(mk["tier_liu"].to_numpy() == 1)

    ctx_rows = []
    bins = meta["day"].astype(str) + "|" + ts.dt.floor("60min").astype(str)
    for c in classes:
        hid = (y_n == benign) & (y_c == c)
        lab = (y_n == c) & (y_c == c)
        src = meta.loc[hid, "Source IP"].value_counts(normalize=True)
        dst = meta.loc[hid, "Destination IP"].value_counts(normalize=True)
        lab_bins = set(bins[lab])
        a = att[hid].dropna()
        ctx_rows.append({
            "class": names[c], "hidden_all": int(hid.sum()),
            "days": ", ".join(f"{k2} {v:.0%}" for k2, v in meta.loc[hid, "day"].value_counts(normalize=True).items()),
            "top3_src_share": float(src.head(3).sum()), "top_src": ", ".join(src.head(3).index.astype(str)),
            "top3_dst_share": float(dst.head(3).sum()), "top_dst": ", ".join(dst.head(3).index.astype(str)),
            "densest_60min_share": densest_window_share(ts[hid], cfg["context"]["densest_window_minutes"]),
            "hidden_time_span": f"{ts[hid].min():%a %H:%M}–{ts[hid].max():%a %H:%M}",
            "labelled_time_span": f"{ts[lab].min():%a %H:%M}–{ts[lab].max():%a %H:%M}" if lab.any() else "—",
            "co_occurrence": float(bins[hid].isin(lab_bins).mean()),
            "attempted_share_tier1": float(a.mean()) if len(a) else float("nan"), "tier1_rows": int(len(a))})
    ctx = pd.DataFrame(ctx_rows)
    r = cfg["readings"]
    ctx["host_concentration"] = (ctx["top3_src_share"] >= 0.8) | (ctx["top3_dst_share"] >= 0.8)
    ctx["time_concentration"] = ctx["densest_60min_share"] >= 0.8
    ctx["within_campaign"] = ctx["co_occurrence"] >= 0.5
    ctx.to_csv(out / "context.csv", index=False, float_format="%.6f")
    save_json({"timestamp_parse_rate": parsed, "classes": [names[c] for c in classes], "k": k}, out / "summary.json")

    # ---- report ---------------------------------------------------------------------------------
    L = []
    w = L.append
    w("# DIRECTION A - Diagnosis A-D2: Where are the systematically missed attacks (feature space and time/host context) \n")
    w(f">Researcher selected on 2026-09-25. Generation time{time.strftime('%Y-%m-%d %H:%M')};Code commit `{git_commit()[:7]}`."
      "The interpretation standard `configs/direction_a_d2_mechanism.yaml` is submitted before calculation. It only performs calculations and does not train the model. \n")
    interp = ROOT / "reports/direction_a_d2_interpretation.md"
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    w("## D2-a Feature space position (training set, preprocessed 77-dimensional feature, Euclidean distance) \n")
    w("r = the closest distance to a correctly labeled attack of the same type / the closest distance to real normal traffic; r < 1 means closer to a similar attack."
      f"The nearest neighbors are{k}The proportion of each type in the most recent training samples (excluding itself). \"Labeled Attack (Control)\" is a random sample of correctly labeled rows of the same type. \n")
    t = feat[["class", "hidden_train", "hidden_median_ratio", "hidden_share_ratio_lt_1", "labelled_median_ratio",
              "hidden_median_d_labelled_c", "hidden_median_d_benign", "hidden_median_d_hidden",
              "hidden_mean_share_labelled_c", "hidden_mean_share_benign", "hidden_mean_share_hidden",
              "proximity_signal", "self_cluster"]].copy()
    t.columns = ["Class", "Hidden rows (train)", "Median r (hidden)", "Share r<1 (hidden)", "Median r (labelled, ref.)",
                 "Median d to labelled c", "Median d to benign", "Median d to other hidden",
                 "Nbr share: labelled c", "Nbr share: benign", "Nbr share: hidden",
                 "Proximity signal", "Self-cluster"]
    w(md_table(t))
    w("\n## D2-b Time and host context (all overlay lines) \n")
    w(f"Timestamp parsing success rate{parsed:.1%}(Original file is in 12 hour format with no AM/PM, hours before 8 o'clock plus 12)."
      "co-occurrence = the proportion of marked attacks of the same type that exist in the (week, 60-minute period) where the missing row is located."
      "Attempted = Proportion labeled \"attempted\" by Liu et al. (only tier-1 rows with exact matches). \n")
    t = ctx[["class", "hidden_all", "days", "hidden_time_span", "labelled_time_span", "top3_src_share", "top_src",
             "top3_dst_share", "densest_60min_share", "co_occurrence", "attempted_share_tier1",
             "host_concentration", "time_concentration", "within_campaign"]].copy()
    t.columns = ["Class", "Hidden rows (all)", "Days", "Hidden time span", "Labelled time span",
                 "Top-3 src share", "Top src IPs", "Top-3 dst share", "Densest 60-min share", "Co-occurrence",
                 "Attempted (tier-1)", "Host conc.", "Time conc.", "Within campaign"]
    w(md_table(t))
    rb = out / "label_robustness_summary.csv"
    if rb.exists():
        w("\n## D2-c [Post-analysis, non-pre-registration] Reliability of revised labels and robustness of first-stage conclusions \n")
        w("D2-b shows: Except for the Infiltration port scan, the correction labels of the missed rows are almost all from tier-2 (inferred by (week, flow ID)),"
          "All the missed Bot rows in tier-1 were marked as \"attempted\" by Liu. Therefore the saved predictions from the first stage (without retraining) are used in the two more reliable"
          "Recalculate on the subset: R1 = only look at tier-1 (exact match) test rows; R2 = R1 and change the \"attempted\" traffic to BENIGN as recommended by the data set author."
          "The value is the mean of 5 models × 3 seeds (`scripts/directionA_d2_label_robustness.py`). \n")
        s = pd.read_csv(rb, header=[0, 1], index_col=[0, 1])
        t = []
        for (variant, cond), r in s.iterrows():
            t.append({"Evaluation subset": variant, "Training labels": cond,
                      "Test rows": int(r[("rows", "mean")]), "Attack rows": int(r[("attack_rows", "mean")]),
                      "Hidden attacks": int(r[("hidden_attack_rows", "mean")]),
                      "Hidden share of attacks": r[("hidden_share_of_attacks", "mean")],
                      "Recovery of hidden": r[("recovery_of_hidden", "mean")],
                      "Attacks missed as BENIGN": r[("attacks_missed_as_benign", "mean")],
                      "BENIGN false alarm": r[("benign_false_alarm", "mean")]})
        w(md_table(pd.DataFrame(t)))
        bc = pd.read_csv(out / "label_robustness_hidden_by_class.csv")
        w("\n*The number of missed attacks by category in each subset (test set):*\n")
        w(md_table(bc.rename(columns={"variant": "Evaluation subset"}), nd=0))
    w("\n## file \n")
    w("`results/directionA_d2/`: feature_space.csv, hidden_rows_feature_space.parquet (consisting of row-by-row distance and nearest neighbor),"
      "context.csv, label_robustness*.csv, summary.json. Code: `scripts/directionA_d2_mechanism.py`,"
      "`scripts/directionA_d2_label_robustness.py`.")
    (ROOT / "reports/DIRECTION_A_D2.md").write_text("\n".join(L) + "\n")
    print(feat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(ctx.drop(columns=["top_src", "top_dst"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"timestamp parse rate {parsed:.3f}")


if __name__ == "__main__":
    main()
