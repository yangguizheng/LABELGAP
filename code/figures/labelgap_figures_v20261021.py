#!/usr/bin/env python
"""Two descriptive figures added to the LABELGAP manuscript on 2026-10-21 (no training, no new runs).
Reads committed result files of nids_failure_discovery read-only:
  results/directionA/per_class_recall_per_run.csv  (controlled comparison, CIC-IDS2017, 5 models x 3 seeds)
  results/directionA_d2/feature_space.csv          (A-D2 feature-space diagnostic, CIC-IDS2017 training partition)
Style follows scripts/labelgap_figures.py. Output: ./out/fig_perclass.pdf, ./out/fig_neighbors.pdf (+ png, csv).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parent / "out"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
C_ORIG, C_CORR, C_HIDDEN, C_BENIGN = "#d95f02", "#1b9e77", "#e7298a", "#9e9e9e"
CLASSES = [("PortScan", "PortScan"), ("DDoS", "DDoS"), ("FTP-Patator", "FTP-Patator"),
           ("Web Attack - Brute Force", "Web brute\nforce"), ("DoS Hulk", "DoS Hulk"), ("Bot", "Bot"),
           ("Web Attack - XSS", "XSS")]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def fig_perclass():
    d = pd.read_csv(RES / "directionA/per_class_recall_per_run.csv")
    assert len(d.groupby(["model", "seed"])) == 15
    series = [("noisy", "recall_unaffected", C_ORIG, 0.45, "trained on original labels: flows the original labels mark as this attack"),
              ("noisy", "recall_affected", C_ORIG, 1.0, "trained on original labels: flows the original labels mark otherwise"),
              ("clean", "recall_affected", C_CORR, 1.0, "trained on corrected labels: flows the original labels mark otherwise")]
    rows = []
    fig, ax = plt.subplots(figsize=(3.45, 3.1))
    h = 0.26
    y0 = np.arange(len(CLASSES))[::-1]
    for i, (cls, _) in enumerate(CLASSES):
        for j, (cond, col, color, alpha, lab) in enumerate(series):
            sel = d[(d["cond"] == cond) & (d["class"] == cls)]
            v = sel[col].to_numpy()
            assert len(v) == 15, (cls, cond, len(v))
            m, lo, hi = v.mean(), v.min(), v.max()
            rows.append({"class": cls, "series": lab, "mean": m, "min": lo, "max": hi,
                         "n_test_marked_otherwise": int(sel["n_affected"].iloc[0]),
                         "n_test_marked_as_attack": int(sel["n_test"].iloc[0] - sel["n_affected"].iloc[0])})
            ax.barh(y0[i] + (1 - j) * h, m, h, color=color, alpha=alpha, edgecolor=color, lw=0.4,
                    xerr=[[max(m - lo, 0.0)], [max(hi - m, 0.0)]], capsize=1.2, error_kw={"lw": 0.5},
                    label=lab if i == 0 else None)
    n = {r["class"]: r["n_test_marked_otherwise"] for r in rows}
    ax.set_yticks(y0, [f"{lab.replace(chr(10), ' ')}\n{n[c]:,}" for c, lab in CLASSES], fontsize=6.5)
    ax.set_xlabel("Recall on test flows")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, frameon=False, fontsize=6.2)
    save(fig, "fig_perclass")
    pd.DataFrame(rows).to_csv(OUT / "fig_perclass_data.csv", index=False, float_format="%.6f")


def fig_neighbors():
    f = pd.read_csv(RES / "directionA_d2/feature_space.csv").set_index("class")
    names = [c for c, _ in CLASSES]
    labels = [lab.replace("\n", " ") for _, lab in CLASSES]
    f = f.loc[names]
    share = pd.DataFrame({"other omitted flows": f["hidden_mean_share_hidden"],
                          "benign flows": f["hidden_mean_share_benign"],
                          "labeled flows of the same class": f["hidden_mean_share_labelled_c"]})
    share["other flows"] = (1 - share.sum(axis=1)).clip(lower=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    left = np.zeros(len(names))
    colors = [C_HIDDEN, C_BENIGN, C_ORIG, "#d9d9d9"]
    y = np.arange(len(names))[::-1]
    for (col, color) in zip(share.columns, colors):
        ax.barh(y, share[col].to_numpy(), left=left, color=color, height=0.62, label=col)
        left += share[col].to_numpy()
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean share of the 10 nearest training flows")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.52), ncol=2, frameon=False, fontsize=6.5)
    ax = axes[1]
    pts = [("hidden_median_d_hidden", "other omitted flow", C_HIDDEN, "o"),
           ("hidden_median_d_benign", "benign flow", C_BENIGN, "s"),
           ("hidden_median_d_labelled_c", "labeled flow of the same class", C_ORIG, "^")]
    for col, lab, color, mk in pts:
        ax.scatter(f[col].to_numpy(), y, color=color, marker=mk, s=16, label=lab, zorder=3)
    for k in range(len(names)):
        vals = [f[c].iloc[k] for c, *_ in pts]
        ax.plot([min(vals), max(vals)], [y[k], y[k]], color="0.8", lw=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Median distance to the nearest flow")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.52), ncol=2, frameon=False, fontsize=6.5)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig_neighbors")
    out = share.copy()
    for col, lab, *_ in pts:
        out[f"median distance to nearest {lab}"] = f[col]
    out.to_csv(OUT / "fig_neighbors_data.csv", float_format="%.6f")


def fig_anatomy():
    """Omission anatomy on CIC-IDS2017: share of each corrected class marked benign, and omitted flows by tier."""
    prof = pd.read_csv(RES / "directionA/data_profile.csv").set_index("class")
    tiers = pd.read_csv(RES / "revision_round1/anatomy_2017_omitted_by_tier_class.csv")
    names = [c for c, _ in CLASSES]
    labels = [lab.replace(chr(10), " ") for _, lab in CLASSES]
    share = prof.loc[names, "hidden_as_benign"] / prof.loc[names, "rows_clean"] * 100
    assert np.allclose(share / 100, prof.loc[names, "noise_rate_clean_class"], atol=1e-5)
    t1 = tiers[tiers.tier == 1].set_index("clean")["flows"].reindex(names).fillna(0)
    t2 = tiers[tiers.tier == 2].set_index("clean")["flows"].reindex(names).fillna(0)
    assert ((t1 + t2) == prof.loc[names, "hidden_as_benign"]).all()
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
    ax = axes[0]
    ax.barh(y, share.to_numpy(), height=0.62, color=C_HIDDEN)
    for yi, v, n in zip(y, share.to_numpy(), prof.loc[names, "rows_clean"].to_numpy()):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=6.5)
    ax.set_xlim(0, 62)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Share of the corrected class marked benign (%)")
    ax = axes[1]
    h = 0.32
    ax.barh(y + h / 2, t1.replace(0, np.nan).to_numpy(), height=h, color="#4c72b0", label="tier 1, exact match")
    ax.barh(y - h / 2, t2.replace(0, np.nan).to_numpy(), height=h, color="#a6bddb", label="tier 2, same identifier")
    for yi, a, b in zip(y, t1.to_numpy(), t2.to_numpy()):
        if a > 0:
            ax.text(a * 1.15, yi + h / 2, f"{int(a):,}", va="center", fontsize=6)
        if b > 0:
            ax.text(b * 1.15, yi - h / 2, f"{int(b):,}", va="center", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlim(1, 1.2e6)
    ax.set_xlabel("Omitted flows, logarithmic scale")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.5)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig_anatomy")
    out = pd.DataFrame({"class": names, "aligned_flows_of_class": prof.loc[names, "rows_clean"].to_numpy(),
                        "omitted_flows": prof.loc[names, "hidden_as_benign"].to_numpy(), "omitted_share_percent": share.to_numpy(),
                        "omitted_tier1": t1.to_numpy(), "omitted_tier2": t2.to_numpy()})
    out.to_csv(OUT / "fig_anatomy_data.csv", index=False, float_format="%.4f")


def fig_gap():
    """CSE-CIC-IDS2018: per-class contributions to the Macro-F1 gap against the original test labels."""
    g = pd.read_csv(RES / "directionA2018_attempted/gap_decomposition.csv")
    cmap = [("class_DoS attacks-SlowHTTPTest", "SlowHTTPTest"), ("class_DoS attacks-GoldenEye", "GoldenEye"),
            ("class_DoS attacks-Slowloris", "Slowloris"), ("class_Benign", "Benign"),
            ("class_FTP-BruteForce", "FTP brute force"), ("class_Brute Force -Web", "Web brute force"),
            ("class_Infilteration", "Infiltration")]
    panels = [("orig", "Attempted flows relabeled benign"), ("K", "Attempted flows keep their attack class")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=True)
    y = np.arange(len(cmap))[::-1]
    rows = []
    for ax, (conv, title) in zip(axes, panels):
        s = g[g.conv == conv]
        assert len(s) == 5
        for yi, (col, lab) in zip(y, cmap):
            v = s[col].to_numpy() * 100
            m = float(v.mean())
            ax.barh(yi, m, height=0.62, color=C_ORIG if m > 0 else C_CORR,
                    xerr=[[m - float(v.min())], [float(v.max()) - m]], capsize=1.2, error_kw={"lw": 0.5})
            rows.append({"convention": conv, "class": lab, "mean_points": m, "min_points": float(v.min()), "max_points": float(v.max())})
        gap = s["gap"].to_numpy() * 100
        assert abs(sum(r["mean_points"] for r in rows if r["convention"] == conv) - gap.mean()) < 1e-3
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{title}\nmean gap {gap.mean():+.1f} points".replace("-", "\u2212"), fontsize=7.5)
        ax.set_xlabel("Contribution to the Macro-F1 gap, in points")
    axes[0].set_yticks(y, [lab for _, lab in cmap])
    h = [plt.Rectangle((0, 0), 1, 1, color=C_ORIG), plt.Rectangle((0, 0), 1, 1, color=C_CORR)]
    fig.legend(h, ["favors the model trained on the original labels", "favors the model trained on the corrected labels"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.17), fontsize=6.5)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_gap")
    pd.DataFrame(rows).to_csv(OUT / "fig_gap_data.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    fig_perclass()
    fig_neighbors()
    fig_anatomy()
    fig_gap()
    print("ok")
