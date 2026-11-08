#!/usr/bin/env python
"""Figures for the direction-A paper (LABELGAP): hidden-attack recovery, ranking inversion,
episode timelines, context rules. Reads the committed result files; the timelines rebuild the
A-D4 context frames. Output: results/figures/labelgap/fig*.{pdf,png} + timeline CSVs.
Revision round 1: CSE-CIC-IDS2018 uses convention K (attempted flows keep their attack class, as in
CIC-IDS2017); the ranking figure adds the Stage-I convention as a third panel (results/revision_round1/).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import load_yaml  # noqa: E402

OUT = ROOT / "results/figures/labelgap"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
MODELS = {"xgboost": "XGBoost", "mlp": "MLP", "cnn": "1D-CNN", "bigru": "BiGRU", "transformer": "Transformer"}
DS = {"cicids2017": "CIC-IDS2017", "cicids2018": "CSE-CIC-IDS2018"}
C_ORIG, C_CORR, C_HIDDEN, C_BENIGN = "#d95f02", "#1b9e77", "#e7298a", "#9e9e9e"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def fig_recovery():
    r = pd.read_csv(ROOT / "results/revision_round1/rescore_per_run.csv")
    methods = [("CE, original labels", "Original\nlabels"), ("GCE", "GCE"), ("Co-teaching", "Co-\nteaching"),
               ("Confident Learning", "Confident\nLearning"), ("CE, corrected labels", "Corrected\nlabels")]
    models = ["xgboost", "mlp", "transformer"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4), sharey=True)
    for ax, ds in zip(axes, ("cicids2017", "cicids2018_K")):
        g = r[r["dataset"] == ds]
        width = 0.26
        for j, m in enumerate(models):
            for i, (key, _) in enumerate(methods):
                q = g[(g["method"] == key) & (g["model"] == m)]["recovery"] * 100
                if q.empty:
                    continue
                x = i + (j - 1) * width
                ax.bar(x, q.mean(), width, yerr=q.std(ddof=1), capsize=1.5, color=colors[j],
                       error_kw={"elinewidth": 0.6}, label=MODELS[m] if i == 0 else None)
                ax.scatter(np.full(len(q), x), q, s=4, color="black", zorder=3, linewidths=0)
        ax.set_xticks(range(len(methods)), [lab for _, lab in methods], fontsize=6.3)
        ax.set_title(DS["cicids2017" if ds == "cicids2017" else "cicids2018"])
        ax.set_ylim(0, 105)
        for y, ls in ((80, "--"), (20, ":")):
            ax.axhline(y, color="0.6", lw=0.6, ls=ls, zorder=0)
    axes[0].set_ylabel("Omitted attacks recovered (%)")
    axes[0].legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.5))
    save(fig, "fig1_recovery")


def fig_ranking():
    d = pd.read_csv(ROOT / "results/revision_round1/fig1_data.csv")
    panels = [("cicids2017", "CIC-IDS2017"),
              ("cicids2018_K", "CSE-CIC-IDS2018\n(attempted = attack, as in CIC-IDS2017)"),
              ("cicids2018_stage1", "CSE-CIC-IDS2018\n(attempted = benign)")]
    markers = {"xgboost": "o", "mlp": "s", "cnn": "^", "bigru": "D", "transformer": "v"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (ds, title) in zip(axes, panels):
        for _, row in d[d["dataset"] == ds].iterrows():
            color = C_ORIG if row["method"] == "CE, original labels" else C_CORR
            ax.plot([0, 1], [row["macro_f1_corrected"], row["macro_f1_original"]], color=color, lw=0.9, alpha=0.85,
                    marker=markers[row["model"]], ms=3)
        ax.set_xticks([0, 1], ["Corrected\ntest labels", "Original\ntest labels"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(title, fontsize=7.5)
    axes[0].set_ylabel("Macro-F1")
    handles = [plt.Line2D([], [], color=C_ORIG, lw=0.9, label="trained on original labels"),
               plt.Line2D([], [], color=C_CORR, lw=0.9, label="trained on corrected labels")]
    handles += [plt.Line2D([], [], color="0.3", marker=mk, ls="none", ms=3, label=MODELS[m]) for m, mk in markers.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=6.5)
    fig.subplots_adjust(bottom=0.24, wspace=0.35)
    save(fig, "fig2_ranking_inversion")


TIMELINES = [("cicids2017", "thursday", "192.168.10.8"), ("cicids2018", "Wednesday-28-02-2018", "172.31.69.24")]
SERIES = [("labelled benign, benign in corrected labels", C_BENIGN, 0.7),
          ("labelled benign, attack in corrected labels", C_HIDDEN, 1.0),
          ("labelled as attack in the original", "black", 1.0)]
DISPLAY = {"labelled benign, benign in corrected labels": "labeled benign, benign in corrected labels",
           "labelled benign, attack in corrected labels": "labeled benign, attack in corrected labels",
           "labelled as attack in the original": "labeled as attack in the original"}


def timeline_data():
    """Per-minute flow counts of the host that performed the omitted scan (rebuilds the A-D4 frames)."""
    from directionA_d4_context import cicids2017, cicids2018
    cfg = load_yaml(ROOT / "configs/direction_a_d4_context.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    for (ds, day, host), loader in zip(TIMELINES, (cicids2017, cicids2018)):
        _, _, ctx = loader(cfg)
        fr = ctx["frame"]
        noisy_benign = ~ctx["attack"] & (fr["noisy"].astype(str).str.lower().eq("benign")).to_numpy()
        sel = (fr["day"].to_numpy() == day) & (fr["src"].to_numpy() == host)
        shown = pd.Series(ctx["clean_class"]).astype(str).str.upper()
        attack_k = ctx["known"] & ~shown.str.startswith("BENIGN").to_numpy() & ~shown.eq("NAN").to_numpy()
        omitted = sel & ctx["cand"] & attack_k          # convention K: attempted flows are attacks
        labelled = sel & ctx["attack"]
        benign = sel & noisy_benign & ~omitted
        t = fr["t"]
        lo = t[labelled | omitted].min() - pd.Timedelta(minutes=20)
        hi = t[labelled | omitted].max() + pd.Timedelta(minutes=20)
        inside = ((t >= lo) & (t <= hi)).to_numpy()
        cols = [t[m & inside].dt.floor("min").value_counts().sort_index().rename(name)
                for (name, _, _), m in zip(SERIES, (benign, omitted, labelled))]
        pd.concat(cols, axis=1).fillna(0).astype(int).to_csv(OUT / f"timeline_{ds}.csv")


def fig_timelines():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (ds, _, host) in zip(axes, TIMELINES):
        c = pd.read_csv(OUT / f"timeline_{ds}.csv", index_col=0, parse_dates=True)
        c = c.reindex(pd.date_range(c.index.min(), c.index.max(), freq="min"), fill_value=0)
        for name, color, lw in SERIES:
            ax.plot(c.index, c[name].where(c[name] > 0), drawstyle="steps-mid", color=color, lw=lw * 0.9,
                    label=DISPLAY[name])
        ax.set_yscale("log")
        ax.set_ylabel("Flows per minute")
        ax.set_title(f"{DS[ds]}, host {host}")
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    h, lab_ = axes[0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=7)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "fig3_episode_timelines")


def fig_context():
    r = pd.read_csv(ROOT / "results/revision_round1/context_rules.csv")
    r = r[(r["convention"] == "K") & (((r["dataset"] == "cicids2017") & (r["subset"] == "tier 1"))
                                      | ((r["dataset"] == "cicids2018") & (r["subset"] == "all matched")))]
    name = {("V1", 60): "Target rule, 60 min", ("V1", 240): "Target rule, 240 min",
            ("V2", 60): "Exposure rule, 60 min", ("V2", 240): "Exposure rule, 240 min",
            ("E2_cleaned", 60): "Exposure rule, cleaned evidence", ("CL_only", 0): "Confident Learning alone"}
    style = {"Target rule, 60 min": ("^", "0.45", 22), "Target rule, 240 min": ("^", "none", 34),
             "Exposure rule, 60 min": ("o", "black", 20), "Exposure rule, 240 min": ("o", "none", 34),
             "Exposure rule, cleaned evidence": ("*", "#d95f02", 70), "Confident Learning alone": ("s", "#7570b3", 18)}
    offset = {"Exposure rule, cleaned evidence": (-0.13, 0.0)}
    titles = {"cicids2017": "CIC-IDS2017 (exact-match flows)", "cicids2018": "CSE-CIC-IDS2018 (all aligned flows)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, ds in zip(axes, DS):
        ax.add_patch(plt.Rectangle((0.5, 0.8), 0.58, 0.23, color="#e8f5e9", zorder=0))
        for _, row in r[r["dataset"] == ds].iterrows():
            lab = name[(row["rule"], int(row["window_min"]))]
            mk, fc, sz = style[lab]
            x, y = row["recall_hidden"], row["precision"]
            if ds == "cicids2017" and lab in offset:           # would sit on the exposure-rule marker
                ax.annotate("", xy=(x, y), xytext=(x + offset[lab][0], y + offset[lab][1]),
                            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.4"})
                x, y = x + offset[lab][0], y + offset[lab][1]
            ax.scatter(x, y, s=sz, marker=mk, facecolors=fc, edgecolors="black" if fc == "none" else fc,
                       linewidths=0.8, zorder=4 if "cleaned" in lab else 3, label=lab)
        ax.set_xlim(-0.03, 1.08)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Recall of omitted attacks")
        ax.set_title(titles[ds])
    axes[0].set_ylabel("Precision")
    h, lab_ = axes[0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=6.5)
    fig.subplots_adjust(bottom=0.3)
    save(fig, "fig4_context_rules")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["recovery", "ranking", "context", "timeline_data", "timelines"]
    for name in todo:
        {"recovery": fig_recovery, "ranking": fig_ranking, "context": fig_context, "timeline_data": timeline_data,
         "timelines": fig_timelines}[name]()
    print("figures written to", OUT)
