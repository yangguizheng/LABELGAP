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

from src.utils import md_table  # noqa: E402
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
    sys.path.insert(0, str(ROOT / "alignment_cicids2017"))
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
    w("# DIRECTION A — 诊断 A-D2：被系统性漏标的攻击在哪里（特征空间与时间/主机上下文）\n")
    w(f"> 研究者 2026-09-25 选定。生成时间 {time.strftime('%Y-%m-%d %H:%M')}；代码 commit `{git_commit()[:7]}`。"
      "判读标准 `configs/direction_a_d2_mechanism.yaml` 在计算前提交。只做计算，不训练模型。\n")
    interp = ROOT / "reports/direction_a_d2_interpretation.md"
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    w("## D2-a　特征空间位置（训练集，预处理后的 77 维特征，欧氏距离）\n")
    w("r = 到同类已正确标注攻击的最近距离 / 到真正正常流量的最近距离；r < 1 表示更靠近同类攻击。"
      f"近邻构成为 {k} 个最近训练样本中各类所占比例（不含自身）。“已标注攻击（对照）”为同类已正确标注行的随机样本。\n")
    t = feat[["class", "hidden_train", "hidden_median_ratio", "hidden_share_ratio_lt_1", "labelled_median_ratio",
              "hidden_median_d_labelled_c", "hidden_median_d_benign", "hidden_median_d_hidden",
              "hidden_mean_share_labelled_c", "hidden_mean_share_benign", "hidden_mean_share_hidden",
              "proximity_signal", "self_cluster"]].copy()
    t.columns = ["Class", "Hidden rows (train)", "Median r (hidden)", "Share r<1 (hidden)", "Median r (labelled, ref.)",
                 "Median d to labelled c", "Median d to benign", "Median d to other hidden",
                 "Nbr share: labelled c", "Nbr share: benign", "Nbr share: hidden",
                 "Proximity signal", "Self-cluster"]
    w(md_table(t))
    w("\n## D2-b　时间与主机上下文（全部覆盖行）\n")
    w(f"时间戳解析成功率 {parsed:.1%}（原始文件为 12 小时制且无 AM/PM，8 点以前的小时加 12）。"
      "co-occurrence = 漏标行所在的（星期, 60 分钟时段）里同时存在同类已标注攻击的比例。"
      "Attempted = Liu 等人标为“未遂”的比例（仅精确匹配的 tier-1 行）。\n")
    t = ctx[["class", "hidden_all", "days", "hidden_time_span", "labelled_time_span", "top3_src_share", "top_src",
             "top3_dst_share", "densest_60min_share", "co_occurrence", "attempted_share_tier1",
             "host_concentration", "time_concentration", "within_campaign"]].copy()
    t.columns = ["Class", "Hidden rows (all)", "Days", "Hidden time span", "Labelled time span",
                 "Top-3 src share", "Top src IPs", "Top-3 dst share", "Densest 60-min share", "Co-occurrence",
                 "Attempted (tier-1)", "Host conc.", "Time conc.", "Within campaign"]
    w(md_table(t))
    rb = out / "label_robustness_summary.csv"
    if rb.exists():
        w("\n## D2-c　【事后分析，非预注册】修正标签的可靠性与第一阶段结论的稳健性\n")
        w("D2-b 显示：除 Infiltration 端口扫描外，被漏标行的修正标签几乎都来自 tier-2（按 (星期, 流 ID) 推断），"
          "而 tier-1 的被漏标 Bot 行全部被 Liu 标为“未遂”。因此用第一阶段已保存的预测（不重新训练）在两个更可靠的"
          "子集上重算：R1 = 只看 tier-1（精确匹配）测试行；R2 = R1 且把“未遂”流量按数据集作者建议改算 BENIGN。"
          "数值为 5 个模型 × 3 个种子的均值（`scripts/directionA_d2_label_robustness.py`）。\n")
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
        w("\n*各子集中被漏标攻击按类别的数量（测试集）：*\n")
        w(md_table(bc.rename(columns={"variant": "Evaluation subset"}), nd=0))
    w("\n## 文件\n")
    w("`results/directionA_d2/`：feature_space.csv、hidden_rows_feature_space.parquet（逐行距离与近邻构成）、"
      "context.csv、label_robustness*.csv、summary.json。代码：`scripts/directionA_d2_mechanism.py`、"
      "`scripts/directionA_d2_label_robustness.py`。")
    (ROOT / "reports/DIRECTION_A_D2.md").write_text("\n".join(L) + "\n")
    print(feat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(ctx.drop(columns=["top_src", "top_dst"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"timestamp parse rate {parsed:.3f}")


if __name__ == "__main__":
    main()
