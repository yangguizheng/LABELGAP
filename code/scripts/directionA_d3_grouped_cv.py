#!/usr/bin/env python
"""A-D3: do group-held-out cross-validated predictions expose the systematically hidden
attacks? Compares fold groupings (exact duplicates = standard Confident Learning,
k-means clusters at three granularities, 60-minute time windows) on the training
partitions of CICIDS2017 and CSE-CIC-IDS2018, using noisy labels only.
Criteria: configs/direction_a_d3_grouped_cv.yaml.
Outputs: results/directionA_d3/*, reports/DIRECTION_A_D3.md
(interpretation spliced from reports/direction_a_d3_interpretation.md).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from cleanlab.filter import find_label_issues  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.utils import md_table  # noqa: E402
from src.utils import git_commit, load_yaml  # noqa: E402

warnings.filterwarnings("ignore")


def time_groups(name: str, common: dict, ids: np.ndarray) -> np.ndarray:
    """(day, 60-min window) of every training row."""
    if name == "cicids2018":
        ri = pd.read_parquet(ROOT / "results/directionA2018/row_index.parquet", columns=["day", "t"])
        ri = ri.iloc[ids]
        return (ri["day"].astype(str) + "|" + (ri["t"] // 3600).astype(str)).to_numpy()
    from directionA_prepare import load_original
    orig = load_original(common)
    mask = pd.read_parquet(common["dataset"]["noise_mask"])
    keep = (mask["tier_liu"].to_numpy() > 0) & (mask["base_orig"].to_numpy() != "UNLABELED")
    meta = orig.loc[keep, ["day", "Timestamp"]].reset_index(drop=True).iloc[ids]
    ts = pd.to_datetime(meta["Timestamp"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    ts = ts.where(ts.dt.hour >= 8, ts + pd.Timedelta(hours=12))        # 12-h clock without AM/PM
    return (meta["day"].astype(str) + "|" + ts.dt.floor("60min").astype(str)).to_numpy()


def oof_probs(x, y, groups, xv, yv, params, n_classes, seed):
    probs = np.zeros((len(y), n_classes))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(x, y, groups):
        present = np.unique(y[tr])
        lookup = np.full(n_classes, -1)
        lookup[present] = np.arange(len(present))
        vm = np.isin(yv, present)
        clf = xgb.XGBClassifier(**params, random_state=seed)
        clf.fit(x[tr], lookup[y[tr]], eval_set=[(xv[vm], lookup[yv[vm]])], verbose=False)
        probs[np.ix_(te, present)] = clf.predict_proba(x[te], iteration_range=(0, clf.best_iteration + 1))
    return probs


def main() -> None:
    cfg = load_yaml(ROOT / "configs/direction_a_d3_grouped_cv.yaml")
    params = load_yaml(ROOT / "configs/xgboost.yaml")["params"]
    km = cfg["kmeans"]
    out = ROOT / "results/directionA_d3"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = []
    for ds, common_path in cfg["datasets"].items():
        common = load_yaml(ROOT / common_path)
        d = np.load(ROOT / common["dataset"]["processed_file"], allow_pickle=False)
        names = [str(c) for c in d["class_names"]]
        x, yn, yc, ids = d["X_train"], d["y_train_noisy"], d["y_train_clean"], d["id_train"]
        xv, yvn = d["X_val"], d["y_val_noisy"]
        benign = names.index(next(n for n in ("BENIGN", "Benign") if n in names))
        hidden = (yn == benign) & (yc != benign)
        affected = yn != yc
        clean_benign = (yc == benign) & (yn == benign)
        groupings = {"exact": pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()}
        for k in (64, 256, 1024):
            groupings[f"kmeans_{k}"] = MiniBatchKMeans(n_clusters=k, random_state=km["random_state"],
                                                       batch_size=km["batch_size"], n_init=km["n_init"]).fit_predict(x)
        groupings["time_60min"] = time_groups(ds, common, ids)
        print(f"[{ds}] groupings ready ({time.perf_counter() - t0:.0f}s)", flush=True)
        for gname, g in groupings.items():
            probs = oof_probs(x, yn, g, xv, yvn, params, len(names), 2026)
            try:
                issues = find_label_issues(yn, probs, filter_by="prune_by_noise_rate")
            except Exception as e:                                        # noqa: BLE001
                print(f"  cleanlab failed for {gname}: {e}", flush=True)
                issues = np.zeros(len(yn), bool)
            pred = probs.argmax(1)
            sizes = pd.Series(g).value_counts()
            r = {"dataset": ds, "grouping": gname, "n_groups": int(len(sizes)),
                 "largest_group_share": float(sizes.iloc[0] / len(g)),
                 "hidden_rows": int(hidden.sum()), "flagged": int(issues.sum()),
                 "precision": float((issues & affected).sum() / max(issues.sum(), 1)),
                 "recall_hidden": float(issues[hidden].mean()),
                 "recall_all_affected": float(issues[affected].mean()),
                 "exposure_hidden": float((pred[hidden] != benign).mean()),
                 "exposure_true_class": float((pred[hidden] == yc[hidden]).mean()),
                 "false_exposure_benign": float((pred[clean_benign] != benign).mean())}
            for c in np.unique(yc[hidden]):
                h = hidden & (yc == c)
                if h.sum() >= 100:
                    r[f"recall_hidden_{names[c]}"] = float(issues[h].mean())
                    r[f"exposure_true_{names[c]}"] = float((pred[h] == c).mean())
            rows.append(r)
            print(f"  {gname}: groups {r['n_groups']}, flagged {r['flagged']:,}, precision {r['precision']:.3f}, "
                  f"recall_hidden {r['recall_hidden']:.3f}, exposure_hidden {r['exposure_hidden']:.3f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "grouped_cv.csv", index=False, float_format="%.6f")

    v = cfg["verdict"]["exposes"]
    res["exposes"] = (res["recall_hidden"] >= v["min_recall_hidden"]) & (res["precision"] >= v["min_precision"])
    both = res.groupby("grouping")["exposes"].all()
    anyds = res.groupby("grouping")["exposes"].any()
    verdict = "signal_found" if both.any() else ("dataset_specific" if anyds.any() else "no_signal")

    L = []
    w = L.append
    w("# DIRECTION A — 诊断 A-D3：整群留出的交叉验证能否让被漏标的攻击显形\n")
    w(f"> 研究者 2026-09-26 选定。生成时间 {time.strftime('%Y-%m-%d %H:%M')}；代码 commit `{git_commit()[:7]}`。"
      "判读标准 `configs/direction_a_d3_grouped_cv.yaml` 在计算前提交。只用原版（带错）标签；修正标签只用于打分。"
      "exact 分组即标准 Confident Learning 的做法（基线）。\n")
    interp = ROOT / "reports/direction_a_d3_interpretation.md"
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    w("## D3-a　各分组方式的查错效果（训练集）\n")
    w(f"exposes ⇔ 被漏标攻击的查全率 ≥ {v['min_recall_hidden']} 且查准率 ≥ {v['min_precision']}。"
      f"**判定：{verdict}**（在两个数据集上都 exposes 的分组：{', '.join(both.index[both]) or '无'}）。\n")
    show = res[["dataset", "grouping", "n_groups", "largest_group_share", "flagged", "precision", "recall_hidden",
                "recall_all_affected", "exposure_hidden", "exposure_true_class", "false_exposure_benign", "exposes"]]
    w(md_table(show.rename(columns={
        "dataset": "Dataset", "grouping": "Grouping", "n_groups": "Groups", "largest_group_share": "Largest group",
        "flagged": "Flagged", "precision": "Precision", "recall_hidden": "Recall (hidden)",
        "recall_all_affected": "Recall (all affected)", "exposure_hidden": "Hidden → attack",
        "exposure_true_class": "Hidden → true class", "false_exposure_benign": "Benign → attack",
        "exposes": "Exposes"})))
    per = [c for c in res.columns if c.startswith(("recall_hidden_", "exposure_true_"))]
    if per:
        w("\n## D3-b　按类别（被漏标行数 ≥ 100 的类别）\n")
        w(md_table(res[["dataset", "grouping"] + per].rename(columns={"dataset": "Dataset", "grouping": "Grouping"})))
    w("\n## 文件\n")
    w("`results/directionA_d3/grouped_cv.csv`。代码：`scripts/directionA_d3_grouped_cv.py`。")
    (ROOT / "reports/DIRECTION_A_D3.md").write_text("\n".join(L) + "\n")
    print(res.drop(columns=per).to_string(index=False, float_format=lambda v_: f"{v_:.3f}"))
    print("verdict:", verdict, f"({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
