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

from src.utils import md_table  # noqa: E402
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
    ap.add_argument("--title", default="CICIDS2017 原版标签 vs Liu 修正标签")
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
    w(f"# DIRECTION A — Stage I：带错标签训练的后果（{args.title}）\n")
    w(f"> 研究者 2026-09-25 选定（与方向 B 并行）。生成时间 {time.strftime('%Y-%m-%d %H:%M')}；代码 commit "
      f"`{git_commit()[:7]}`。判读规则 `configs/direction_a_label_noise.yaml` 在训练前提交。同一批行、同一划分，"
      f"只换训练/验证标签：noisy = 原版标签，clean = Liu 修正标签（上限对照）。测试一律按修正标签评估。"
      f"{len(models)} 个模型 × {len(seeds)} 个种子 × 2 种条件。\n")
    interp = ROOT / args.interp
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    s = load_json(out / "data_summary.json")
    aff_test_n = int(affected.sum())
    w("## A-0　数据与第 4 步数据体检\n")
    w(f"覆盖行 {s['rows']:,}（训练 {s['train']:,} / 验证 {s['val']:,} / 测试 {s['test']:,}），{s['n_features']} 个特征；"
      f"两版标签不一致的行 {s['affected_rows_total']:,}（{s['global_noise_rate']:.2%}），其中测试集 "
      f"{s['affected_rows_test']:,}。其中“攻击被原版标成正常”的占 "
      f"{((yn == benign) & (yc != benign)).sum() / max(aff_test_n, 1):.1%}（测试集），其余方向见 label_transitions.csv。\n")
    p = prof.copy()
    p.loc[p["class"] == names[benign], "hidden_as_benign"] = np.nan
    w(md_table(p[["class", "rows_clean", "rows_noisy", "hidden_as_benign", "noise_rate_clean_class", "test_clean",
                  "test_affected", "train_conflict_rate_clean_labels", "separability_limited"]].rename(columns={
        "class": "Class (clean)", "rows_clean": "Rows (clean)", "rows_noisy": "Rows (noisy)",
        "hidden_as_benign": "Hidden as BENIGN", "noise_rate_clean_class": "Noise rate",
        "test_clean": "Test rows", "test_affected": "Test affected",
        "train_conflict_rate_clean_labels": "Train conflict rate (clean)",
        "separability_limited": "Separability-limited"})))
    w("\n## A-1　整体结果（测试集按修正标签评估；3 个种子 mean ± std）\n")
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
    w(f"\n## A-2　A1：带错标签导致的类别召回下降（配对种子；测试行 ≥ {min_sup} 的类别）\n")
    thr = r["A1_noise_induced_class_failure"]["min_recall_drop"]
    w(f"规则：召回(clean 训练) − 召回(noisy 训练) ≥ {thr}，且所有种子的差值为正，在 ≥ {kmin} 个模型中成立即标记；"
      f"上限对照：clean 训练召回 ≥ {rec_ok}。\n")
    cols = ["class", "n_test", "n_affected"] + [f"{m}_{k}" for m in models for k in ("noisy", "clean")] + \
           ["models_failing", "flagged", "oracle_recovers", "method_addressable"]
    show = a1[cols].copy()
    show.columns = ["Class", "Test n", "Affected n"] + [f"{MODEL_DISPLAY[m]} {k}" for m in models for k in ("noisy", "clean")] + \
                   ["Models failing", "Flagged", "Oracle recovers", "Method-addressable"]
    w(md_table(show))
    w(f"\n## A-3　A2：被错标的测试行上，带错标签训练的模型能否认出真实类别\n")
    w(f"数值为被错标行中预测为真实（修正）类别的比例（3 个种子均值）。规则：noisy 训练 ≤ "
      f"{r['A2_hidden_attack_failure']['noisy_trained_at_most']} 的模型 ≥ {kmin} 个即标记；上限对照：clean 训练 ≥ {rec_ok}。\n")
    cols = ["class", "n_affected_test"] + [f"{m}_{k}" for m in models for k in ("noisy", "clean")] + \
           ["models_failing", "models_oracle_recovers", "flagged", "oracle_recovers", "method_addressable"]
    show = a2[cols].copy()
    show.columns = ["Class", "Affected test n"] + [f"{MODEL_DISPLAY[m]} {k}" for m in models for k in ("noisy", "clean")] + \
                   ["Models failing", "Models oracle ok", "Flagged", "Oracle recovers", "Method-addressable"]
    w(md_table(show))
    w(f"\n## A-4　A3：常规评测（按原版测试标签）是否掩盖问题\n")
    w(f"noisy 训练模型：Macro-F1(原版测试标签) − Macro-F1(修正测试标签)；≥ {r['A3_evaluation_masking']['min_overstatement']} "
      f"的模型 ≥ {kmin} 个即标记。**判定：{'标记' if a3_flag else '不标记'}**。\n")
    w(md_table(pd.DataFrame({"Model": [MODEL_DISPLAY[m] for m in models],
                             "Macro-F1 noisy test": [g.loc[("noisy", m), "macro_f1_noisy"] for m in models],
                             "Macro-F1 clean test": [g.loc[("noisy", m), "macro_f1_clean"] for m in models],
                             "Overstatement": [over[m] for m in models],
                             "Accuracy noisy test": [g.loc[("noisy", m), "acc_noisy"] for m in models],
                             "Accuracy clean test": [g.loc[("noisy", m), "acc_clean"] for m in models]})))
    w("\n## 文件\n")
    w(f"`{args.results}/`：data_profile.csv、data_summary.json、overall_per_run.csv、per_class_recall_per_run.csv、"
      f"A1_class_failures.csv、A2_hidden_attack_failures.csv。运行结果：`{args.runs}/{{noisy,clean}}/`。"
      "代码：`scripts/directionA_analyze.py`（数据准备与运行脚本见对应数据集的 prepare / run 脚本）。")
    (ROOT / args.report).write_text("\n".join(L) + "\n")
    print(pd.DataFrame(tab).to_string(index=False))
    print(a1[["class", "n_test", "models_failing", "flagged", "oracle_recovers", "method_addressable"]].to_string(index=False))
    print(a2[["class", "n_affected_test", "models_failing", "models_oracle_recovers", "flagged", "method_addressable"]]
          .to_string(index=False))
    print("A3 overstatement:", over.round(4).to_dict(), "flag:", a3_flag)


if __name__ == "__main__":
    main()
