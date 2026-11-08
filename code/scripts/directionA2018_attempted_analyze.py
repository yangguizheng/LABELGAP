#!/usr/bin/env python
"""Direction A, CSE-CIC-IDS2018: attempted-flow convention sensitivity -- analysis.

Scores three conventions on one footing (same metric code for all of them):
  orig  Stage-I convention (attempted -> Benign): runs directionA2018/{noisy,clean}
  K     attempted keep their base attack class: noisy runs = Stage I (identical rows and labels),
        clean runs directionA2018_attK/clean
  E     attempted rows removed: runs directionA2018_attE/{noisy,clean}
Decision rules: configs/direction_a2018_attempted.yaml (committed before any run).
Outputs: results/directionA2018_attempted/*.csv, reports/DIRECTION_A2018_ATTEMPTED.md
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

from src.utils import MODEL_DISPLAY, MODEL_ORDER, git_commit, load_json, load_yaml, save_json  # noqa: E402

CONDS = ("noisy", "clean")
RUNS = {"orig": {"noisy": "directionA2018/noisy", "clean": "directionA2018/clean"},
        "K": {"noisy": "directionA2018/noisy", "clean": "directionA2018_attK/clean"},
        "E": {"noisy": "directionA2018_attE/noisy", "clean": "directionA2018_attE/clean"}}
LABEL = {"orig": "Stage I（未遂→正常）", "K": "K（未遂保留攻击类别，同 2017）", "E": "E（剔除未遂流量）"}


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.{nd}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pm(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"


def main() -> None:
    rules = load_yaml(ROOT / "configs/direction_a2018_attempted.yaml")
    stage1 = load_yaml(ROOT / "configs/directionA2018_common.yaml")
    out = ROOT / rules["outputs"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    files = {"orig": stage1["dataset"]["processed_file"],
             "K": rules["data"]["variants"]["K"]["processed_file"],
             "E": rules["data"]["variants"]["E"]["processed_file"]}
    meta = pd.read_parquet(ROOT / "results/directionA2018/row_index.parquet")
    is_att_all = meta["att"].to_numpy() != -1
    groups = rules["metrics"]["class_groups"]

    rows, pcf = [], []
    tests = {}
    for conv, f in files.items():
        d = np.load(ROOT / f, allow_pickle=False)
        names = [str(c) for c in d["class_names"]]
        yc, yn, ids = d["y_test_clean"], d["y_test_noisy"], d["id_test"]
        ben, inf = names.index("Benign"), names.index("Infilteration")
        lab_c = [c for c in range(len(names)) if (yc == c).any()]
        lab_n = [c for c in range(len(names)) if (yn == c).any()]
        lab_both = sorted(set(lab_c) & set(lab_n))
        omitted = (yn == ben) & (yc == inf)
        tests[conv] = {"names": names, "yc": yc, "yn": yn, "ids": ids, "lab_n": lab_n, "lab_c": lab_c}
        for cond in CONDS:
            for m in MODEL_ORDER:
                for p in sorted((ROOT / RUNS[conv][cond] / "results/raw" / m).glob("seed*/test_predictions.parquet")):
                    q = pd.read_parquet(p, columns=["sample_id", "pred_idx"])
                    assert (q["sample_id"].to_numpy() == ids).all(), (conv, cond, m, p)
                    pred = q["pred_idx"].to_numpy()
                    seed = int(p.parent.name[4:])
                    f1n = f1_score(yn, pred, labels=lab_n, average=None, zero_division=0)
                    f1c = f1_score(yc, pred, labels=lab_c, average=None, zero_division=0)
                    rows.append({"conv": conv, "cond": cond, "model": m, "seed": seed,
                                 "macro_f1_corrected": float(f1c.mean()), "macro_f1_original": float(f1n.mean()),
                                 "macro_f1_original_fixed": f1_score(yn, pred, labels=lab_both, average="macro",
                                                                     zero_division=0),
                                 "macro_f1_corrected_fixed": f1_score(yc, pred, labels=lab_both, average="macro",
                                                                      zero_division=0),
                                 "acc_corrected": float((pred == yc).mean()),
                                 "false_alarm": float((pred[yc == ben] != ben).mean()),
                                 "recovery": float((pred[omitted] == inf).mean()),
                                 "n_classes_original": len(lab_n), "n_classes_corrected": len(lab_c)})
                    for c, v in zip(lab_n, f1n):
                        pcf.append({"conv": conv, "cond": cond, "model": m, "seed": seed, "labels": "original",
                                    "class": names[c], "f1": float(v), "n_test": int((yn == c).sum())})
                    for c, v in zip(lab_c, f1c):
                        pcf.append({"conv": conv, "cond": cond, "model": m, "seed": seed, "labels": "corrected",
                                    "class": names[c], "f1": float(v), "n_test": int((yc == c).sum())})
    ov, pc = pd.DataFrame(rows), pd.DataFrame(pcf)
    ov.to_csv(out / "overall_per_run.csv", index=False, float_format="%.6f")
    pc.to_csv(out / "per_class_f1_per_run.csv", index=False, float_format="%.6f")
    models = [m for m in MODEL_ORDER if (ov["model"] == m).any()]
    n_runs = ov.groupby(["conv", "cond", "model"]).size()
    assert (n_runs == 3).all(), n_runs[n_runs != 3]

    # ---- inversion and gap decomposition (original test labels, seed means) ------------------------------
    g = ov.groupby(["conv", "cond", "model"]).mean(numeric_only=True)
    pcm = pc[pc["labels"] == "original"].groupby(["conv", "cond", "model", "class"])["f1"].mean()
    inv, dec = [], []
    for conv in RUNS:
        n_cls = len(tests[conv]["lab_n"])
        for m in models:
            a, b = g.loc[(conv, "noisy", m), "macro_f1_original"], g.loc[(conv, "clean", m), "macro_f1_original"]
            inv.append({"conv": conv, "model": m, "noisy_trained": a, "clean_trained": b, "gap": a - b,
                        "inverts": bool(b < a)})
            contrib = {c: (pcm.loc[(conv, "noisy", m, c)] - pcm.loc[(conv, "clean", m, c)]) / n_cls
                       for c in pcm.loc[(conv, "noisy", m)].index}
            grp = {"infiltration": sum(v for c, v in contrib.items() if c in groups["infiltration"]),
                   "attack_type": sum(v for c, v in contrib.items() if c in groups["attack_type"])}
            grp["other"] = sum(contrib.values()) - grp["infiltration"] - grp["attack_type"]
            gap = sum(contrib.values())
            assert abs(gap - (a - b)) < 1e-9
            dec.append({"conv": conv, "model": m, "gap": gap, **{f"contrib_{k}": v for k, v in grp.items()},
                        **{f"share_{k}": (v / gap if gap > 0 else np.nan) for k, v in grp.items()},
                        **{f"class_{c}": v for c, v in contrib.items()}})
    inv, dec = pd.DataFrame(inv), pd.DataFrame(dec)
    inv.to_csv(out / "inversion_per_model.csv", index=False, float_format="%.6f")
    dec.to_csv(out / "gap_decomposition.csv", index=False, float_format="%.6f")

    def q_inversion(conv: str) -> str:
        k = int(inv[inv["conv"] == conv]["inverts"].sum())
        return "replicates" if k == len(models) else ("absent" if k == 0 else "partial")

    verdicts = {"Q1_inversion_K": q_inversion("K"), "Q3_inversion_E": q_inversion("E"),
                "reference_inversion_orig": q_inversion("orig")}
    if verdicts["Q1_inversion_K"] in ("replicates", "partial"):
        dk = dec[(dec["conv"] == "K") & dec["model"].isin(inv[(inv["conv"] == "K") & inv["inverts"]]["model"])]
        maj = len(dk) // 2 + 1
        if (dk["share_infiltration"] >= 0.5).sum() >= maj:
            verdicts["Q2_driver_K"] = "infiltration_driven"
        elif (dk["share_attack_type"] >= 0.5).sum() >= maj:
            verdicts["Q2_driver_K"] = "attack_type_driven"
        else:
            verdicts["Q2_driver_K"] = "mixed"
    else:
        verdicts["Q2_driver_K"] = "not_applicable"

    # ---- sanity: clean-trained recovery of the omitted flows ---------------------------------------------
    sanity = {}
    for conv in ("K", "E"):
        rec = [g.loc[(conv, "clean", m), "recovery"] for m in models]
        sanity[conv] = {"clean_trained_recovery": dict(zip(models, map(float, rec))),
                        "passes": sum(r >= 0.80 for r in rec) >= 3}
    verdicts["sanity_pass"] = all(v["passes"] for v in sanity.values())

    # ---- Q4: where the Stage-I false alarms come from (noisy-trained models) ---------------------------
    t = tests["orig"]
    ben = t["names"].index("Benign")
    cb = t["yc"] == ben
    att = is_att_all[t["ids"]]
    grp_def = {"attempted": cb & att, "non_attempted_attack_label": cb & ~att & (t["yn"] != ben),
               "benign_in_both": cb & ~att & (t["yn"] == ben)}
    q4 = []
    for m in models:
        for p in sorted((ROOT / RUNS["orig"]["noisy"] / "results/raw" / m).glob("seed*/test_predictions.parquet")):
            pred = pd.read_parquet(p, columns=["pred_idx"])["pred_idx"].to_numpy()
            fa = pred != ben
            tot = int((fa & cb).sum())
            for k, sel in grp_def.items():
                q4.append({"model": m, "seed": int(p.parent.name[4:]), "group": k, "n": int(sel.sum()),
                           "false_alarm_rate": float(fa[sel].mean()), "false_alarms": int((fa & sel).sum()),
                           "share_of_all_false_alarms": float((fa & sel).sum() / max(tot, 1)),
                           "overall_rate": float(fa[cb].mean())})
    q4 = pd.DataFrame(q4)
    q4.to_csv(out / "false_alarm_decomposition_per_run.csv", index=False, float_format="%.6f")
    save_json({"verdicts": verdicts, "sanity": sanity, "models": models,
               "classes_original_test": {c: [tests[c]["names"][i] for i in tests[c]["lab_n"]] for c in RUNS},
               "classes_corrected_test": {c: [tests[c]["names"][i] for i in tests[c]["lab_c"]] for c in RUNS}},
              out / "verdicts.json")

    # ---- report ---------------------------------------------------------------------------------------------
    vs = load_json(out / "variants_summary.json")
    L = []
    w = L.append
    w("# DIRECTION A（CSE-CIC-IDS2018）— 未遂流量处理方式的敏感性\n")
    w(f"> 生成时间 {time.strftime('%Y-%m-%d %H:%M')}；代码 commit `{git_commit()[:7]}`。判读规则 "
      "`configs/direction_a2018_attempted.yaml` 在构建变体和训练之前提交。Stage I 把修正版中所有“未遂”流量映射为正常；"
      "CIC-IDS2017 则保留未遂流量的攻击类别。本分析在同一批行、同一划分、同一预处理与训练协议下比较三种约定："
      "Stage I、K（与 2017 一致，主结果）、E（剔除未遂流量，辅助）。K 只重训修正标签组（原版标签组的行与标签与 Stage I 完全相同，"
      "直接用 Stage I 预测按 K 的测试标签重新计分）；E 两组都重训。5 个模型 × 3 个种子。\n")
    for a in rules.get("amendments", []):
        w(f"> 补充说明（{a['date']}）：{a['what']}\n")
    w("## 判定\n")
    w(f"- **Q1（K 上排名反转是否复现）**：`{verdicts['Q1_inversion_K']}`")
    w(f"- **Q2（K 上反转由哪类错误驱动）**：`{verdicts['Q2_driver_K']}`")
    w(f"- **Q3（E 上排名反转是否复现）**：`{verdicts['Q3_inversion_E']}`")
    w(f"- 参照：Stage I 约定下 `{verdicts['reference_inversion_orig']}`")
    w(f"- 合理性检查（K、E 中修正标签训练的漏标找回率 ≥ 0.80 的模型 ≥ 3 个）：{'通过' if verdicts['sanity_pass'] else '**未通过**'}\n")
    w("## 1　三种约定下的测试集\n")
    w(f"未遂流量共 {vs['attempted_rows']:,} 行（测试集 {vs['attempted_by_split']['test']:,} 行）。")
    for v in ("K", "E"):
        w(f"{v}：测试集 {vs[v]['rows']['test']:,} 行，两版标签不一致 {vs[v]['disagreeing_rows']['test']:,} 行；"
          f"修正版测试标签中出现的类别：{', '.join(vs[v]['clean_classes_with_test_rows'])}。")
    w("两版标签的对应关系见 `variants_K_transitions.csv`、`variants_E_transitions.csv`。\n")
    w("## 2　总体结果（3 个种子 mean ± std）\n")
    tab = []
    for conv in RUNS:
        for cond in CONDS:
            for m in models:
                q = ov[(ov["conv"] == conv) & (ov["cond"] == cond) & (ov["model"] == m)]
                tab.append({"约定": LABEL[conv], "训练标签": cond, "模型": MODEL_DISPLAY[m],
                            "Macro-F1（修正测试标签）": pm(q["macro_f1_corrected"]),
                            "Macro-F1（原版测试标签）": pm(q["macro_f1_original"]),
                            "Macro-F1（原版，共同类别）": pm(q["macro_f1_original_fixed"]),
                            "误报率（修正版正常流量）": pm(q["false_alarm"]),
                            "漏标找回率": pm(q["recovery"])})
    w(md_table(pd.DataFrame(tab)))
    w("\n## 3　Q1 / Q3：按原版测试标签评估时的排名反转\n")
    w("每个模型取 3 个种子的均值；修正标签训练 < 原版标签训练即记为反转。\n")
    show = inv.assign(约定=inv["conv"].map(LABEL), 模型=inv["model"].map(MODEL_DISPLAY))[
        ["约定", "模型", "noisy_trained", "clean_trained", "gap", "inverts"]].rename(columns={
            "noisy_trained": "原版标签训练", "clean_trained": "修正标签训练", "gap": "差值", "inverts": "反转"})
    w(md_table(show))
    w("\n## 4　Q2：Macro-F1 差值按类别分解（原版测试标签）\n")
    w("类别 k 的贡献 =（原版训练 F1_k − 修正训练 F1_k）/ 原版测试标签中出现的类别数，各类贡献之和等于 Macro-F1 差值。"
      "分组：infiltration = Benign + Infilteration；attack_type = FTP-BruteForce + DoS attacks-SlowHTTPTest；其余为 other。\n")
    cols = ["conv", "model", "gap", "contrib_infiltration", "contrib_attack_type", "contrib_other",
            "share_infiltration", "share_attack_type", "share_other"]
    w(md_table(dec[cols].assign(conv=dec["conv"].map(LABEL), model=dec["model"].map(MODEL_DISPLAY))))
    w("\n逐类贡献见 `gap_decomposition.csv`，逐类 F1 见 `per_class_f1_per_run.csv`。\n")
    w("## 5　Q4：Stage I 的误报从哪里来（原版标签训练的模型）\n")
    q4m = q4.groupby(["model", "group"]).mean(numeric_only=True).reset_index()
    w(md_table(q4m.assign(model=q4m["model"].map(MODEL_DISPLAY))[
        ["model", "group", "n", "false_alarm_rate", "share_of_all_false_alarms", "overall_rate"]]))
    w("\n## 6　合理性检查\n")
    for conv in ("K", "E"):
        w(f"- {LABEL[conv]}：修正标签训练的漏标找回率 " + ", ".join(
            f"{MODEL_DISPLAY[m]} {v:.3f}" for m, v in sanity[conv]["clean_trained_recovery"].items())
          + f"；{'通过' if sanity[conv]['passes'] else '未通过'}")
    w("\n## 7　预先写定的解读\n")
    for s in rules["interpretation_fixed_in_advance"]:
        w(f"- {s}")
    w("\n## 文件\n")
    w(f"`{rules['outputs']['results']}`：variants_summary.json、variants_*_transitions.csv、overall_per_run.csv、"
      "per_class_f1_per_run.csv、inversion_per_model.csv、gap_decomposition.csv、"
      "false_alarm_decomposition_per_run.csv、verdicts.json。运行结果：directionA2018_attK/clean、"
      "directionA2018_attE/{noisy,clean}；原版标签组的 K 结果复用 directionA2018/noisy。"
      "代码：scripts/directionA2018_attempted_variants.py、scripts/directionA2018_attempted_analyze.py、"
      "scripts/run_directionA2018_attempted.sh。")
    (ROOT / rules["outputs"]["report"]).write_text("\n".join(L) + "\n")
    print(pd.DataFrame(tab).to_string(index=False))
    print(inv.to_string(index=False))
    print(dec[cols].to_string(index=False))
    print(verdicts)


if __name__ == "__main__":
    main()
