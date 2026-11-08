#!/usr/bin/env python
"""LABELGAP revision, round 1 (descriptive): label-disagreement anatomy of CSE-CIC-IDS2018 under convention K
and the Stage-I convention, alignment coverage of both datasets by day and original label, the offset search
repeated on 21 February 2018, the CIC-IDS2017 anatomy by alignment tier, and the attempted status of the omitted
CIC-IDS2017 flows.
Plan: configs/revision_round1.yaml (committed before this script is run).
Outputs: results/revision_round1/anatomy_*.csv, anatomy_summary.json
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

from directionA2018_prepare import CLEAN_MAP  # noqa: E402
from directionA_2018_alignment import DAYS, ORIG_FILE, corrected, key_hash, original, shifted  # noqa: E402
from src.utils import load_yaml, save_json  # noqa: E402

EXCLUDED = "Wednesday-21-02-2018"
SUFFIX = " - Attempted"
CLEAN_MAP_2017 = {"Infiltration-Portscan": "PortScan"}


def stage1(liu: str, att: bool) -> str:
    """Stage-I map (directionA2018_prepare.map_clean): attempted flows are benign."""
    if att or liu.endswith("Attempted"):
        return "Benign"
    return "Infilteration" if liu.startswith("Infiltration") else CLEAN_MAP[liu]


def conv_k(liu: str, att: bool) -> str:
    """Convention K: attempted flows keep the base attack class of their corrected label."""
    base = liu[: -len(SUFFIX)] if liu.endswith(SUFFIX) else liu
    return "Infilteration" if base.startswith("Infiltration") else CLEAN_MAP[base]


def etype(orig: str, clean: str, benign: str) -> str:
    if orig == clean:
        return "agree"
    if orig == benign:
        return "omitted"
    return "opposite" if clean == benign else "attack_type"


def main() -> None:
    plan = load_yaml(ROOT / "configs/revision_round1.yaml")
    out = ROOT / plan["outputs"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    S: dict = {}

    # ---- CSE-CIC-IDS2018 anatomy (all aligned flows) -------------------------------------------------------
    tr_all = pd.read_csv(ROOT / "results/directionA_2018/label_transitions.csv")
    tr_all["attempted"] = tr_all["attempted"].astype(str).str.lower().eq("true")
    tr = tr_all[tr_all["day"] != EXCLUDED].copy()
    for conv, fn in (("K", conv_k), ("stage1", stage1)):
        tr[f"clean_{conv}"] = [fn(a, b) for a, b in zip(tr["liu"], tr["attempted"])]
        tr[f"type_{conv}"] = [etype(o, c, "Benign") for o, c in zip(tr["orig"], tr[f"clean_{conv}"])]
    n_all = int(tr["n"].sum())
    S["cicids2018"] = {"aligned_flows": n_all, "tier_A_share": float(tr.loc[tr["tier"] == "A", "n"].sum() / n_all)}
    for conv in ("K", "stage1"):
        t = tr.groupby(f"type_{conv}")["n"].sum()
        dis = int(t.drop("agree", errors="ignore").sum())
        inf = tr[f"clean_{conv}"] == "Infilteration"
        S["cicids2018"][conv] = {
            "disagreeing": dis, "disagreeing_share_of_aligned": dis / n_all,
            **{k: int(t.get(k, 0)) for k in ("omitted", "opposite", "attack_type")},
            **{f"{k}_share_of_disagreeing": float(t.get(k, 0) / dis) for k in ("omitted", "opposite", "attack_type")},
            "true_infiltration": int(tr.loc[inf, "n"].sum()),
            "true_infiltration_labeled_infiltration": int(tr.loc[inf & (tr["orig"] == "Infilteration"), "n"].sum()),
            "true_infiltration_labeled_benign": int(tr.loc[inf & (tr["orig"] == "Benign"), "n"].sum()),
            "attack_labeled_flows": int(tr.loc[tr["orig"] != "Benign", "n"].sum())}
        S["cicids2018"][conv]["infiltration_labeled_benign_share"] = (
            S["cicids2018"][conv]["true_infiltration_labeled_benign"] / S["cicids2018"][conv]["true_infiltration"])
        (tr[tr[f"type_{conv}"] != "agree"]
         .groupby([f"type_{conv}", "orig", f"clean_{conv}", "liu", "attempted"])["n"].sum().rename("flows")
         .reset_index().sort_values("flows", ascending=False)
         .to_csv(out / f"anatomy_2018_{conv}_disagreements.csv", index=False))
        (tr[tr[f"type_{conv}"] == "omitted"].groupby(["day", "tier", "liu"])["n"].sum().rename("flows")
         .reset_index().to_csv(out / f"anatomy_2018_{conv}_omitted_by_day_tier.csv", index=False))
        om = tr[tr[f"type_{conv}"] == "omitted"]
        S["cicids2018"][conv]["omitted_by_tier"] = {k: int(v) for k, v in om.groupby("tier")["n"].sum().items()}
        S["cicids2018"][conv]["omitted_by_day"] = {k: int(v) for k, v in om.groupby("day")["n"].sum().items()}
    # the 106-flow residual of the Stage-I opposite direction (S8)
    op = tr[tr["type_stage1"] == "opposite"]
    rest = op[~((op["orig"] == "Infilteration") & (op["liu"] == "BENIGN")) & ~op["attempted"]]
    S["cicids2018"]["stage1_opposite_residual"] = {
        "benign_labeled_infiltration": int(op.loc[(op["orig"] == "Infilteration") & (op["liu"] == "BENIGN"), "n"].sum()),
        "attempted": int(op.loc[op["attempted"], "n"].sum()), "other": int(rest["n"].sum()),
        "other_by_label": {f"{a} -> {b}": int(v) for (a, b), v in rest.groupby(["orig", "liu"])["n"].sum().items()}}
    print(f"[2018] anatomy done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- CSE-CIC-IDS2018 coverage by day and original label (R8) ---------------------------------------------
    align = load_yaml(ROOT / "configs/direction_a_2018_alignment.yaml")
    by_day = pd.read_csv(ROOT / "results/directionA_2018/alignment_by_day.csv")
    rows = []
    for day in DAYS:
        o = pd.read_csv(Path(align["data"]["original_dir"]) / ORIG_FILE[day], usecols=["Dst Port", "Label"],
                        dtype=str, engine="pyarrow")
        o = o[o["Dst Port"] != "Dst Port"]
        assert len(o) == int(by_day.loc[by_day["day"] == day, "original_rows"].iloc[0]), day
        cnt = o["Label"].str.strip().value_counts()
        m = tr_all[tr_all["day"] == day]
        for lab, n in cnt.items():
            mm = m[m["orig"] == lab]
            rows.append({"day": day, "original_label": lab, "original_flows": int(n),
                         "aligned_flows": int(mm["n"].sum()), "tier_A": int(mm.loc[mm["tier"] == "A", "n"].sum())})
    cov = pd.DataFrame(rows)
    cov["coverage"] = cov["aligned_flows"] / cov["original_flows"]
    cov.to_csv(out / "coverage_2018_by_day_label.csv", index=False, float_format="%.6f")
    lab = cov.groupby("original_label")[["original_flows", "aligned_flows", "tier_A"]].sum()
    lab["coverage"] = lab["aligned_flows"] / lab["original_flows"]
    lab.reset_index().to_csv(out / "coverage_2018_by_label.csv", index=False, float_format="%.6f")
    print(f"[2018] coverage done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- 21 February: the Stage-I offset search on that day alone (R8) -----------------------------------------
    o, liu = original(align, EXCLUDED), corrected(align, EXCLUDED)
    sample = o.sample(n=min(300_000, len(o)), random_state=2026)
    hs = key_hash(sample)
    hs = hs[hs != 0]
    curve = [{"mode": mode, "hours": h, "match_rate": float(np.isin(hs, key_hash(shifted(liu, h, mode))).mean())}
             for mode in ("floor", "round") for h in range(-12, 13)]
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "feb21_offset_search.csv", index=False, float_format="%.6f")
    notime = ["dport", "proto", "dur", "fp", "bp", "fb", "bb"]

    def notime_rate(orig_df: pd.DataFrame, liu_df: pd.DataFrame) -> float:
        a = pd.util.hash_pandas_object(orig_df[notime].fillna(-1).astype("int64"), index=False).to_numpy()
        b = pd.util.hash_pandas_object(liu_df[notime].fillna(-1).astype("int64"), index=False).to_numpy()
        return float(np.isin(a, b).mean())

    ref_day = "Thursday-22-02-2018"
    o2, l2 = original(align, ref_day), corrected(align, ref_day)
    best = curve.loc[curve["match_rate"].idxmax()]
    S["feb21"] = {"best": {"mode": str(best["mode"]), "hours": int(best["hours"]), "match_rate": float(best["match_rate"])},
                  "match_rate_at_stage1_offset": float(curve[(curve["mode"] == "floor") & (curve["hours"] == -4)]
                                                       ["match_rate"].iloc[0]),
                  "time_free_match_rate": notime_rate(sample, liu),
                  "time_free_match_rate_reference_day": notime_rate(o2.sample(n=300_000, random_state=2026), l2),
                  "reference_day": ref_day, "original_rows": int(len(o)), "corrected_rows": int(len(liu)),
                  "original_labels": {k: int(v) for k, v in o["label"].str.strip().value_counts().items()}}
    del o, liu, o2, l2
    print(f"[2018] 21-Feb diagnostic done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- CIC-IDS2017: coverage, anatomy by tier (R8, R10) -----------------------------------------------------------
    ca = load_yaml(ROOT / "configs/directionA_common.yaml")
    mask = pd.read_parquet(ca["dataset"]["noise_mask"])
    noisy = mask["base_orig"].astype(str)
    clean = mask["corrected_liu"].astype(object).replace(CLEAN_MAP_2017)
    tier = mask["tier_liu"].to_numpy()
    labeled = (noisy != "UNLABELED").to_numpy()
    covered = labeled & (tier > 0)
    c17 = (pd.DataFrame({"day": mask["day"].astype(str), "original_label": noisy, "tier": tier})[labeled]
           .groupby(["day", "original_label"]).agg(original_flows=("tier", "size"),
                                                   aligned_flows=("tier", lambda s: int((s > 0).sum())),
                                                   tier_1=("tier", lambda s: int((s == 1).sum()))).reset_index())
    c17["coverage"] = c17["aligned_flows"] / c17["original_flows"]
    c17.to_csv(out / "coverage_2017_by_day_label.csv", index=False, float_format="%.6f")
    lab17 = c17.groupby("original_label")[["original_flows", "aligned_flows", "tier_1"]].sum()
    lab17["coverage"] = lab17["aligned_flows"] / lab17["original_flows"]
    lab17.reset_index().to_csv(out / "coverage_2017_by_label.csv", index=False, float_format="%.6f")
    a17 = pd.DataFrame({"tier": tier[covered], "orig": noisy[covered].to_numpy(),
                        "clean": clean[covered].astype(str).to_numpy()})
    a17["type"] = [etype(o, c, "BENIGN") for o, c in zip(a17["orig"], a17["clean"])]
    by_tier = a17.groupby(["tier", "type"]).size().rename("flows").reset_index()
    by_tier.to_csv(out / "anatomy_2017_by_tier.csv", index=False)
    (a17[a17["type"] == "omitted"].groupby(["tier", "clean"]).size().rename("flows").reset_index()
     .to_csv(out / "anatomy_2017_omitted_by_tier_class.csv", index=False))
    S["cicids2017"] = {"labeled_flows": int(labeled.sum()), "aligned_flows": int(covered.sum()),
                       "coverage": float(covered.sum() / labeled.sum()),
                       "tier_1_share": float((tier[covered] == 1).mean()),
                       "by_tier": {f"tier{t}_{k}": int(v) for t, k, v in by_tier.itertuples(index=False)}}
    print(f"[2017] coverage and anatomy by tier done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- CIC-IDS2017: attempted status of the omitted flows (R9) ------------------------------------------------------
    sys.path.insert(0, str(ROOT / "alignment_cicids2017"))
    diff = import_module("10_label_diff")
    liu17 = diff.load_version(str(Path(ca["dataset"]["original_parquet_dir"]).parent / "liu"), diff.LIU_DAY_FILES, "liu")
    key = ["day", "Flow ID", "dur", "occ"]
    att1 = mask[key].merge(liu17[key + ["att_liu"]], on=key, how="left")["att_liu"]
    assert len(att1) == len(mask)
    share = liu17.groupby(["day", "Flow ID"])["att_liu"].mean().rename("share").reset_index()
    att2 = mask[["day", "Flow ID"]].merge(share, on=["day", "Flow ID"], how="left")["share"]
    assert len(att2) == len(mask)
    status = np.where(tier == 1, np.where(att1.astype(float).to_numpy() == 1, "attempted", "not attempted"),
                      np.where(att2.to_numpy() == 1, "attempted",
                               np.where(att2.to_numpy() == 0, "not attempted", "mixed")))
    om17 = covered & (noisy == "BENIGN").to_numpy() & (clean.astype(str) != "BENIGN").to_numpy()
    t9 = (pd.DataFrame({"tier": tier[om17], "class": clean[om17].astype(str).to_numpy(), "status": status[om17]})
          .groupby(["tier", "class", "status"]).size().rename("flows").reset_index())
    t9.to_csv(out / "attempted_2017_omitted.csv", index=False)
    S["cicids2017"]["omitted"] = int(om17.sum())
    S["cicids2017"]["omitted_attempted_status"] = {k: int(v) for k, v in t9.groupby("status")["flows"].sum().items()}
    S["cicids2017"]["omitted_attempted_status_tier1"] = {
        k: int(v) for k, v in t9[t9["tier"] == 1].groupby("status")["flows"].sum().items()}
    S["minutes"] = (time.perf_counter() - t0) / 60
    save_json(S, out / "anatomy_summary.json")
    print(pd.Series({k: v for k, v in S["cicids2018"]["K"].items() if not isinstance(v, dict)}).to_string())
    print(S["cicids2018"]["stage1_opposite_residual"])
    print(S["feb21"]["best"], S["feb21"]["match_rate_at_stage1_offset"], S["feb21"]["time_free_match_rate"],
          S["feb21"]["time_free_match_rate_reference_day"])
    print(S["cicids2017"])


if __name__ == "__main__":
    main()
