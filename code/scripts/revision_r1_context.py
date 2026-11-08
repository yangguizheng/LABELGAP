#!/usr/bin/env python
"""LABELGAP revision, round 1 (descriptive): context rules re-scored under convention K (attempted flows are
attacks of their base class) in both datasets; CSE-CIC-IDS2018 also on tier-A candidates and on tier A only
(candidates and evidence), with the tier and address composition of the omitted flows, the full decomposition of
the cleaned rule's false flags, and the scan-host count behind 24,615 versus 24,649.
Reuses the A-D4 rules and the saved A-D5 Confident Learning flags; the rebuilt frames must reproduce the A-D4
exposure-rule counts first.
Plan: configs/revision_round1.yaml (committed before this script is run).
Outputs: results/revision_round1/context_*.csv, context_summary.json
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

from directionA_2018_alignment import DAYS, LIU_COLS, NUM, ORIG_COLS, ORIG_FILE, key_hash, shifted  # noqa: E402
from directionA_d4_context import flag, score  # noqa: E402
from directionA_d5_evidence_cleaning import flag_exposed, host_trust  # noqa: E402
from directionA_prepare import load_original  # noqa: E402
from src.utils import load_json, load_yaml, save_json  # noqa: E402

D4 = load_yaml(ROOT / "configs/direction_a_d4_context.yaml")
D5 = load_yaml(ROOT / "configs/direction_a_d5_evidence_cleaning.yaml")
REF = pd.read_csv(ROOT / "results/directionA_d4/context_rules.csv")


def frame2018() -> pd.DataFrame:
    """The A-D5 CSE-CIC-IDS2018 frame (same rows and order), without features, plus tier and address unanimity."""
    align = load_yaml(ROOT / "configs/direction_a_2018_alignment.yaml")
    common = load_yaml(ROOT / "configs/directionA2018_common.yaml")
    off = load_json(ROOT / "results/directionA_2018/summary.json")
    cols = dict(LIU_COLS, **{"Src IP": "src", "Dst IP": "dst"})
    parts = []
    for day in DAYS:
        if day in common["dataset"]["excluded_days"]:
            continue
        o = pd.read_csv(Path(common["dataset"]["original_dir"]) / ORIG_FILE[day], usecols=list(ORIG_COLS),
                        dtype=str, engine="pyarrow")
        o = o[o["Dst Port"] != "Dst Port"].reset_index(drop=True)
        k = o[list(ORIG_COLS)].rename(columns=ORIG_COLS)
        for c in NUM:
            k[c] = pd.to_numeric(k[c], errors="coerce")
        ts = pd.to_datetime(k["ts"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        k["t"] = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        h = key_hash(k)
        liu = pd.read_csv(Path(align["data"]["corrected_dir"]) / f"{day}.csv", usecols=list(cols), dtype=str,
                          engine="pyarrow").rename(columns=cols)
        for c in NUM:
            liu[c] = pd.to_numeric(liu[c], errors="coerce")
        tsl = pd.to_datetime(liu["ts"], format="ISO8601", errors="coerce")
        liu["tf"] = (tsl - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        liu["att"] = pd.to_numeric(liu["att"], errors="coerce").fillna(-1).astype(int)
        liu = shifted(liu.drop(columns="ts"), off["time_offset_hours"], off["rounding"])
        liu["h"] = key_hash(liu)
        grp = liu[liu["h"] != 0].groupby("h").agg(n_lab=("label", "nunique"), n_att=("att", "nunique"),
                                                   lab=("label", "first"), att=("att", "first"),
                                                   src=("src", "first"), dst=("dst", "first"),
                                                   n_l=("label", "size"), n_src=("src", "nunique"),
                                                   n_dst=("dst", "nunique"))
        m = pd.DataFrame({"h": h}).join(grp, on="h")
        m["n_o"] = m["h"].map(pd.Series(h[h != 0]).value_counts())
        ok = (m["lab"].notna() & (m["n_lab"] == 1) & (m["n_att"] == 1)).to_numpy() & (h != 0)
        mm = m[ok]
        parts.append(pd.DataFrame({
            "day": day, "t": pd.to_datetime(k.loc[ok, "t"].to_numpy(), unit="s"),
            "src": mm["src"].to_numpy(), "dst": mm["dst"].to_numpy(),
            "noisy": k.loc[ok, "label"].astype(str).str.strip().to_numpy(),
            "liu": mm["lab"].astype(str).to_numpy(), "att": mm["att"].astype(int).to_numpy(),
            "tier": np.where((mm["n_l"] == 1).to_numpy() & (mm["n_o"] == 1).to_numpy(), "A", "B"),
            "addr_unanimous": ((mm["n_src"] == 1) & (mm["n_dst"] == 1)).to_numpy()}))
        print(f"[2018] {day}: matched {int(ok.sum()):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def run_rules(ds: str, frame: pd.DataFrame, cand: np.ndarray, attack: np.ndarray, issues: np.ndarray,
              subsets: dict, conventions: dict, clean_class: np.ndarray, evidence: np.ndarray | None = None,
              tag: str = "") -> tuple[list, dict]:
    """All rules on the candidates, with attack evidence restricted to `evidence` if given; scored per subset and
    convention (hidden = candidates that are attacks under the convention)."""
    ev = attack if evidence is None else attack & evidence
    cidx = np.where(cand)[0]
    win, min_share = D5["window_minutes"], D5["min_surviving_share"]
    flags = {}
    for rule in D4["rules"]:
        rk = rule.split("_")[0]
        for w in D4["windows_minutes"]:
            flags[(rk, w)] = flag(frame, cidx, ev, rk, w)
    surviving = ev & ~issues
    trust = host_trust(frame, ev, surviving)
    flags[("E2_cleaned", win)] = flag_exposed(frame, cidx, surviving, trust, win, min_share)
    flags[("CL_only", 0)] = issues & cand
    rows = []
    for conv, clean_attack in conventions.items():
        for sub, sel in subsets.items():
            for (rk, w), f in flags.items():
                r = score(ds, sub + tag, sel, cand, cand & clean_attack, clean_attack, f, clean_class, rk, w)
                rows.append({"convention": conv, **r})
    return rows, flags


def main() -> None:
    plan = load_yaml(ROOT / "configs/revision_round1.yaml")
    out = ROOT / plan["outputs"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    S: dict = {}
    rows = []

    # ---- CSE-CIC-IDS2018 ---------------------------------------------------------------------------------------
    fr = frame2018()
    issues = np.load(ROOT / "results/directionA_d5/cl_issues_cicids2018.npy")
    assert len(issues) == len(fr), (len(issues), len(fr))
    cand = fr["noisy"].str.lower().eq("benign").to_numpy()
    attack = ~cand
    is_att = (fr["att"].ne(-1) | fr["liu"].str.endswith("Attempted")).to_numpy()
    not_benign = ~fr["liu"].str.upper().eq("BENIGN").to_numpy()
    conv = {"stage1": not_benign & ~is_att, "K": not_benign}
    everything = np.ones(len(fr), bool)
    tier_a = (fr["tier"] == "A").to_numpy()
    # reproduce A-D4 (V2, 60 min, Stage-I convention) before computing anything new
    f_v2 = flag(fr, np.where(cand)[0], attack, "V2", 60)
    r = score("cicids2018", "all matched (primary)", everything, cand, cand & conv["stage1"], conv["stage1"], f_v2,
              fr["liu"].to_numpy(dtype=object), "V2", 60)
    q = REF[(REF["dataset"] == "cicids2018") & (REF["rule"] == "V2") & (REF["window_min"] == 60)]
    assert len(q) == 1 and all(int(q[c].iloc[0]) == r[c] for c in ("candidates", "hidden", "flagged")), (q, r)
    print(f"[2018] A-D4 reproduced ({time.perf_counter() - t0:.0f}s)", flush=True)
    cls = np.where(is_att, fr["liu"] + " [attempted]", fr["liu"]).astype(object)
    rr, flags18 = run_rules("cicids2018", fr, cand, attack, issues,
                            {"all matched": everything, "tier-A candidates": tier_a}, conv, cls)
    rows += rr
    rr, _ = run_rules("cicids2018", fr, cand, attack, issues, {"tier A only": tier_a}, conv, cls,
                      evidence=tier_a, tag=" (tier-A evidence)")
    rows += rr
    om = {c: cand & v for c, v in conv.items()}
    S["cicids2018"] = {c: {"omitted": int(o.sum()), "omitted_tier_A": int((o & tier_a).sum()),
                           "omitted_tier_B": int((o & ~tier_a).sum()),
                           "omitted_tier_B_non_unanimous_addresses": int(
                               (o & ~tier_a & ~fr["addr_unanimous"].to_numpy()).sum())}
                       for c, o in om.items()}
    S["cicids2018"]["tier_B_rows"] = int((~tier_a).sum())
    S["cicids2018"]["tier_B_non_unanimous_addresses"] = int((~tier_a & ~fr["addr_unanimous"].to_numpy()).sum())
    # false flags of the cleaned rule, all matched rows (S13)
    win = D5["window_minutes"]
    for c, ca in conv.items():
        ff = flags18[("E2_cleaned", win)] & cand & ~ca
        d = fr.loc[ff, ["day", "src"]].value_counts().rename("false_flags").reset_index()
        d["internal"] = d["src"].astype(str).str.startswith("172.31.")
        d.to_csv(out / f"context_2018_cleaned_false_flags_{c}.csv", index=False)
        S["cicids2018"][c]["cleaned_rule_false_flags"] = int(ff.sum())
    # 28 February, host 172.31.69.24: benign-labeled flows by corrected label (S7)
    h24 = (fr["day"] == "Wednesday-28-02-2018") & (fr["src"] == "172.31.69.24") & pd.Series(cand)
    S["cicids2018"]["host_172_31_69_24_benign_labeled_by_corrected"] = {
        k: int(v) for k, v in fr.loc[h24, "liu"].value_counts().items()}
    om28 = (fr["day"] == "Wednesday-28-02-2018") & pd.Series(om["stage1"])
    S["cicids2018"]["omitted_28feb_by_source_host"] = {k: int(v) for k, v in fr.loc[om28, "src"].value_counts().head(5).items()}
    del fr
    print(f"[2018] rules done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- CIC-IDS2017 --------------------------------------------------------------------------------------------
    ca = load_yaml(ROOT / "configs/directionA_common.yaml")
    orig = load_original(ca)
    mask = pd.read_parquet(ca["dataset"]["noise_mask"])
    assert len(orig) == len(mask) and (orig["row_idx"].to_numpy() == mask["row_idx_orig"].to_numpy()).all()
    ts = pd.to_datetime(orig["Timestamp"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    ts = ts.where(ts.dt.hour >= 8, ts + pd.Timedelta(hours=12))        # 12-h clock without AM/PM (as in A-D4)
    fr17 = pd.DataFrame({"t": ts, "src": orig["Source IP"].astype(str).str.strip(),
                         "dst": orig["Destination IP"].astype(str).str.strip(), "day": orig["day"].to_numpy()})
    del orig
    noisy = mask["base_orig"].astype(str).to_numpy()
    clean = mask["corrected_liu"].astype(object).to_numpy()
    tier = mask["tier_liu"].to_numpy()
    sys.path.insert(0, str(ROOT / "alignment_cicids2017"))
    diff = import_module("10_label_diff")
    liu = diff.load_version(str(Path(ca["dataset"]["original_parquet_dir"]).parent / "liu"), diff.LIU_DAY_FILES, "liu")
    key = ["day", "Flow ID", "dur", "occ"]
    att = mask[key].merge(liu[key + ["att_liu"]], on=key, how="left")["att_liu"].astype(float).to_numpy()
    del liu
    covered = (tier > 0) & (noisy != "UNLABELED")
    cand17 = covered & (noisy == "BENIGN")
    attack17 = (noisy != "UNLABELED") & (noisy != "BENIGN")
    base_attack = covered & (clean != "BENIGN") & pd.notna(clean)
    conv17 = {"stage1": base_attack & ~(att == 1), "K": base_attack}
    issues17 = np.load(ROOT / "results/directionA_d5/cl_issues_cicids2017.npy")
    assert len(issues17) == len(fr17)
    f_v2 = flag(fr17, np.where(cand17)[0], attack17, "V2", 60)
    for sub, sel in (("tier-1 (primary)", tier == 1), ("all covered", covered)):
        r = score("cicids2017", sub, sel, cand17, cand17 & conv17["stage1"] & sel, conv17["stage1"], f_v2,
                  np.asarray(clean, dtype=object), "V2", 60)
        q = REF[(REF["dataset"] == "cicids2017") & (REF["subset"] == sub) & (REF["rule"] == "V2")
                & (REF["window_min"] == 60)]
        assert len(q) == 1 and all(int(q[c].iloc[0]) == r[c] for c in ("candidates", "hidden", "flagged")), (sub, q, r)
    print(f"[2017] A-D4 reproduced ({time.perf_counter() - t0:.0f}s)", flush=True)
    cls17 = np.where(att == 1, pd.Series(clean).astype(str) + " [attempted]", pd.Series(clean).astype(str)).astype(object)
    rr, _ = run_rules("cicids2017", fr17, cand17, attack17, issues17,
                      {"tier 1": tier == 1, "all aligned": covered}, conv17, cls17)
    rows += rr
    S["cicids2017"] = {c: {"omitted_tier1": int((cand17 & v & (tier == 1)).sum()),
                           "omitted_all": int((cand17 & v).sum())} for c, v in conv17.items()}
    res = pd.DataFrame(rows)
    res.to_csv(out / "context_rules.csv", index=False, float_format="%.6f")
    S["minutes"] = (time.perf_counter() - t0) / 60
    save_json(S, out / "context_summary.json")
    show = res[["convention", "dataset", "subset", "rule", "window_min", "candidates", "hidden", "flagged",
                "flagged_share_of_candidates", "precision", "recall_hidden"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(S)


if __name__ == "__main__":
    main()
