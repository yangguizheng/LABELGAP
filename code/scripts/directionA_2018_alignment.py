#!/usr/bin/env python
"""Direction A, second dataset: can original CSE-CIC-IDS2018 rows be paired with
Liu et al.'s corrected flows? Exact matching on an 8-field key (the original files
mostly lack Flow ID / IPs). Criteria: configs/direction_a_2018_alignment.yaml.
Outputs: results/directionA_2018/*, reports/DIRECTION_A_2018_ALIGNMENT.md
(interpretation spliced from reports/direction_a_2018_interpretation.md).
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

from build_failure_report import md_table  # noqa: E402
from src.utils import git_commit, load_yaml, save_json  # noqa: E402

DAYS = ["Wednesday-14-02-2018", "Thursday-15-02-2018", "Friday-16-02-2018", "Tuesday-20-02-2018",
        "Wednesday-21-02-2018", "Thursday-22-02-2018", "Friday-23-02-2018", "Wednesday-28-02-2018",
        "Thursday-01-03-2018", "Friday-02-03-2018"]
ORIG_FILE = {d: f"{d}_TrafficForML_CICFlowMeter.csv" for d in DAYS}
ORIG_FILE["Tuesday-20-02-2018"] = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"  # CIC's file name
ORIG_COLS = {"Dst Port": "dport", "Protocol": "proto", "Timestamp": "ts", "Flow Duration": "dur",
             "Tot Fwd Pkts": "fp", "Tot Bwd Pkts": "bp", "TotLen Fwd Pkts": "fb", "TotLen Bwd Pkts": "bb",
             "Label": "label"}
LIU_COLS = {"Dst Port": "dport", "Protocol": "proto", "Timestamp": "ts", "Flow Duration": "dur",
            "Total Fwd Packet": "fp", "Total Bwd packets": "bp", "Total Length of Fwd Packet": "fb",
            "Total Length of Bwd Packet": "bb", "Label": "label", "Attempted Category": "att"}
KEY = ["dport", "proto", "t", "dur", "fp", "bp", "fb", "bb"]
NUM = ["dport", "proto", "dur", "fp", "bp", "fb", "bb"]


def read(path: Path, cols: dict) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(cols), dtype=str, engine="pyarrow").rename(columns=cols)
    df = df[df["dport"] != "Dst Port"]                     # header lines repeated inside some files
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def original(cfg: dict, day: str) -> pd.DataFrame:
    df = read(Path(cfg["data"]["original_dir"]) / ORIG_FILE[day], ORIG_COLS)
    ts = pd.to_datetime(df["ts"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["t"] = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return df.drop(columns="ts")


def corrected(cfg: dict, day: str) -> pd.DataFrame:
    df = read(Path(cfg["data"]["corrected_dir"]) / f"{day}.csv", LIU_COLS)
    ts = pd.to_datetime(df["ts"], format="ISO8601", errors="coerce")
    df["tf"] = (ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    df["att"] = pd.to_numeric(df["att"], errors="coerce").fillna(-1).astype(int)
    return df.drop(columns="ts")


def key_hash(df: pd.DataFrame) -> np.ndarray:
    ok = df[KEY].notna().all(1)
    h = pd.util.hash_pandas_object(df[KEY].fillna(-1).astype("int64"), index=False).to_numpy()
    return np.where(ok.to_numpy(), h, 0)                   # 0 = incomplete key, never matched


def shifted(liu: pd.DataFrame, hours: int, mode: str) -> pd.DataFrame:
    t = liu["tf"] + hours * 3600
    out = liu.copy()
    out["t"] = (np.floor(t) if mode == "floor" else np.round(t)).astype("Int64")
    return out


def main() -> None:
    cfg = load_yaml(ROOT / "configs/direction_a_2018_alignment.yaml")
    out = ROOT / "results/directionA_2018"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ---- time offset, chosen once on a sample of the first day ------------------------------------
    o = original(cfg, DAYS[0])
    liu = corrected(cfg, DAYS[0])
    sample = o.sample(n=min(300_000, len(o)), random_state=2026)
    hs = key_hash(sample)
    hs = hs[hs != 0]
    curve = []
    for mode in ("floor", "round"):
        for h in range(-12, 13):
            curve.append({"mode": mode, "hours": h,
                          "match_rate": float(np.isin(hs, key_hash(shifted(liu, h, mode))).mean())})
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "offset_search.csv", index=False, float_format="%.6f")
    best = curve.loc[curve["match_rate"].idxmax()]
    mode, hours = str(best["mode"]), int(best["hours"])
    print(f"[offset] best {hours:+d} h ({mode}), sample match rate {best['match_rate']:.4f}; "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- per-day matching ------------------------------------------------------------------------
    by_day, trans, hidden = [], [], []
    for day in DAYS:
        if day != DAYS[0]:
            o, liu = original(cfg, day), corrected(cfg, day)
        liu = shifted(liu, hours, mode)
        o["h"], liu["h"] = key_hash(o), key_hash(liu)
        o_lab = o["label"].astype(str).str.strip()
        n_labelled = int((o_lab != "Label").sum())
        grp = liu[liu["h"] != 0].groupby("h").agg(n_l=("label", "size"), lab=("label", "first"),
                                                   att=("att", "first"),
                                                   n_lab=("label", "nunique"), n_att=("att", "nunique"))
        n_o = o[o["h"] != 0].groupby("h").size().rename("n_o")
        m = o[["h"]].join(grp, on="h").join(n_o, on="h")
        present = m["n_l"].notna() & (o["h"] != 0).to_numpy()
        unanimous = present & (m["n_lab"] == 1) & (m["n_att"] == 1)
        tier_a = unanimous & (m["n_l"] == 1) & (m["n_o"] == 1)
        tier_b = unanimous & ~tier_a
        amb = present & ~unanimous
        matched = tier_a | tier_b
        by_day.append({"day": day, "original_rows": n_labelled, "corrected_rows": int(len(liu)),
                       "incomplete_key": int((o["h"] == 0).sum()), "tier_A": int(tier_a.sum()),
                       "tier_B": int(tier_b.sum()), "ambiguous": int(amb.sum()),
                       "unmatched": int((~present).sum()),
                       "coverage": float(matched.sum() / max(n_labelled, 1))})
        mm = pd.DataFrame({"orig": o_lab[matched].to_numpy(), "liu": m.loc[matched, "lab"].astype(str).to_numpy(),
                           "attempted": (m.loc[matched, "att"].to_numpy() != -1),
                           "tier": np.where(tier_a[matched], "A", "B")})
        t = mm.groupby(["orig", "liu", "attempted", "tier"]).size().rename("n").reset_index()
        trans.append(t.assign(day=day))
        hid = mm[(mm["orig"].str.lower() == "benign") & (mm["liu"].str.upper() != "BENIGN")]
        for (lab, att), g in hid.groupby(["liu", "attempted"]):
            hidden.append({"day": day, "corrected_label": lab, "attempted": bool(att), "rows": int(len(g)),
                           "tier_A_share": float((g["tier"] == "A").mean())})
        print(f"[{day}] original {n_labelled:,} corrected {len(liu):,} tierA {int(tier_a.sum()):,} "
              f"tierB {int(tier_b.sum()):,} ambiguous {int(amb.sum()):,} coverage {by_day[-1]['coverage']:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    by_day = pd.DataFrame(by_day)
    trans = pd.concat(trans, ignore_index=True)
    hidden = pd.DataFrame(hidden)
    by_day.to_csv(out / "alignment_by_day.csv", index=False, float_format="%.6f")
    trans.to_csv(out / "label_transitions.csv", index=False)
    hidden.to_csv(out / "hidden_episodes.csv", index=False, float_format="%.6f")

    # ---- feasibility ----------------------------------------------------------------------------
    f = cfg["feasibility"]
    coverage = float((by_day["tier_A"] + by_day["tier_B"]).sum() / by_day["original_rows"].sum())
    episodes = hidden[(~hidden["attempted"]) & (hidden["rows"] >= f["F2_hidden_episodes"]["min_rows_per_episode"])] \
        if len(hidden) else hidden
    f1 = coverage >= f["F1_coverage"]["min_share_of_original_rows_matched"]
    f2 = len(episodes) >= f["F2_hidden_episodes"]["min_episodes"]
    verdict = "feasible" if (f1 and f2) else ("no_second_episode" if f1 else "infeasible")
    save_json({"time_offset_hours": hours, "rounding": mode, "sample_match_rate": float(best["match_rate"]),
               "coverage": coverage, "F1": bool(f1), "F2": bool(f2), "episodes": int(len(episodes)),
               "verdict": verdict, "minutes": (time.perf_counter() - t0) / 60}, out / "summary.json")

    # ---- report -----------------------------------------------------------------------------------
    L = []
    w = L.append
    w("# DIRECTION A — Second data set: Alignment feasibility of CSE-CIC-IDS2018 original version and Liu’s revised version \n")
    w(f">Protocol selected on 2026-09-26. Generated at {time.strftime('%Y-%m-%d %H:%M')}; code commit `{git_commit()[:7]}`. "
      "The interpretation standard `configs/direction_a_2018_alignment.yaml` is submitted before calculation. It only performs data alignment and does not train the model. \n")
    interp = ROOT / "reports/direction_a_2018_interpretation.md"
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    w("## 2018-a Time offset \n")
    w(f"An integer-hour offset was searched on 300,000 original rows from Wednesday-14-02-2018. The best offset is **{hours:+d} hours (corrected timestamps "
      f"{'truncated' if mode == 'floor' else 'rounded'} to seconds)**, with an exact-match rate of {best['match_rate']:.1%}. "
      "See `offset_search.csv` for the complete curve. \n")
    top = curve.sort_values("match_rate", ascending=False).head(5)
    w(md_table(top.rename(columns={"mode": "Rounding", "hours": "Offset (h)", "match_rate": "Sample match rate"})))
    w("\n## 2018-b Daily matching coverage \n")
    w("tier_A = key appears only once in both versions; tier_B = key is duplicated but revised label matches attempted status; ambiguous = revised label"
      "Inconsistent (excluded); coverage = (tier_A + tier_B) / number of original rows. \n")
    w(md_table(by_day, nd=3))
    w(f"\n **Total Coverage{coverage:.1%}**(F1 threshold{f['F1_coverage']['min_share_of_original_rows_matched']:.0%}).\n")
    w("## 2018-c Omitted attacks among matched flows (original label Benign, corrected label attack)\n")
    if len(hidden):
        hs_ = hidden.sort_values("rows", ascending=False)
        w(md_table(hs_.rename(columns={"day": "Day", "corrected_label": "Corrected label", "attempted": "Attempted",
                                       "rows": "Matched rows", "tier_A_share": "Tier-A share"}), nd=3))
    else:
        w("There is no such transfer in the matched row. \n")
    w(f"\nEpisodes meeting F2 (same day and corrected class, not attempted, at least {f['F2_hidden_episodes']['min_rows_per_episode']} flows): "
      f"**{len(episodes)}** (threshold {f['F2_hidden_episodes']['min_episodes']} episodes). **Feasibility verdict: {verdict}**.\n")
    w("## 2018-d Label transfer of matched rows (top 25 rows) \n")
    tt = trans.groupby(["orig", "liu", "attempted"])["n"].sum().sort_values(ascending=False).head(25).reset_index()
    w(md_table(tt.rename(columns={"orig": "Original label", "liu": "Corrected label", "attempted": "Attempted",
                                  "n": "Rows"}), nd=0))
    w("\n## file \n")
    w("`results/directionA_2018/`:offset_search.csv、alignment_by_day.csv、label_transitions.csv、"
      "hidden_episodes.csv, summary.json. Code: `scripts/directionA_2018_alignment.py`.")
    (ROOT / "reports/DIRECTION_A_2018_ALIGNMENT.md").write_text("\n".join(L) + "\n")
    print(by_day.to_string(index=False))
    print(hidden.sort_values("rows", ascending=False).head(20).to_string(index=False) if len(hidden) else "no hidden")
    print("coverage", round(coverage, 4), "episodes", len(episodes), "verdict", verdict)


if __name__ == "__main__":
    main()
