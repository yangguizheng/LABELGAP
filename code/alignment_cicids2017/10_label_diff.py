#!/usr/bin/env python
"""Three-way label diff: original vs Engelen(WTMC2021) vs Liu(CNS2022) CIC-IDS2017.

Join key: (day, Flow ID, Flow Duration, occurrence index) — occurrence index is
the cumcount within identical (Flow ID, Duration) after chronological ordering,
so repeated identical flows (e.g. zero-duration scans) align by order.

Outputs (reports/):
  t1_class_counts_three_versions.csv   per-class totals, all versions
  t2_engelen_vs_liu_transitions.csv    flow-aligned label transition counts E<->L
  t3_original_vs_engelen_transitions.csv  same for O<->E (matched subset)
  t4_original_vs_liu_transitions.csv      same for O<->L
  coverage_stats.txt                   join coverage + unlabeled-row accounting
  natural_noise_mask.parquet           per original row: corrected labels where matched
"""
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.environ.get("LABELGAP_WORK_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROC = os.path.join(BASE, "data", "processed")
REP = os.path.join(BASE, "reports")
os.makedirs(REP, exist_ok=True)

ORIG_DAY_FILES = {  # chronological order within day
    "monday": ["Monday-WorkingHours.pcap_ISCX.parquet"],
    "tuesday": ["Tuesday-WorkingHours.pcap_ISCX.parquet"],
    "wednesday": ["Wednesday-workingHours.pcap_ISCX.parquet"],
    "thursday": [
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.parquet",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.parquet",
    ],
    "friday": [
        "Friday-WorkingHours-Morning.pcap_ISCX.parquet",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.parquet",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.parquet",
    ],
}
ENG_DAY_FILES = {d: [f"{d.capitalize()}-WorkingHours.parquet"] for d in ORIG_DAY_FILES}
LIU_DAY_FILES = {d: [f"{d}.parquet"] for d in ORIG_DAY_FILES}

CANON = {
    "portscan": "PortScan",
    "bot": "Bot",
    "botnet": "Bot",
    "dos slowloris": "DoS slowloris",
    "dos slowhttptest": "DoS Slowhttptest",
    "dos hulk": "DoS Hulk",
    "dos goldeneye": "DoS GoldenEye",
    "web attack - brute force": "Web Attack - Brute Force",
    "web attack - xss": "Web Attack - XSS",
    "web attack - sql injection": "Web Attack - SQL Injection",
    "infiltration": "Infiltration",
    "infiltration - portscan": "Infiltration-Portscan",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "ddos": "DDoS",
    "heartbleed": "Heartbleed",
    "benign": "BENIGN",
}


def canon_label(s: pd.Series) -> pd.DataFrame:
    """Return DataFrame with columns: base (canonical), attempted (bool), missing (bool)."""
    raw = s.astype("string").str.strip()
    missing = raw.isna() | (raw == "") | (raw.str.lower() == "nan")
    attempted = raw.str.endswith("- Attempted", na=False)
    base = raw.str.replace(r"\s*-\s*Attempted$", "", regex=True).str.strip()
    base_lc = base.str.lower()
    mapped = base_lc.map(CANON)
    unmapped = base[mapped.isna() & ~missing].unique()
    if len(unmapped):
        print(f"    UNMAPPED LABELS: {list(unmapped)[:20]}")
    out = pd.DataFrame({"base": mapped, "attempted": attempted.fillna(False), "missing": missing})
    out.loc[missing, "base"] = "UNLABELED"
    return out


def load_version(proc_dir: str, day_files: dict, version: str) -> pd.DataFrame:
    parts = []
    for day, files in day_files.items():
        for order, fname in enumerate(files):
            path = os.path.join(proc_dir, fname)
            df = pd.read_parquet(path, columns=["Flow ID", "Flow Duration", "Label", "row_idx"])
            df["day"] = day
            df["file_order"] = order
            parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    lab = canon_label(df["Label"])
    df[f"label_{version}"] = lab["base"].where(~lab["attempted"], lab["base"] + " [ATT]")
    df[f"base_{version}"] = lab["base"]
    df[f"att_{version}"] = lab["attempted"]
    df = df.sort_values(["day", "file_order", "row_idx"], kind="stable").reset_index(drop=True)
    df["dur"] = pd.to_numeric(df["Flow Duration"], errors="coerce").astype("Int64")
    df["occ"] = df.groupby(["day", "Flow ID", "dur"], dropna=False, sort=False).cumcount()
    keep = ["day", "Flow ID", "dur", "occ", f"label_{version}", f"base_{version}", f"att_{version}", "row_idx"]
    return df[keep].rename(columns={"row_idx": f"row_idx_{version}"})


def transitions(df: pd.DataFrame, a: str, b: str, path: str) -> pd.DataFrame:
    t = df.groupby([f"label_{a}", f"label_{b}"], observed=True).size().reset_index(name="n")
    t = t.sort_values("n", ascending=False)
    t.to_csv(path, index=False)
    return t


def main() -> None:
    print("== loading three versions ==")
    orig = load_version(os.path.join(PROC, "original"), ORIG_DAY_FILES, "orig")
    eng = load_version(os.path.join(PROC, "engelen"), ENG_DAY_FILES, "eng")
    liu = load_version(os.path.join(PROC, "liu"), LIU_DAY_FILES, "liu")
    print(f"orig={len(orig):,}  eng={len(eng):,}  liu={len(liu):,}")

    lines = []

    def say(msg: str) -> None:
        print(msg)
        lines.append(msg)

    # T1: class counts
    t1 = pd.concat(
        [
            orig["label_orig"].value_counts().rename("original"),
            eng["label_eng"].value_counts().rename("engelen"),
            liu["label_liu"].value_counts().rename("liu"),
        ],
        axis=1,
    ).fillna(0).astype(int)
    t1.to_csv(os.path.join(REP, "t1_class_counts_three_versions.csv"))
    say("\n== T1 class counts ==\n" + t1.to_string())

    key = ["day", "Flow ID", "dur", "occ"]

    # E <-> L join
    el = eng.merge(liu, on=key, how="outer", indicator=True)
    n_both = int((el["_merge"] == "both").sum())
    say(f"\n== Engelen<->Liu join: matched {n_both:,} / eng {len(eng):,} ({n_both/len(eng):.2%}) / liu {len(liu):,} ({n_both/len(liu):.2%})")
    elb = el[el["_merge"] == "both"]
    dis = elb[elb["label_eng"] != elb["label_liu"]]
    say(f"   label disagreement on matched: {len(dis):,} ({len(dis)/n_both:.2%})")
    dis_base = elb[elb["base_eng"] != elb["base_liu"]]
    say(f"   base-label (ignoring Attempted flag) disagreement: {len(dis_base):,} ({len(dis_base)/n_both:.2%})")
    transitions(elb, "eng", "liu", os.path.join(REP, "t2_engelen_vs_liu_transitions.csv"))

    # O <-> E join
    oe = orig.merge(eng, on=key, how="outer", indicator=True)
    n_both_oe = int((oe["_merge"] == "both").sum())
    n_orig_labeled = int((orig["label_orig"] != "UNLABELED").sum())
    say(f"\n== Original<->Engelen join: matched {n_both_oe:,} / orig {len(orig):,} ({n_both_oe/len(orig):.2%})")
    say(f"   orig labeled rows: {n_orig_labeled:,}; unlabeled: {len(orig)-n_orig_labeled:,}")
    oeb = oe[(oe["_merge"] == "both") & (oe["label_orig"] != "UNLABELED")]
    dis_oe = oeb[oeb["base_orig"] != oeb["base_eng"]]
    say(f"   matched-and-labeled: {len(oeb):,}; base-label changes: {len(dis_oe):,} ({len(dis_oe)/max(len(oeb),1):.2%})")
    transitions(oeb, "orig", "eng", os.path.join(REP, "t3_original_vs_engelen_transitions.csv"))

    # O <-> L join
    ol = orig.merge(liu, on=key, how="outer", indicator=True)
    n_both_ol = int((ol["_merge"] == "both").sum())
    say(f"\n== Original<->Liu join: matched {n_both_ol:,} / orig {len(orig):,} ({n_both_ol/len(orig):.2%})")
    olb = ol[(ol["_merge"] == "both") & (ol["label_orig"] != "UNLABELED")]
    dis_ol = olb[olb["base_orig"] != olb["base_liu"]]
    say(f"   matched-and-labeled: {len(olb):,}; base-label changes: {len(dis_ol):,} ({len(dis_ol)/max(len(olb),1):.2%})")
    transitions(olb, "orig", "liu", os.path.join(REP, "t4_original_vs_liu_transitions.csv"))

    # natural noise mask: original rows + their corrected labels where matched
    mask = orig[key + ["label_orig", "base_orig", "row_idx_orig"]].copy()
    mask = mask.merge(oe[key + ["base_eng", "att_eng"]].dropna(subset=["base_eng"]), on=key, how="left")
    mask = mask.merge(ol[key + ["base_liu", "att_liu"]].dropna(subset=["base_liu"]), on=key, how="left")
    mask.to_parquet(os.path.join(REP, "natural_noise_mask.parquet"), index=False)
    say(f"\nnatural_noise_mask.parquet written: {len(mask):,} rows")

    with open(os.path.join(REP, "coverage_stats.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
