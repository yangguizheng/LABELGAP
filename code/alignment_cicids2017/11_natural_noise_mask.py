#!/usr/bin/env python
"""Build a high-coverage natural-noise mask for the original CIC-IDS2017 rows.

Strategy (documented, conservative):
  Tier 1: exact (day, Flow ID, duration, occurrence) match  -> per-flow label
  Tier 2: (day, Flow ID) groups whose corrected base-labels are unanimous
          -> propagate that label to all original rows of the group
  Rows in neither tier stay uncovered (reported, never guessed).

Output: reports/natural_noise_mask_v2.parquet + coverage/per-class noise table.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

diff_mod = import_module("10_label_diff")  # reuse loaders

BASE = os.environ.get("LABELGAP_WORK_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROC = os.path.join(BASE, "data", "processed")
REP = os.path.join(BASE, "reports")


def unanimous_map(df: pd.DataFrame, version: str) -> pd.DataFrame:
    g = df.groupby(["day", "Flow ID"], sort=False)[f"base_{version}"].agg(["nunique", "first", "size"])
    una = g[g["nunique"] == 1].reset_index()
    return una.rename(columns={"first": f"tier2_{version}"})[["day", "Flow ID", f"tier2_{version}"]]


def main() -> None:
    orig = diff_mod.load_version(os.path.join(PROC, "original"), diff_mod.ORIG_DAY_FILES, "orig")
    eng = diff_mod.load_version(os.path.join(PROC, "engelen"), diff_mod.ENG_DAY_FILES, "eng")
    liu = diff_mod.load_version(os.path.join(PROC, "liu"), diff_mod.LIU_DAY_FILES, "liu")

    key = ["day", "Flow ID", "dur", "occ"]
    mask = orig[key + ["base_orig", "att_orig", "row_idx_orig"]].copy()

    for version, corr in (("eng", eng), ("liu", liu)):
        t1 = corr[key + [f"base_{version}", f"att_{version}"]].rename(
            columns={f"base_{version}": f"tier1_{version}", f"att_{version}": f"tier1_att_{version}"}
        )
        mask = mask.merge(t1, on=key, how="left")
        mask = mask.merge(unanimous_map(corr, version), on=["day", "Flow ID"], how="left")
        mask[f"corrected_{version}"] = mask[f"tier1_{version}"].fillna(mask[f"tier2_{version}"])
        mask[f"tier_{version}"] = np.where(
            mask[f"tier1_{version}"].notna(), 1, np.where(mask[f"tier2_{version}"].notna(), 2, 0)
        )

    lines = []

    def say(msg: str) -> None:
        print(msg)
        lines.append(msg)

    labeled = mask[mask["base_orig"] != "UNLABELED"]
    say(f"original labeled rows: {len(labeled):,}")
    for v in ("eng", "liu"):
        cov = labeled[f"corrected_{v}"].notna()
        t1n = int((labeled[f"tier_{v}"] == 1).sum())
        t2n = int((labeled[f"tier_{v}"] == 2).sum())
        say(f"[{v}] coverage {cov.mean():.2%} (tier1 {t1n:,}, tier2 {t2n:,}, uncovered {int((~cov).sum()):,})")
        sub = labeled[cov]
        chg = sub["base_orig"] != sub[f"corrected_{v}"]
        say(f"[{v}] base-label change rate on covered: {chg.mean():.2%} ({int(chg.sum()):,} rows)")
        per = (
            sub.assign(changed=chg)
            .groupby("base_orig", observed=True)
            .agg(n=("changed", "size"), changed=("changed", "sum"))
        )
        per["rate"] = per["changed"] / per["n"]
        per = per.sort_values("n", ascending=False)
        say(f"[{v}] per-class change rate:\n{per.to_string()}")
        per.to_csv(os.path.join(REP, f"noise_per_class_{v}.csv"))

    both = labeled[labeled["corrected_eng"].notna() & labeled["corrected_liu"].notna()]
    agree = both["corrected_eng"] == both["corrected_liu"]
    say(f"\nrows covered by BOTH corrected versions: {len(both):,}; corrected-label agreement {agree.mean():.2%}")
    cons = both[agree & (both["base_orig"] != both["corrected_eng"])]
    say(f"consensus natural-noise rows (both corrections agree, differ from original): {len(cons):,} "
        f"({len(cons)/len(both):.2%} of dual-covered)")
    trans = cons.groupby(["base_orig", "corrected_eng"], observed=True).size().sort_values(ascending=False)
    say(f"top consensus transitions:\n{trans.head(20).to_string()}")
    trans.rename("n").reset_index().to_csv(os.path.join(REP, "consensus_noise_transitions.csv"), index=False)

    keep = key + ["base_orig", "att_orig", "row_idx_orig",
                  "corrected_eng", "tier_eng", "corrected_liu", "tier_liu"]
    mask[keep].to_parquet(os.path.join(REP, "natural_noise_mask_v2.parquet"), index=False)
    say(f"\nwritten reports/natural_noise_mask_v2.parquet ({len(mask):,} rows)")
    with open(os.path.join(REP, "natural_noise_coverage.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
