#!/usr/bin/env python
"""Mirror the Engelen (WTMC2021) / Liu (CNS2022) improved CIC-IDS2017 CSVs into parquet.

Same discipline as 01: no cleaning, all rows kept, provenance columns added.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if len(csvs) != 5:
        print(f"WARNING: expected 5 csvs, found {len(csvs)}")
    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    for path in csvs:
        name = os.path.basename(path)
        print(f"[{name}]")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object and col not in ("Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Attempted Category"):
                df[col] = pd.to_numeric(
                    df[col].replace({"Infinity": np.inf, "-Infinity": -np.inf, "NaN": np.nan}),
                    errors="coerce",
                )
        df["source_file"] = name
        df["row_idx"] = np.arange(len(df), dtype=np.int64)
        out = os.path.join(args.out_dir, os.path.splitext(name)[0] + ".parquet")
        df.to_parquet(out, index=False)
        total += len(df)
        print(f"    {len(df):,} rows -> {out}")
        print(df["Label"].value_counts().to_string())
        if "Attempted Category" in df.columns:
            print("    Attempted Category:", dict(df["Attempted Category"].value_counts().head(8)))
    print(f"\nTOTAL rows: {total:,}")


if __name__ == "__main__":
    sys.exit(main())
