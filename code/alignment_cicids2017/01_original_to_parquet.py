#!/usr/bin/env python
"""Mirror CIC-IDS2017 original GeneratedLabelledFlows CSVs into parquet, byte-faithful.

No cleaning. Every row kept, provenance columns added (source_file, row_idx).
Known quirks handled: latin-1 encoding (0x96 en-dash in web-attack labels),
'Infinity'/'NaN' strings in rate columns, whitespace-padded column names,
repeated header rows.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

RATE_COLS = ["Flow Bytes/s", "Flow Packets/s"]


def load_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", low_memory=False, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    # repeated header rows appear as data rows whose Label cell is 'Label'
    if "Label" in df.columns:
        n_hdr = int((df["Label"] == "Label").sum())
        if n_hdr:
            df = df[df["Label"] != "Label"].reset_index(drop=True)
            print(f"    dropped {n_hdr} repeated header rows")
    for col in RATE_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].replace({"Infinity": np.inf, "-Infinity": -np.inf, "NaN": np.nan}),
                errors="raise",
            )
    # normalize the en-dash that latin-1 renders as \x96 so all versions agree
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.replace("\x96", "-", regex=False).str.strip()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir containing the 8 *_ISCX.csv files")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if len(csvs) != 8:
        print(f"WARNING: expected 8 csvs, found {len(csvs)}")
    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    for path in csvs:
        name = os.path.basename(path)
        print(f"[{name}]")
        df = load_one(path)
        df["source_file"] = name
        df["row_idx"] = np.arange(len(df), dtype=np.int64)
        out = os.path.join(args.out_dir, name.replace(".csv", ".parquet"))
        df.to_parquet(out, index=False)
        total += len(df)
        print(f"    {len(df):,} rows -> {out}")
        print(df["Label"].value_counts().to_string())
    print(f"\nTOTAL rows: {total:,}")


if __name__ == "__main__":
    sys.exit(main())
