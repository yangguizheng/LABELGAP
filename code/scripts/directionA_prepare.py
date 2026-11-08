#!/usr/bin/env python
"""Direction A, Steps 3-5: build the CICIDS2017 natural-label-noise dataset.

Original CICFlowMeter features + two labels for the same rows: the original
(noisy) label and the Liu et al. (CNS 2022) corrected (clean) label, aligned
through the CIC-IDS2017 alignment mask. Includes the Step-4 data sanity gate.
Outputs: data/processed/cicids2017_labelnoise_v1.npz, results/directionA/data_*.
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.transforms import signed_log1p  # noqa: E402
from src.utils import load_yaml, save_json  # noqa: E402

ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Protocol", "Timestamp"]
DROP = ID_COLS + ["Label", "source_file", "row_idx", "Fwd Header Length.1"]
CLEAN_MAP = {"Infiltration-Portscan": "PortScan"}


def load_original(cfg: dict) -> pd.DataFrame:
    """Original rows in the exact order used to build the nids_audit mask
    (day, file order, row_idx)."""
    sys.path.insert(0, str(ROOT / "alignment_cicids2017"))
    day_files = import_module("10_label_diff").ORIG_DAY_FILES
    parts = []
    for day, files in day_files.items():
        for order, fname in enumerate(files):
            df = pd.read_parquet(Path(cfg["dataset"]["original_parquet_dir"]) / fname)
            df.columns = [c.strip() for c in df.columns]
            df["day"], df["file_order"] = day, order
            parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["day", "file_order", "row_idx"], kind="stable").reset_index(drop=True)


def main() -> None:
    cfg = load_yaml(ROOT / "configs/directionA_common.yaml")
    gate = load_yaml(ROOT / "configs/direction_a_label_noise.yaml")["data_sanity_gate"]
    out = ROOT / "results/directionA"
    out.mkdir(parents=True, exist_ok=True)

    orig = load_original(cfg)
    mask = pd.read_parquet(cfg["dataset"]["noise_mask"])
    assert len(orig) == len(mask), (len(orig), len(mask))
    assert (orig["day"].to_numpy() == mask["day"].to_numpy()).all()
    assert (orig["row_idx"].to_numpy() == mask["row_idx_orig"].to_numpy()).all()
    print(f"[ok] {len(orig):,} original rows aligned with the noise mask")

    keep = (mask["tier_liu"].to_numpy() > 0) & (mask["base_orig"].to_numpy() != "UNLABELED")
    df = orig.loc[keep].reset_index(drop=True)
    m = mask.loc[keep].reset_index(drop=True)
    y_noisy_name = m["base_orig"].astype(str).to_numpy()
    y_clean_name = m["corrected_liu"].astype(str).replace(CLEAN_MAP).to_numpy()
    feats = [c for c in df.columns if c not in DROP + ["day", "file_order"]]
    assert len(feats) == 77, len(feats)  # 78 MachineLearningCVE columns minus the duplicate
    x = np.array(df[feats].apply(pd.to_numeric, errors="coerce"), dtype=np.float64)  # writable copy
    x[~np.isfinite(x)] = np.nan
    print(f"[ok] {len(df):,} covered rows, {len(feats)} features, NaN/inf cells {int(np.isnan(x).sum()):,}")

    counts = pd.Series(y_clean_name).value_counts()
    classes = sorted(counts.index, key=lambda c: (-counts[c], c))
    assert set(y_noisy_name) <= set(classes), set(y_noisy_name) - set(classes)
    cmap = {c: i for i, c in enumerate(classes)}
    y_clean = np.array([cmap[c] for c in y_clean_name])
    y_noisy = np.array([cmap[c] for c in y_noisy_name])

    # ---- split: grouped by exact feature vector, stratified by clean label ---------------------
    groups = pd.util.hash_pandas_object(pd.DataFrame(x).fillna(-1.2345e300), index=False).to_numpy()
    sp = cfg["split"]
    folds = np.full(len(df), -1)
    cv = StratifiedGroupKFold(n_splits=sp["n_splits"], shuffle=True, random_state=sp["seed"])
    for k, (_, idx) in enumerate(cv.split(x, y_clean, groups)):
        folds[idx] = k
    te, va = folds == sp["test_fold"], folds == sp["val_fold"]
    tr = ~(te | va)
    for a, b in ((tr, te), (tr, va), (va, te)):
        assert np.intersect1d(groups[a], groups[b]).size == 0

    # ---- Step 4: data sanity gate (before any training) -------------------------------------
    rows = []
    g = pd.DataFrame({"h": groups[tr], "clean": y_clean[tr], "noisy": y_noisy[tr]})
    n_clean = g.groupby("h")["clean"].transform("nunique").to_numpy()
    n_noisy = g.groupby("h")["noisy"].transform("nunique").to_numpy()
    for c, name in enumerate(classes):
        sel_c, sel_n = g["clean"].to_numpy() == c, g["noisy"].to_numpy() == c
        affected = (y_clean != y_noisy) & (y_clean == c)
        rows.append({"class": name,
                     "rows_clean": int((y_clean == c).sum()), "rows_noisy": int((y_noisy == c).sum()),
                     "hidden_as_benign": int(((y_clean == c) & (y_noisy == cmap["BENIGN"])).sum()),
                     "noise_rate_clean_class": float(affected.sum() / max((y_clean == c).sum(), 1)),
                     "train_clean": int(sel_c.sum()), "val_clean": int(((y_clean == c) & va).sum()),
                     "test_clean": int(((y_clean == c) & te).sum()),
                     "test_affected": int((affected & te).sum()),
                     "train_conflict_rate_clean_labels": float((n_clean[sel_c] > 1).mean()) if sel_c.any() else np.nan,
                     "train_conflict_rate_noisy_labels": float((n_noisy[sel_n] > 1).mean()) if sel_n.any() else np.nan})
    prof = pd.DataFrame(rows)
    prof["separability_limited"] = prof["train_conflict_rate_clean_labels"] >= gate["conflict_rate_limit"]
    prof.to_csv(out / "data_profile.csv", index=False, float_format="%.6f")
    print(prof.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- Step 5: preprocessing fitted on the training partition only ------------------------
    med = np.nanmedian(x[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(np.isnan(x), med[None, :], x)
    scaler = StandardScaler().fit(signed_log1p(x[tr]))
    assert scaler.n_samples_seen_ == tr.sum()
    xs = scaler.transform(signed_log1p(x)).astype(np.float32)
    assert np.isfinite(xs).all()
    rid = m["row_idx_orig"].to_numpy()
    uid = np.array([f"{d}:{f}:{r}" for d, f, r in zip(df["day"], df["file_order"], rid)])
    arrays = {}
    for name, sel in (("train", tr), ("val", va), ("test", te)):
        arrays.update({f"X_{name}": xs[sel], f"y_{name}_clean": y_clean[sel], f"y_{name}_noisy": y_noisy[sel],
                       f"id_{name}": np.arange(len(df))[sel], f"tier_{name}": m["tier_liu"].to_numpy()[sel]})
    path = ROOT / cfg["dataset"]["processed_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, class_names=np.array(classes), feature_names=np.array(feats))
    joblib.dump({"scaler": scaler, "median": med, "features": feats}, path.parent / "cicids2017_preprocessor.joblib")
    pd.DataFrame({"row": np.arange(len(df)), "uid": uid, "split": np.where(te, "test", np.where(va, "val", "train")),
                  "clean": y_clean_name, "noisy": y_noisy_name}).to_parquet(out / "row_index.parquet", index=False)
    save_json({"rows": int(len(df)), "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
               "classes": classes, "n_features": len(feats),
               "affected_rows_total": int((y_clean != y_noisy).sum()),
               "affected_rows_test": int(((y_clean != y_noisy) & te).sum()),
               "global_noise_rate": float((y_clean != y_noisy).mean()),
               "split": sp, "separability_limited": prof.loc[prof["separability_limited"], "class"].tolist()},
              out / "data_summary.json")
    print(f"[ok] wrote {path}: train {tr.sum():,} val {va.sum():,} test {te.sum():,}; "
          f"affected rows {int((y_clean != y_noisy).sum()):,} ({(y_clean != y_noisy).mean():.2%})")


if __name__ == "__main__":
    main()
