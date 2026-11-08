#!/usr/bin/env python
"""Direction A on CSE-CIC-IDS2018, Steps 3-5: build the paired noisy/clean dataset.

Original TrafficForML rows matched to Liu et al.'s corrected flows exactly as in the
alignment check (same key, offset and tiers), labels mapped into the original
taxonomy ("Attempted" -> Benign), uniform 40% subsample, grouped stratified split,
Step-4 data sanity gate, train-only preprocessing.
Outputs: data/processed/cicids2018_labelnoise_v1.npz, results/directionA2018/data_*
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from directionA_2018_alignment import DAYS, NUM, ORIG_COLS, ORIG_FILE, corrected, key_hash, shifted  # noqa: E402
from src.data.transforms import signed_log1p  # noqa: E402
from src.utils import load_json, load_yaml, save_json  # noqa: E402

DROP = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Label"]
CLEAN_MAP = {"BENIGN": "Benign", "DoS Slowloris": "DoS attacks-Slowloris", "DoS GoldenEye": "DoS attacks-GoldenEye",
             "DoS Hulk": "DoS attacks-Hulk", "DoS SlowHTTPTest": "DoS attacks-SlowHTTPTest",
             "FTP-BruteForce": "FTP-BruteForce", "SSH-BruteForce": "SSH-Bruteforce",
             "DDoS-LOIC-HTTP": "DDoS attacks-LOIC-HTTP", "DDoS-HOIC": "DDOS attack-HOIC",
             "DDoS-LOIC-UDP": "DDOS attack-LOIC-UDP", "Web Attack - Brute Force": "Brute Force -Web",
             "Web Attack - XSS": "Brute Force -XSS", "Web Attack - SQL": "SQL Injection", "Botnet Ares": "Bot"}


def map_clean(label: str, att: int) -> str:
    if att != -1 or label.endswith("Attempted"):
        return "Benign"                                   # dataset authors' recommendation
    if label.startswith("Infiltration"):
        return "Infilteration"                            # original spelling
    return CLEAN_MAP[label]                               # unknown names fail loudly


def main() -> None:
    cfg = load_yaml(ROOT / "configs/directionA2018_common.yaml")
    align_cfg = load_yaml(ROOT / "configs/direction_a_2018_alignment.yaml")
    gate = load_yaml(ROOT / "configs/direction_a_label_noise.yaml")["data_sanity_gate"]
    off = load_json(ROOT / "results/directionA_2018/summary.json")
    out = ROOT / "results/directionA2018"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    xs, meta, feats = [], [], None
    for day in DAYS:
        if day in cfg["dataset"]["excluded_days"]:
            continue
        o = pd.read_csv(Path(cfg["dataset"]["original_dir"]) / ORIG_FILE[day], dtype=str, engine="pyarrow")
        o = o[o["Dst Port"] != "Dst Port"].reset_index(drop=True)      # repeated header lines
        f = [c for c in o.columns if c not in DROP]
        assert feats is None or f == feats, (day, set(f) ^ set(feats))
        feats = f
        k = o[list(ORIG_COLS)].rename(columns=ORIG_COLS)
        for c in NUM:
            k[c] = pd.to_numeric(k[c], errors="coerce")
        ts = pd.to_datetime(k["ts"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        k["t"] = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        h = key_hash(k)
        liu = shifted(corrected(align_cfg, day), off["time_offset_hours"], off["rounding"])
        liu["h"] = key_hash(liu)
        grp = liu[liu["h"] != 0].groupby("h").agg(n_l=("label", "size"), lab=("label", "first"),
                                                   att=("att", "first"), n_lab=("label", "nunique"),
                                                   n_att=("att", "nunique"))
        m = pd.DataFrame({"h": h}).join(grp, on="h")
        m["n_o"] = m["h"].map(pd.Series(h[h != 0]).value_counts())
        present = m["n_l"].notna().to_numpy() & (h != 0)
        unanimous = present & (m["n_lab"] == 1).to_numpy() & (m["n_att"] == 1).to_numpy()
        tier = np.where(unanimous & (m["n_l"] == 1).to_numpy() & (m["n_o"] == 1).to_numpy(), 1,
                        np.where(unanimous, 2, 0))
        keep = tier > 0
        x = np.array(o.loc[keep, feats].apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
        xs.append(x)
        meta.append(pd.DataFrame({"day": day, "tier": tier[keep], "t": k.loc[keep, "t"].to_numpy(),
                                  "noisy": o.loc[keep, "Label"].str.strip().to_numpy(),
                                  "liu": m.loc[keep, "lab"].astype(str).to_numpy(),
                                  "att": m.loc[keep, "att"].astype(int).to_numpy()}))
        print(f"[{day}] matched {keep.sum():,} of {len(o):,} ({time.perf_counter() - t0:.0f}s)", flush=True)
        del o, k, liu, grp, m

    x = np.vstack(xs)
    meta = pd.concat(meta, ignore_index=True)
    meta["clean"] = [map_clean(a, b) for a, b in zip(meta["liu"], meta["att"])]
    rng = np.random.default_rng(cfg["dataset"]["subsample"]["seed"])
    sel = np.sort(rng.choice(len(meta), size=int(round(cfg["dataset"]["subsample"]["fraction"] * len(meta))),
                             replace=False))
    x, meta = x[sel], meta.iloc[sel].reset_index(drop=True)
    x[~np.isfinite(x)] = np.nan
    assert len(feats) == 78, len(feats)
    print(f"[ok] {len(meta):,} rows after the {cfg['dataset']['subsample']['fraction']:.0%} subsample", flush=True)

    cc, nc = meta["clean"].value_counts(), meta["noisy"].value_counts()
    classes = sorted(cc.index, key=lambda c: (-cc[c], c)) + sorted(set(nc.index) - set(cc.index), key=lambda c: -nc[c])
    cmap = {c: i for i, c in enumerate(classes)}
    y_clean = meta["clean"].map(cmap).to_numpy()
    y_noisy = meta["noisy"].map(cmap).to_numpy()
    benign = cmap["Benign"]

    groups = pd.util.hash_pandas_object(pd.DataFrame(x).fillna(-1.2345e300), index=False).to_numpy()
    sp = cfg["split"]
    folds = np.full(len(meta), -1)
    for kf, (_, idx) in enumerate(StratifiedGroupKFold(n_splits=sp["n_splits"], shuffle=True,
                                                        random_state=sp["seed"]).split(x, y_clean, groups)):
        folds[idx] = kf
    te, va = folds == sp["test_fold"], folds == sp["val_fold"]
    tr = ~(te | va)
    for a, b in ((tr, te), (tr, va), (va, te)):
        assert np.intersect1d(groups[a], groups[b]).size == 0

    # ---- Step 4: data sanity gate ------------------------------------------------------------
    g = pd.DataFrame({"h": groups[tr], "clean": y_clean[tr], "noisy": y_noisy[tr]})
    n_clean = g.groupby("h")["clean"].transform("nunique").to_numpy()
    n_noisy = g.groupby("h")["noisy"].transform("nunique").to_numpy()
    rows = []
    for c, name in enumerate(classes):
        sel_c, sel_n = g["clean"].to_numpy() == c, g["noisy"].to_numpy() == c
        aff = (y_clean != y_noisy) & (y_clean == c)
        rows.append({"class": name, "rows_clean": int((y_clean == c).sum()), "rows_noisy": int((y_noisy == c).sum()),
                     "hidden_as_benign": int(((y_clean == c) & (y_noisy == benign)).sum()) if c != benign else np.nan,
                     "noise_rate_clean_class": float(aff.sum() / max((y_clean == c).sum(), 1)),
                     "train_clean": int(sel_c.sum()), "val_clean": int(((y_clean == c) & va).sum()),
                     "test_clean": int(((y_clean == c) & te).sum()), "test_affected": int((aff & te).sum()),
                     "train_conflict_rate_clean_labels": float((n_clean[sel_c] > 1).mean()) if sel_c.any() else np.nan,
                     "train_conflict_rate_noisy_labels": float((n_noisy[sel_n] > 1).mean()) if sel_n.any() else np.nan})
    prof = pd.DataFrame(rows)
    prof["separability_limited"] = prof["train_conflict_rate_clean_labels"] >= gate["conflict_rate_limit"]
    prof.to_csv(out / "data_profile.csv", index=False, float_format="%.6f")
    trans = meta.groupby(["noisy", "clean"]).size().rename("rows").reset_index().sort_values("rows", ascending=False)
    trans.to_csv(out / "label_transitions.csv", index=False)
    print(prof.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- Step 5: preprocessing fitted on the training partition only -------------------------------
    med = np.nanmedian(x[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(np.isnan(x), med[None, :], x)
    scaler = StandardScaler().fit(signed_log1p(x[tr]))
    assert scaler.n_samples_seen_ == tr.sum()
    xz = scaler.transform(signed_log1p(x)).astype(np.float32)
    assert np.isfinite(xz).all()
    arrays = {}
    for name, s in (("train", tr), ("val", va), ("test", te)):
        arrays.update({f"X_{name}": xz[s], f"y_{name}_clean": y_clean[s], f"y_{name}_noisy": y_noisy[s],
                       f"id_{name}": np.arange(len(meta))[s], f"tier_{name}": meta["tier"].to_numpy()[s]})
    path = ROOT / cfg["dataset"]["processed_file"]
    np.savez_compressed(path, **arrays, class_names=np.array(classes), feature_names=np.array(feats))
    joblib.dump({"scaler": scaler, "median": med, "features": feats}, path.parent / "cicids2018_preprocessor.joblib")
    meta.assign(split=np.where(te, "test", np.where(va, "val", "train"))).to_parquet(out / "row_index.parquet",
                                                                                    index=False)
    aff = y_clean != y_noisy
    save_json({"rows": int(len(meta)), "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
               "classes": classes, "n_features": len(feats), "affected_rows_total": int(aff.sum()),
               "affected_rows_test": int((aff & te).sum()), "global_noise_rate": float(aff.mean()),
               "hidden_attack_rows": int(((y_noisy == benign) & (y_clean != benign)).sum()),
               "split": sp, "subsample": cfg["dataset"]["subsample"],
               "separability_limited": prof.loc[prof["separability_limited"], "class"].tolist(),
               "minutes": (time.perf_counter() - t0) / 60}, out / "data_summary.json")
    print(f"[ok] wrote {path}: train {tr.sum():,} val {va.sum():,} test {te.sum():,}; affected {aff.sum():,} "
          f"({aff.mean():.2%}); hidden attacks {int(((y_noisy == benign) & (y_clean != benign)).sum()):,}")


if __name__ == "__main__":
    main()
