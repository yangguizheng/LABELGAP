#!/usr/bin/env python
"""Direction A, CSE-CIC-IDS2018: attempted-flow convention variants.

K: attempted rows keep the base attack class of their corrected label (the CIC-IDS2017 convention).
E: attempted rows are removed from training, validation and test.
Everything else (rows, split, features, preprocessing, class indices) is the Stage-I data.
Rules: configs/direction_a2018_attempted.yaml (committed before this script is run).
Outputs: the two processed files named there, results/directionA2018_attempted/variants_*.{csv,json}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from directionA2018_prepare import CLEAN_MAP  # noqa: E402
from src.utils import load_yaml, save_json  # noqa: E402

SUFFIX = " - Attempted"


def base_class(liu: str) -> str:
    """Base attack class of a corrected label, in the original taxonomy (Stage-I map)."""
    base = liu[: -len(SUFFIX)] if liu.endswith(SUFFIX) else liu
    if base.startswith("Infiltration"):
        return "Infilteration"                            # original spelling
    return CLEAN_MAP[base]                                # unknown names fail loudly


def main() -> None:
    rules = load_yaml(ROOT / "configs/direction_a2018_attempted.yaml")
    stage1 = load_yaml(ROOT / "configs/directionA2018_common.yaml")
    out = ROOT / rules["outputs"]["results"]
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(ROOT / stage1["dataset"]["processed_file"], allow_pickle=False)
    d = {k: d[k] for k in d.files}
    meta = pd.read_parquet(ROOT / "results/directionA2018/row_index.parquet")
    names = [str(c) for c in d["class_names"]]
    cmap = {c: i for i, c in enumerate(names)}

    att_code = meta["att"].to_numpy() != -1
    att_name = meta["liu"].astype(str).str.endswith("Attempted").to_numpy()
    assert (att_code == att_name).all(), "attempted code and label suffix disagree"
    is_att = att_code
    y_base = np.full(len(meta), -1)
    y_base[is_att] = [cmap[base_class(s)] for s in meta.loc[is_att, "liu"].astype(str)]
    clean_all = meta["clean"].map(cmap).to_numpy()
    noisy_all = meta["noisy"].map(cmap).to_numpy()

    variants = {"K": {}, "E": {}}
    for s in ("train", "val", "test"):
        ids = d[f"id_{s}"]
        assert (clean_all[ids] == d[f"y_{s}_clean"]).all() and (noisy_all[ids] == d[f"y_{s}_noisy"]).all()
        att = is_att[ids]
        y_k = d[f"y_{s}_clean"].copy()
        y_k[att] = y_base[ids][att]
        variants["K"].update({f"X_{s}": d[f"X_{s}"], f"y_{s}_clean": y_k, f"y_{s}_noisy": d[f"y_{s}_noisy"],
                              f"id_{s}": ids, f"tier_{s}": d[f"tier_{s}"]})
        keep = ~att
        variants["E"].update({f"X_{s}": d[f"X_{s}"][keep], f"y_{s}_clean": d[f"y_{s}_clean"][keep],
                              f"y_{s}_noisy": d[f"y_{s}_noisy"][keep], f"id_{s}": ids[keep],
                              f"tier_{s}": d[f"tier_{s}"][keep]})

    summary = {"attempted_rows": int(is_att.sum()),
               "attempted_by_split": {s: int(is_att[d[f"id_{s}"]].sum()) for s in ("train", "val", "test")}}
    for v, arrs in variants.items():
        path = ROOT / rules["data"]["variants"][v]["processed_file"]
        np.savez_compressed(path, **arrs, class_names=d["class_names"], feature_names=d["feature_names"])
        rows = []
        for s in ("train", "val", "test"):
            yc, yn = arrs[f"y_{s}_clean"], arrs[f"y_{s}_noisy"]
            t = pd.DataFrame({"noisy": np.array(names)[yn], "clean": np.array(names)[yc]})
            t = t.groupby(["noisy", "clean"]).size().rename("rows").reset_index()
            rows.append(t.assign(split=s))
        trans = pd.concat(rows).pivot_table(index=["noisy", "clean"], columns="split", values="rows",
                                            fill_value=0).reset_index()
        trans["all"] = trans[["train", "val", "test"]].sum(axis=1)
        trans = trans.sort_values("all", ascending=False)
        trans.to_csv(out / f"variants_{v}_transitions.csv", index=False)
        n = {s: int(len(arrs[f"y_{s}_clean"])) for s in ("train", "val", "test")}
        aff = {s: int((arrs[f"y_{s}_clean"] != arrs[f"y_{s}_noisy"]).sum()) for s in ("train", "val", "test")}
        summary[v] = {"processed_file": str(path.relative_to(ROOT)), "rows": n, "disagreeing_rows": aff,
                      "clean_classes_with_test_rows": [names[c] for c in np.unique(arrs["y_test_clean"])],
                      "noisy_classes_with_test_rows": [names[c] for c in np.unique(arrs["y_test_noisy"])]}
        print(f"[{v}] wrote {path.name}: rows {n}, disagreeing {aff}", flush=True)
        print(trans.head(12).to_string(index=False), flush=True)
    save_json(summary, out / "variants_summary.json")


if __name__ == "__main__":
    main()
