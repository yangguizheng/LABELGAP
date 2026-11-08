#!/usr/bin/env python
"""LABELGAP revision, round 1 (descriptive checks of statements carried over into the K-convention text):
(1) CSE-CIC-IDS2018: share of false and true attack evidence removed by the A-D5 cleaning, under K and Stage I;
(2) CSE-CIC-IDS2018: composition of the A-D1 Confident Learning training flags (original and K labels, and whether
the flagged feature vector occurs with more than one original label in the training partition);
(3) CIC-IDS2017: corrected labels and attempted status of the infiltration-labeled flows of 192.168.10.8.
Outputs: results/revision_round1/checks_summary.json
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from directionA_prepare import load_original  # noqa: E402
from revision_r1_context import frame2018  # noqa: E402
from src.utils import load_yaml, save_json  # noqa: E402


def main() -> None:
    plan = load_yaml(ROOT / "configs/revision_round1.yaml")
    out = ROOT / plan["outputs"]["results"]
    S: dict = {}
    # (1) evidence cleaning, CSE-CIC-IDS2018
    fr = frame2018()
    issues = np.load(ROOT / "results/directionA_d5/cl_issues_cicids2018.npy")
    attack = ~fr["noisy"].str.lower().eq("benign").to_numpy()
    benign_corr = fr["liu"].str.upper().eq("BENIGN").to_numpy()
    is_att = (fr["att"].ne(-1) | fr["liu"].str.endswith("Attempted")).to_numpy()
    for conv, false_ev in (("K", attack & benign_corr), ("stage1", attack & (benign_corr | is_att))):
        true_ev = attack & ~false_ev
        S[f"cleaning_{conv}"] = {"false_evidence": int(false_ev.sum()), "true_evidence": int(true_ev.sum()),
                                 "false_removed": float(issues[false_ev].mean()),
                                 "true_removed": float(issues[true_ev].mean())}
    del fr
    # (2) A-D1 Confident Learning flags on the CSE-CIC-IDS2018 training partition
    d = np.load(ROOT / "data/processed/cicids2018_labelnoise_attK.npz", allow_pickle=False)
    names = np.array([str(c) for c in d["class_names"]])
    x, yn, yk = d["X_train"], d["y_train_noisy"], d["y_train_clean"]
    h = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
    n_lab = pd.DataFrame({"h": h, "y": yn}).groupby("h")["y"].transform("nunique").to_numpy()
    for f in sorted((ROOT / "results/directionA2018_d1").glob("cl_issues_seed*.npy")):
        iss = np.load(f)
        comp = pd.DataFrame({"orig": names[yn[iss]], "K": names[yk[iss]]}).value_counts().head(8)
        S[f"cl_{f.stem}"] = {"flagged": int(iss.sum()),
                             "share_vector_with_several_original_labels": float((n_lab[iss] > 1).mean()),
                             "top_orig_K_pairs": {f"{a} -> {b}": int(v) for (a, b), v in comp.items()}}
    # (3) CIC-IDS2017, host 192.168.10.8, infiltration-labeled flows
    ca = load_yaml(ROOT / "configs/directionA_common.yaml")
    orig = load_original(ca)
    mask = pd.read_parquet(ca["dataset"]["noise_mask"])
    sys.path.insert(0, str(ROOT / "alignment_cicids2017"))
    diff = import_module("10_label_diff")
    liu = diff.load_version(str(Path(ca["dataset"]["original_parquet_dir"]).parent / "liu"), diff.LIU_DAY_FILES, "liu")
    key = ["day", "Flow ID", "dur", "occ"]
    att = mask[key].merge(liu[key + ["att_liu"]], on=key, how="left")["att_liu"]
    sel = (orig["Source IP"].astype(str).str.strip() == "192.168.10.8").to_numpy() \
        & (mask["base_orig"].astype(str) == "Infiltration").to_numpy()
    t = pd.DataFrame({"corrected": mask.loc[sel, "corrected_liu"].astype(str).to_numpy(),
                      "attempted": att[sel].astype(str).to_numpy(), "tier": mask.loc[sel, "tier_liu"].to_numpy()})
    S["cicids2017_192_168_10_8_infiltration"] = {f"{a} | att={b} | tier={c}": int(v)
                                                 for (a, b, c), v in t.value_counts().items()}
    save_json(S, out / "checks_summary.json")
    print(S)


if __name__ == "__main__":
    main()
