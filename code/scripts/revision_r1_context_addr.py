#!/usr/bin/env python
"""LABELGAP revision, round 1 -- POST HOC addition (see configs/revision_round1.yaml, amendments).
Restricting the CSE-CIC-IDS2018 context-rule evidence to tier-A flows cut the exposure rule's recall from 99% to
23%. Tier B only means that a key is repeated; a tier-B flow's addresses are unambiguous when every corrected flow
sharing its key has the same address pair. This script separates the two: evidence restricted to flows with
unambiguous addresses (tier A, or tier B with one address pair), and the tier and address composition of the
attack-labeled evidence of the two scanning hosts. Convention K.
Outputs: results/revision_round1/context_addr_*.csv, context_addr_summary.json
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

from revision_r1_context import frame2018, run_rules  # noqa: E402
from src.utils import load_yaml, save_json  # noqa: E402

SCANNERS = [("Wednesday-28-02-2018", "172.31.69.24"), ("Thursday-01-03-2018", "172.31.69.13")]


def main() -> None:
    plan = load_yaml(ROOT / "configs/revision_round1.yaml")
    out = ROOT / plan["outputs"]["results"]
    t0 = time.perf_counter()
    fr = frame2018()
    issues = np.load(ROOT / "results/directionA_d5/cl_issues_cicids2018.npy")
    assert len(issues) == len(fr)
    cand = fr["noisy"].str.lower().eq("benign").to_numpy()
    attack = ~cand
    k_attack = ~fr["liu"].str.upper().eq("BENIGN").to_numpy()
    tier_a = (fr["tier"] == "A").to_numpy()
    unamb = tier_a | fr["addr_unanimous"].to_numpy()
    cls = fr["liu"].to_numpy(dtype=object)
    rows = []
    for name, sel, ev in (("all matched, unambiguous-address evidence", np.ones(len(fr), bool), unamb),
                          ("tier-A candidates, unambiguous-address evidence", tier_a, unamb)):
        rr, _ = run_rules("cicids2018", fr, cand, attack, issues, {name: sel}, {"K": k_attack}, cls, evidence=ev)
        rows += rr
    res = pd.DataFrame(rows)
    res.to_csv(out / "context_addr_rules.csv", index=False, float_format="%.6f")
    comp = []
    for day, host in SCANNERS:
        for role in ("src", "dst"):
            e = attack & (fr["day"] == day).to_numpy() & (fr[role] == host).to_numpy()
            comp.append({"day": day, "host": host, "role": role, "evidence_flows": int(e.sum()),
                         "tier_A": int((e & tier_a).sum()), "tier_B_unanimous": int((e & ~tier_a & unamb).sum()),
                         "tier_B_ambiguous": int((e & ~unamb).sum())})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "context_addr_scanner_evidence.csv", index=False)
    save_json({"evidence_rows": int(attack.sum()), "evidence_tier_A": int((attack & tier_a).sum()),
               "evidence_tier_B_unanimous": int((attack & ~tier_a & unamb).sum()),
               "evidence_tier_B_ambiguous": int((attack & ~unamb).sum()),
               "minutes": (time.perf_counter() - t0) / 60}, out / "context_addr_summary.json")
    print(res[["subset", "rule", "window_min", "candidates", "hidden", "flagged", "flagged_share_of_candidates",
               "precision", "recall_hidden"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
