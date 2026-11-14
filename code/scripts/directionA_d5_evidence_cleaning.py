#!/usr/bin/env python
"""A-D5: does label-free cleaning of the attack evidence rescue the host/time context rule?
Confident Learning (original labels + flow features only) flags suspect labels; a host counts
as exposed on a day only if at least half of its original attack flows survive; benign-labelled
flows of exposed hosts within 60 minutes after surviving evidence are flagged.
Criteria: configs/direction_a_d5_evidence_cleaning.yaml.
Outputs: results/directionA_d5/*, reports/DIRECTION_A_D5.md
(interpretation spliced from reports/direction_a_d5_interpretation.md).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from cleanlab.filter import find_label_issues  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from build_failure_report import md_table  # noqa: E402
from directionA2018_prepare import DROP as DROP18  # noqa: E402
from directionA_2018_alignment import DAYS, LIU_COLS, NUM, ORIG_COLS, ORIG_FILE, key_hash, shifted  # noqa: E402
from directionA_d4_context import flag, score  # noqa: E402
from directionA_prepare import DROP as DROP17  # noqa: E402
from directionA_prepare import load_original  # noqa: E402
from src.utils import git_commit, load_json, load_yaml, save_json  # noqa: E402

warnings.filterwarnings("ignore")


# ---- data: the A-D4 context frames plus flow features --------------------------------------------
def data2017() -> dict:
    ca = load_yaml(ROOT / "configs/directionA_common.yaml")
    orig = load_original(ca)
    mask = pd.read_parquet(ca["dataset"]["noise_mask"])
    assert len(orig) == len(mask) and (orig["row_idx"].to_numpy() == mask["row_idx_orig"].to_numpy()).all()
    ts = pd.to_datetime(orig["Timestamp"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    ts = ts.where(ts.dt.hour >= 8, ts + pd.Timedelta(hours=12))        # 12-h clock without AM/PM
    noisy = mask["base_orig"].astype(str).to_numpy()
    frame = pd.DataFrame({"t": ts, "src": orig["Source IP"].astype(str).str.strip(),
                          "dst": orig["Destination IP"].astype(str).str.strip(), "day": orig["day"].to_numpy()})
    clean = mask["corrected_liu"].astype(object).to_numpy()
    tier = mask["tier_liu"].to_numpy()
    sys.path.insert(0, ca["dataset"]["mask_code_dir"])
    diff = import_module("10_label_diff")
    liu = diff.load_version(str(Path(ca["dataset"]["original_parquet_dir"]).parent / "liu"), diff.LIU_DAY_FILES, "liu")
    key = ["day", "Flow ID", "dur", "occ"]
    att = mask[key].merge(liu[key + ["att_liu"]], on=key, how="left")["att_liu"].astype(float).to_numpy()
    assert len(att) == len(mask)
    covered = (tier > 0) & (noisy != "UNLABELED")
    cand = covered & (noisy == "BENIGN")
    clean_attack = covered & (clean != "BENIGN") & pd.notna(clean) & ~(att == 1)   # Attempted is not an attack
    labelled = noisy != "UNLABELED"
    feats = [c for c in orig.columns if c not in DROP17 + ["day", "file_order"]]
    assert len(feats) == 77, len(feats)
    x = np.array(orig.loc[labelled, feats].apply(pd.to_numeric, errors="coerce"), dtype=np.float32)
    del orig, liu
    return {"frame": frame, "noisy": noisy, "labelled": labelled, "x": x, "cand": cand,
            "attack": labelled & (noisy != "BENIGN"), "clean_attack": clean_attack, "known": tier > 0,
            "subsets": {"tier-1 (primary)": tier == 1, "all covered": covered},
            "hidden": {"tier-1 (primary)": cand & clean_attack & (tier == 1), "all covered": cand & clean_attack},
            "clean_class": np.asarray(clean, dtype=object)}


def data2018() -> dict:
    align = load_yaml(ROOT / "configs/direction_a_2018_alignment.yaml")
    common = load_yaml(ROOT / "configs/directionA2018_common.yaml")
    off = load_json(ROOT / "results/directionA_2018/summary.json")
    cols = dict(LIU_COLS, **{"Src IP": "src", "Dst IP": "dst"})
    xs, parts, feats = [], [], None
    for day in DAYS:
        if day in common["dataset"]["excluded_days"]:
            continue
        o = pd.read_csv(Path(common["dataset"]["original_dir"]) / ORIG_FILE[day], dtype=str, engine="pyarrow")
        o = o[o["Dst Port"] != "Dst Port"].reset_index(drop=True)      # repeated header lines
        f = [c for c in o.columns if c not in DROP18]
        assert feats is None or f == feats, day
        feats = f
        k = o[list(ORIG_COLS)].rename(columns=ORIG_COLS)
        for c in NUM:
            k[c] = pd.to_numeric(k[c], errors="coerce")
        ts = pd.to_datetime(k["ts"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        k["t"] = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        h = key_hash(k)
        liu = pd.read_csv(Path(align["data"]["corrected_dir"]) / f"{day}.csv", usecols=list(cols), dtype=str,
                          engine="pyarrow").rename(columns=cols)
        for c in NUM:
            liu[c] = pd.to_numeric(liu[c], errors="coerce")
        tsl = pd.to_datetime(liu["ts"], format="ISO8601", errors="coerce")
        liu["tf"] = (tsl - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        liu["att"] = pd.to_numeric(liu["att"], errors="coerce").fillna(-1).astype(int)
        liu = shifted(liu.drop(columns="ts"), off["time_offset_hours"], off["rounding"])
        liu["h"] = key_hash(liu)
        grp = liu[liu["h"] != 0].groupby("h").agg(n_lab=("label", "nunique"), n_att=("att", "nunique"),
                                                   lab=("label", "first"), att=("att", "first"),
                                                   src=("src", "first"), dst=("dst", "first"))
        m = pd.DataFrame({"h": h}).join(grp, on="h")
        ok = (m["lab"].notna() & (m["n_lab"] == 1) & (m["n_att"] == 1)).to_numpy() & (h != 0)
        xs.append(np.array(o.loc[ok, feats].apply(pd.to_numeric, errors="coerce"), dtype=np.float32))
        parts.append(pd.DataFrame({"day": day, "t": pd.to_datetime(k.loc[ok, "t"].to_numpy(), unit="s"),
                                   "src": m.loc[ok, "src"].to_numpy(), "dst": m.loc[ok, "dst"].to_numpy(),
                                   "noisy": o.loc[ok, "Label"].astype(str).str.strip().to_numpy(),
                                   "liu": m.loc[ok, "lab"].astype(str).to_numpy(),
                                   "att": m.loc[ok, "att"].astype(int).to_numpy()}))
        print(f"[2018] {day}: matched {int(ok.sum()):,}", flush=True)
        del o, k, liu, grp, m
    assert len(feats) == 78, len(feats)
    frame = pd.concat(parts, ignore_index=True)
    noisy_benign = frame["noisy"].str.lower().eq("benign").to_numpy()
    clean_attack = (~frame["liu"].str.upper().eq("BENIGN") & frame["att"].eq(-1)
                    & ~frame["liu"].str.endswith("Attempted")).to_numpy()
    everything = np.ones(len(frame), bool)
    return {"frame": frame[["t", "src", "dst", "day"]], "noisy": frame["noisy"].to_numpy(), "labelled": everything,
            "x": np.vstack(xs), "cand": noisy_benign, "attack": ~noisy_benign, "clean_attack": clean_attack,
            "known": everything, "subsets": {"all matched (primary)": everything},
            "hidden": {"all matched (primary)": noisy_benign & clean_attack},
            "clean_class": frame["liu"].to_numpy(dtype=object)}


# ---- Confident Learning on original labels -------------------------------------------------------
def confident_learning(x: np.ndarray, labels: np.ndarray, seed: int, params: dict) -> tuple[np.ndarray, dict]:
    names, y = np.unique(labels, return_inverse=True)
    x[~np.isfinite(x)] = np.nan
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    x = np.where(np.isnan(x), med[None, :], x)
    groups = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
    probs = np.zeros((len(y), len(names)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    best = []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(cv.split(x, y, groups)):
        es = rng.random(len(tr)) < 0.1
        fit, ev = tr[~es], tr[es]
        present = np.unique(y[fit])
        lookup = np.full(len(names), -1)
        lookup[present] = np.arange(len(present))
        ev = ev[np.isin(y[ev], present)]
        clf = xgb.XGBClassifier(**params, random_state=seed)
        clf.fit(x[fit], lookup[y[fit]], eval_set=[(x[ev], lookup[y[ev]])], verbose=False)
        probs[np.ix_(te, present)] = clf.predict_proba(x[te], iteration_range=(0, clf.best_iteration + 1))
        best.append(int(clf.best_iteration))
        print(f"  fold {k}: best_iteration {clf.best_iteration}, classes {len(present)}/{len(names)}", flush=True)
    issues = find_label_issues(y, probs, filter_by="prune_by_noise_rate", n_jobs=1)
    return issues, {"classes": [str(n) for n in names], "best_iterations": best, "flagged": int(issues.sum())}


# ---- rules ---------------------------------------------------------------------------------------
def host_trust(frame: pd.DataFrame, attack: np.ndarray, surviving: np.ndarray) -> pd.DataFrame:
    """Per (day, host): share of its original attack flows (as source or destination) that survive."""
    a = frame.loc[attack, ["day", "src", "dst"]].assign(ok=surviving[attack])
    both = pd.concat([a[["day", "src", "ok"]].rename(columns={"src": "host"}),
                      a[["day", "dst", "ok"]].rename(columns={"dst": "host"})]).dropna(subset=["host"])
    return both.groupby(["day", "host"])["ok"].agg(surviving_share="mean", evidence_flows="size")


def flag_exposed(frame: pd.DataFrame, candidates: np.ndarray, surviving: np.ndarray, trust: pd.DataFrame,
                 minutes: int, min_share: float) -> np.ndarray:
    exposed = trust.index[trust["surviving_share"] >= min_share]
    ev = pd.concat([frame.loc[surviving, ["day", "t", "src"]].rename(columns={"src": "host"}),
                    frame.loc[surviving, ["day", "t", "dst"]].rename(columns={"dst": "host"})]).dropna()
    ev = ev[pd.MultiIndex.from_frame(ev[["day", "host"]]).isin(exposed)]
    ev = ev[["t", "host"]].drop_duplicates().sort_values("t")
    ev["hit"] = 1
    cand = frame.loc[candidates, ["t", "src"]].rename(columns={"src": "host"}).dropna()
    cand = cand.assign(row=cand.index).sort_values("t")
    m = pd.merge_asof(cand, ev, on="t", by="host", direction="backward", tolerance=pd.Timedelta(minutes=minutes))
    out = np.zeros(len(frame), bool)
    out[m.loc[m["hit"].notna(), "row"].to_numpy()] = True
    return out


def breakdown(ds: str, d: dict, flagged: np.ndarray, trust: pd.DataFrame, top: int = 8) -> tuple:
    subset = next(s for s in d["subsets"] if "primary" in s)
    sel, fr = d["subsets"][subset], d["frame"]
    c, h = d["cand"] & sel, d["hidden"][subset]
    t = pd.DataFrame({"day": fr["day"].to_numpy(), "candidates": c, "hidden": h, "flagged": flagged & c,
                      "flagged_hidden": flagged & h, "false_flags": flagged & c & ~d["clean_attack"]})
    by_day = t.groupby("day").sum()
    by_day = by_day[(by_day["flagged"] + by_day["hidden"]) > 0].reset_index()
    by_day.insert(0, "dataset", ds)
    by_day["precision"] = 1 - by_day["false_flags"] / by_day["flagged"].clip(lower=1)
    by_day["recall_hidden"] = by_day["flagged_hidden"] / by_day["hidden"].clip(lower=1)
    ff = t["false_flags"].to_numpy()
    hosts = (pd.DataFrame({"day": fr["day"].to_numpy()[ff], "host": fr["src"].to_numpy()[ff]}).value_counts()
             .head(top).rename("false_flags").reset_index())
    hosts.insert(0, "dataset", ds)
    hosts["share"] = hosts["false_flags"] / max(ff.sum(), 1)
    return by_day, hosts.join(trust, on=["day", "host"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-cl", action="store_true",
                    help="load the saved Confident Learning flags (rerun after a failure downstream of them)")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "configs/direction_a_d5_evidence_cleaning.yaml")
    params = load_yaml(ROOT / "configs/xgboost.yaml")["params"]
    out = ROOT / "results/directionA_d5"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    win, min_share, seed = cfg["window_minutes"], cfg["min_surviving_share"], cfg["seed"]
    ref = pd.read_csv(ROOT / "results/directionA_d4/context_rules.csv")

    data = {}
    for ds, loader in (("cicids2017", data2017), ("cicids2018", data2018)):
        d = loader()
        cidx = np.where(d["cand"])[0]
        d["flag_v2"] = flag(d["frame"], cidx, d["attack"], "V2", win)
        for sub in d["subsets"]:                                   # control must reproduce A-D4 exactly
            r = score(ds, sub, d["subsets"][sub], d["cand"], d["hidden"][sub], d["clean_attack"], d["flag_v2"],
                      d["clean_class"], "V2_uncleaned", win)
            q = ref[(ref["dataset"] == ds) & (ref["subset"] == sub) & (ref["rule"] == "V2") & (ref["window_min"] == win)]
            assert len(q) == 1 and all(int(q[k].iloc[0]) == r[k] for k in ("candidates", "hidden", "flagged")), (ds, sub)
        data[ds] = d
        print(f"[{ds}] loaded, A-D4 V2 reproduced ({time.perf_counter() - t0:.0f}s)", flush=True)

    rows, cleaning, keys, days, hosts, cl_info = [], [], [], [], [], {}
    d4_hosts = pd.concat([pd.read_csv(ROOT / "results/directionA_d4/breakdown_hidden_hosts.csv").assign(role="hidden-attack host"),
                          pd.read_csv(ROOT / "results/directionA_d4/breakdown_false_flag_hosts.csv").assign(role="A-D4 false-flag host")])
    for ds, d in data.items():
        f_issues, f_info = out / f"cl_issues_{ds}.npy", out / f"cl_info_{ds}.json"
        if args.reuse_cl and f_issues.exists():
            issues, cl_info[ds] = np.load(f_issues), load_json(f_info)
            d.pop("x")
            print(f"[{ds}] reusing saved Confident Learning flags", flush=True)
        else:
            print(f"[{ds}] Confident Learning on {int(d['labelled'].sum()):,} rows", flush=True)
            issues = np.zeros(len(d["frame"]), bool)
            issues[d["labelled"]], cl_info[ds] = confident_learning(d.pop("x"), d["noisy"][d["labelled"]], seed, params)
            np.save(f_issues, issues)
            save_json(cl_info[ds], f_info)
        surviving = d["attack"] & ~issues
        trust = host_trust(d["frame"], d["attack"], surviving)
        cidx = np.where(d["cand"])[0]
        flags = {"E2_trusted_host": flag_exposed(d["frame"], cidx, surviving, trust, win, min_share),
                 "E1_cleaned_evidence": flag(d["frame"], cidx, surviving, "V2", win),
                 "V2_uncleaned": d["flag_v2"], "CL_only": issues}
        for rule, f in flags.items():
            for sub in d["subsets"]:
                rows.append(score(ds, sub, d["subsets"][sub], d["cand"], d["hidden"][sub], d["clean_attack"], f,
                                  d["clean_class"], rule, 0 if rule == "CL_only" else win))
                print(f"  {rule} / {sub}: recall {rows[-1]['recall_hidden']:.3f} precision {rows[-1]['precision']:.3f} "
                      f"flagged {rows[-1]['flagged']:,}", flush=True)
        att, kn, ca = d["attack"], d["known"], d["clean_attack"]
        false_ev, true_ev = att & kn & ~ca, att & kn & ca
        cleaning.append({"dataset": ds, "labelled_rows": int(d["labelled"].sum()), "flagged_by_cl": int(issues.sum()),
                         "evidence_rows": int(att.sum()), "evidence_removed": int((issues & att).sum()),
                         "false_evidence_rows": int(false_ev.sum()),
                         "false_evidence_removed": float(issues[false_ev].mean()) if false_ev.any() else np.nan,
                         "true_evidence_rows": int(true_ev.sum()),
                         "true_evidence_removed": float(issues[true_ev].mean()) if true_ev.any() else np.nan,
                         "host_days_with_evidence": int(len(trust)),
                         "host_days_exposed": int((trust["surviving_share"] >= min_share).sum())})
        k = d4_hosts[d4_hosts["dataset"] == ds][["dataset", "role", "day", "src", "rows"]].rename(
            columns={"src": "host", "rows": "A-D4 rows"})
        k = k.join(trust, on=["day", "host"])
        k["exposed"] = k["surviving_share"] >= min_share
        keys.append(k)
        b = breakdown(ds, d, flags["E2_trusted_host"], trust)
        days.append(b[0])
        hosts.append(b[1])
        print(f"[{ds}] done ({time.perf_counter() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    v = cfg["verdict"]["signal"]
    res["signal"] = (res["recall_hidden"] >= v["min_recall_hidden"]) & (res["precision"] >= v["min_precision"])
    res.to_csv(out / "rules.csv", index=False, float_format="%.6f")
    cleaning, keys = pd.DataFrame(cleaning), pd.concat(keys, ignore_index=True)
    days, hosts = pd.concat(days, ignore_index=True), pd.concat(hosts, ignore_index=True)
    cleaning.to_csv(out / "cleaning.csv", index=False, float_format="%.6f")
    keys.to_csv(out / "key_hosts.csv", index=False, float_format="%.6f")
    days.to_csv(out / "e2_by_day.csv", index=False, float_format="%.6f")
    hosts.to_csv(out / "e2_false_flag_hosts.csv", index=False, float_format="%.6f")
    prim = res[res["subset"].str.contains("primary") & res["rule"].eq("E2_trusted_host")].set_index("dataset")["signal"]
    verdict = "method_route_open" if len(prim) == 2 and prim.all() else "route_closed"
    save_json({"verdict": verdict, "e2_signal_by_dataset": {k_: bool(v_) for k_, v_ in prim.items()},
               "confident_learning": cl_info, "reused_cl": args.reuse_cl, "minutes": (time.perf_counter() - t0) / 60},
              out / "summary.json")
    base = ["dataset", "subset", "rule", "candidates", "hidden", "flagged", "flagged_share_of_candidates",
            "precision", "recall_hidden", "signal"]
    print(res[base].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(cleaning.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(keys.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("verdict:", verdict, f"({(time.perf_counter() - t0) / 60:.1f} min)")

    L = []
    w = L.append
    w("# DIRECTION A - Diagnosis A-D5: After cleaning the attack evidence, can the host/time context retrieve the systematically missed attack \n")
    w(f">Researcher 2026-09-26 Selected, the last pass of the method route in direction A. Generation time{time.strftime('%Y-%m-%d %H:%M')};"
      f"code commit `{git_commit()[:7]}`. Interpretation criteria `configs/direction_a_d5_evidence_cleaning.yaml` is submitted before calculation."
      "Only original labels and flow features are used for cleaning; corrected labels are only used for scoring. \n")
    interp = ROOT / "reports/direction_a_d5_interpretation.md"
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    w("## D5-a Rule effect \n")
    w(f"E2 (main rule) treats a host as exposed when at least {min_share:.0%} of its original attack-labeled flows on that day survive evidence cleaning; "
      f"it flags the host's benign-labeled flows occurring within {win} minutes after the cleaned evidence. "
      f"E1 (ablation) omits the exposure condition; V2_uncleaned is the A-D4 control; CL_only flags original benign-labeled flows identified by Confident Learning. "
      f"The criterion on the primary subset is recall ≥ {v['min_recall_hidden']} and precision ≥ {v['min_precision']}. "
      f"**Verdict: {verdict}** (E2 by dataset: " +
      ",".join(f"{k_} {'Meet the standard' if v_ else 'Not up to standard'}" for k_, v_ in prim.items()) + ").\n")
    w(md_table(res[base].rename(columns={
        "dataset": "Dataset", "subset": "Subset", "rule": "Rule", "candidates": "Benign-labelled rows",
        "hidden": "Hidden attacks", "flagged": "Flagged", "flagged_share_of_candidates": "Flagged share",
        "precision": "Precision", "recall_hidden": "Recall (hidden)", "signal": "Signal"})))
    w("\n## D5-b Evidence cleaning effect \n")
    w("false_evidence = flows labeled attack originally but benign or attempted in the corrected labels; "
      "true_evidence = flows labeled attack in both versions; *_removed = the share flagged as suspect by Confident Learning and excluded from evidence. "
      "host_days counts (day, host) pairs appearing among original attack-labeled flows and the number classified as exposed.\n")
    w(md_table(cleaning))
    w("\n## D5-c | Whether the key hosts in A-D4 are still judged to be exposed after cleaning \n")
    w("hidden-attack host = the source host in A-D4 that was attacked by false positives (should be exposed); A-D4 false-flag host = the host with the most false positives in A-D4"
      "(Should not be exposed). surviving_share = The proportion of the host's original attack flow that can withstand cleaning that day. \n")
    w(md_table(keys))
    w("\n## D5-d E2 Disassembled by day (main subset) \n")
    w(md_table(days))
    w("\n## D5-e E2 The source host with the highest concentration of false positives (primary subset) \n")
    w(md_table(hosts))
    w("\n## file \n")
    w("`results/directionA_d5/`:rules.csv、cleaning.csv、key_hosts.csv、e2_by_day.csv、e2_false_flag_hosts.csv、"
      "summary.json (cl_issues_*.npy / cl_info_*.json are cleaning results and will not be stored in the database). Code: `scripts/directionA_d5_evidence_cleaning.py`.")
    (ROOT / "reports/DIRECTION_A_D5.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
