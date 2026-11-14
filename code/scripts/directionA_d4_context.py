#!/usr/bin/env python
"""A-D4: host/time context signal for the systematically hidden attacks (rule-based,
no training). V1: the flow's source host was the destination of an original-attack
flow within the preceding T minutes; V2: it appeared in one as source or destination.
Criteria: configs/direction_a_d4_context.yaml.
Outputs: results/directionA_d4/*, reports/DIRECTION_A_D4.md
(interpretation spliced from reports/direction_a_d4_interpretation.md).
"""
from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_failure_report import md_table  # noqa: E402
from directionA_2018_alignment import DAYS, LIU_COLS, NUM, key_hash, original, shifted  # noqa: E402
from directionA_prepare import load_original  # noqa: E402
from src.utils import git_commit, load_json, load_yaml, save_json  # noqa: E402


def flag(frame: pd.DataFrame, candidates: np.ndarray, attack: np.ndarray, rule: str, minutes: int) -> np.ndarray:
    """frame: columns t (datetime64), src, dst. Returns a boolean flag per row (candidates only)."""
    ev = [frame.loc[attack, ["t", "dst"]].rename(columns={"dst": "host"})]
    if rule == "V2":
        ev.append(frame.loc[attack, ["t", "src"]].rename(columns={"src": "host"}))
    ev = pd.concat(ev).dropna().drop_duplicates().sort_values("t")
    ev["hit"] = 1
    cand = frame.loc[candidates, ["t", "src"]].rename(columns={"src": "host"}).dropna()
    cand = cand.assign(row=cand.index).sort_values("t")
    m = pd.merge_asof(cand, ev, on="t", by="host", direction="backward",
                      tolerance=pd.Timedelta(minutes=minutes))
    out = np.zeros(len(frame), bool)
    out[m.loc[m["hit"].notna(), "row"].to_numpy()] = True
    return out


def score(ds: str, subset: str, sel: np.ndarray, cand: np.ndarray, hidden: np.ndarray, clean_attack: np.ndarray,
          flagged: np.ndarray, clean_class: np.ndarray, rule: str, minutes: int) -> dict:
    c, h = cand & sel, hidden & sel
    r = {"dataset": ds, "subset": subset, "rule": rule, "window_min": minutes,
         "candidates": int(c.sum()), "hidden": int(h.sum()), "flagged": int((flagged & c).sum()),
         "flagged_share_of_candidates": float((flagged & c).sum() / max(c.sum(), 1)),
         "precision": float((flagged & c & clean_attack).sum() / max((flagged & c).sum(), 1)),
         "recall_hidden": float((flagged & h).sum() / max(h.sum(), 1))}
    for k in pd.unique(clean_class[h]):
        hk = h & (clean_class == k)
        if hk.sum() >= 100:
            r[f"recall_{k}"] = float(flagged[hk].mean())
    return r


def breakdown(ds: str, ctx: dict, rule: str, minutes: int, top: int = 8) -> tuple[pd.DataFrame, ...]:
    """Descriptive: where the hidden attacks and the false flags of one rule sit (primary subset)."""
    fr, sel = ctx["frame"], ctx["sel"]
    cand, hid = ctx["cand"] & sel, ctx["hidden"] & sel
    f = flag(fr, np.where(ctx["cand"])[0], ctx["attack"], rule, minutes) & cand
    d = pd.DataFrame({"day": fr["day"].to_numpy(), "src": fr["src"].to_numpy(), "clean": ctx["clean_class"],
                      "candidates": cand, "hidden": hid, "flagged": f, "flagged_hidden": f & hid,
                      "false_flags": f & ~ctx["clean_attack"]})
    by_day = d.groupby("day")[["candidates", "hidden", "flagged", "flagged_hidden", "false_flags"]].sum()
    by_day = by_day[(by_day["flagged"] + by_day["hidden"]) > 0].reset_index()
    by_day.insert(0, "dataset", ds)
    by_day["precision"] = 1 - by_day["false_flags"] / by_day["flagged"].clip(lower=1)
    by_day["recall_hidden"] = by_day["flagged_hidden"] / by_day["hidden"].clip(lower=1)
    att = fr.loc[ctx["attack"], ["day", "src", "dst", "noisy", "dport"]].assign(
        confirmed=ctx["clean_attack"][ctx["attack"]], known=ctx["known"][ctx["attack"]])
    as_src, as_dst = att.groupby(["day", "src"]).size(), att.groupby(["day", "dst"]).size()
    ev = pd.concat([att[["day", "src", "noisy"]].rename(columns={"src": "host"}),
                    att[["day", "dst", "noisy"]].rename(columns={"dst": "host"})])
    ev = ev.groupby(["day", "host", "noisy"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    main_ev = ev.drop_duplicates(["day", "host"]).set_index(["day", "host"])["noisy"]

    def hosts(col: str) -> pd.DataFrame:
        sub = d[d[col]]
        g = sub.groupby(["day", "src"]).size().sort_values(ascending=False).head(top).rename("rows").reset_index()
        g.insert(0, "dataset", ds)
        g["share"] = g["rows"] / max(len(sub), 1)
        keys = list(zip(g["day"], g["src"]))
        g["clean_label"] = [sub.loc[(sub["day"] == a) & (sub["src"] == b), "clean"].astype(str).value_counts().index[0]
                            for a, b in keys]
        g["attack_flows_as_src"] = [int(as_src.get(k, 0)) for k in keys]
        g["attack_flows_as_dst"] = [int(as_dst.get(k, 0)) for k in keys]
        g["main_evidence_label"] = [str(main_ev.get(k, "")) for k in keys]
        out_ev = [att[(att["day"] == a) & (att["src"] == b)] for a, b in keys]
        peers = [e["dst"].value_counts() for e in out_ev]
        g["top_peer_as_src"] = [str(v.index[0]) if len(v) else "" for v in peers]
        g["top_peer_share"] = [float(v.iloc[0] / v.sum()) if len(v) else np.nan for v in peers]
        g["top_peer_port"] = [str(e.loc[e["dst"] == v.index[0], "dport"].value_counts().index[0]) if len(v) else ""
                              for e, v in zip(out_ev, peers)]
        all_ev = [att[(att["day"] == a) & ((att["src"] == b) | (att["dst"] == b)) & att["known"]] for a, b in keys]
        g["evidence_confirmed_share"] = [float(e["confirmed"].mean()) if len(e) else np.nan for e in all_ev]
        return g

    return by_day, hosts("hidden"), hosts("false_flags")


def cicids2017(cfg: dict) -> tuple[list, dict, dict]:
    ca = load_yaml(ROOT / "configs/directionA_common.yaml")
    orig = load_original(ca)
    mask = pd.read_parquet(ca["dataset"]["noise_mask"])
    assert len(orig) == len(mask) and (orig["day"].to_numpy() == mask["day"].to_numpy()).all()
    assert (orig["row_idx"].to_numpy() == mask["row_idx_orig"].to_numpy()).all()   # row order matches the mask
    fid = float((orig["Flow ID"].fillna("").astype(str).str.strip().to_numpy()
                 == mask["Flow ID"].fillna("").astype(str).str.strip().to_numpy()).mean())   # blank rows: "" == ""
    assert fid > 0.999, fid
    ts = pd.to_datetime(orig["Timestamp"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    ts = ts.where(ts.dt.hour >= 8, ts + pd.Timedelta(hours=12))        # 12-h clock without AM/PM
    noisy = mask["base_orig"].astype(str).to_numpy()
    frame = pd.DataFrame({"t": ts, "src": orig["Source IP"].astype(str).str.strip(),
                          "dst": orig["Destination IP"].astype(str).str.strip(),
                          "day": orig["day"].to_numpy(), "noisy": noisy,
                          "dport": pd.to_numeric(orig["Destination Port"], errors="coerce").astype("Int64")})
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
    # clean attack, excluding flows Liu marks "Attempted" (known for tier-1 rows only)
    clean_attack = covered & (clean != "BENIGN") & pd.notna(clean) & ~(att == 1)
    hidden_all = cand & clean_attack
    hidden_t1 = hidden_all & (tier == 1)
    attack = (noisy != "BENIGN") & (noisy != "UNLABELED")
    rows = []
    for rule in cfg["rules"]:
        rk = rule.split("_")[0]
        for w in cfg["windows_minutes"]:
            f = flag(frame, np.where(noisy == "BENIGN")[0], attack, rk, w)
            rows.append(score("cicids2017", "tier-1 (primary)", tier == 1, cand, hidden_t1, clean_attack, f,
                              np.asarray(clean, dtype=object), rk, w))
            rows.append(score("cicids2017", "all covered", covered, cand, hidden_all, clean_attack, f,
                              np.asarray(clean, dtype=object), rk, w))
            print(f"[2017] {rk} {w}min: recall(t1) {rows[-2]['recall_hidden']:.3f} precision(t1) "
                  f"{rows[-2]['precision']:.3f}", flush=True)
    info = {"rows": int(len(frame)), "covered": int(covered.sum()), "hidden_tier1": int(hidden_t1.sum()),
            "hidden_all": int(hidden_all.sum()), "timestamp_parse_rate": float(ts.notna().mean()),
            "flow_id_agreement": fid}
    shown = np.where(att == 1, pd.Series(clean).astype(str) + " - Attempted", pd.Series(clean).astype(str))
    ctx = {"frame": frame, "cand": cand, "sel": tier == 1, "attack": attack, "hidden": hidden_t1,
           "clean_attack": clean_attack, "clean_class": shown, "known": tier > 0}
    return rows, info, ctx


def cicids2018(cfg: dict) -> tuple[list, dict, dict]:
    align = load_yaml(ROOT / "configs/direction_a_2018_alignment.yaml")
    common = load_yaml(ROOT / "configs/directionA2018_common.yaml")
    off = load_json(ROOT / "results/directionA_2018/summary.json")
    cols = dict(LIU_COLS, **{"Src IP": "src", "Dst IP": "dst"})
    parts = []
    for day in DAYS:
        if day in common["dataset"]["excluded_days"]:
            continue
        o = original(align, day)
        liu = pd.read_csv(Path(align["data"]["corrected_dir"]) / f"{day}.csv", usecols=list(cols), dtype=str,
                          engine="pyarrow").rename(columns=cols)
        for c in NUM:
            liu[c] = pd.to_numeric(liu[c], errors="coerce")
        tsl = pd.to_datetime(liu["ts"], format="ISO8601", errors="coerce")
        liu["tf"] = (tsl - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        liu["att"] = pd.to_numeric(liu["att"], errors="coerce").fillna(-1).astype(int)
        liu = shifted(liu.drop(columns="ts"), off["time_offset_hours"], off["rounding"])
        o["h"], liu["h"] = key_hash(o), key_hash(liu)
        grp = liu[liu["h"] != 0].groupby("h").agg(n_lab=("label", "nunique"), n_att=("att", "nunique"),
                                                   lab=("label", "first"), att=("att", "first"),
                                                   src=("src", "first"), dst=("dst", "first"))
        m = o[["h", "t", "label", "dport"]].join(grp, on="h")
        ok = m["lab"].notna() & (m["n_lab"] == 1) & (m["n_att"] == 1) & (o["h"] != 0)
        m = m[ok]
        parts.append(pd.DataFrame({"day": day, "t": pd.to_datetime(m["t"], unit="s"), "src": m["src"],
                                   "dst": m["dst"], "noisy": m["label"].astype(str).str.strip(),
                                   "liu": m["lab"].astype(str), "att": m["att"].astype(int),
                                   "dport": m["dport"].astype("Int64")}))
        print(f"[2018] {day}: matched {int(ok.sum()):,}", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    noisy_benign = frame["noisy"].str.lower().eq("benign").to_numpy()
    clean_attack = (~frame["liu"].str.upper().eq("BENIGN") & frame["att"].eq(-1)
                    & ~frame["liu"].str.endswith("Attempted")).to_numpy()
    hidden = noisy_benign & clean_attack
    attack = ~noisy_benign
    everything = np.ones(len(frame), bool)
    rows = []
    for rule in cfg["rules"]:
        rk = rule.split("_")[0]
        for w in cfg["windows_minutes"]:
            f = flag(frame, np.where(noisy_benign)[0], attack, rk, w)
            rows.append(score("cicids2018", "all matched (primary)", everything, noisy_benign, hidden, clean_attack,
                              f, frame["liu"].to_numpy(dtype=object), rk, w))
            print(f"[2018] {rk} {w}min: recall {rows[-1]['recall_hidden']:.3f} precision {rows[-1]['precision']:.3f}",
                  flush=True)
    attempted = (frame["att"].ne(-1) & ~frame["liu"].str.endswith("Attempted")).to_numpy()
    shown = np.where(attempted, frame["liu"] + " - Attempted", frame["liu"])
    ctx = {"frame": frame, "cand": noisy_benign, "sel": everything, "attack": attack, "hidden": hidden,
           "clean_attack": clean_attack, "clean_class": shown, "known": everything}
    return rows, {"matched_rows": int(len(frame)), "hidden": int(hidden.sum())}, ctx


def main() -> None:
    cfg = load_yaml(ROOT / "configs/direction_a_d4_context.yaml")
    out = ROOT / "results/directionA_d4"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    r17, i17, c17 = cicids2017(cfg)
    r18, i18, c18 = cicids2018(cfg)
    res = pd.DataFrame(r17 + r18)
    res.to_csv(out / "context_rules.csv", index=False, float_format="%.6f")
    bd = [breakdown("cicids2017", c17, "V2", 60), breakdown("cicids2018", c18, "V2", 60)]
    bd = [pd.concat([b[i] for b in bd], ignore_index=True) for i in range(3)]
    for name, t in zip(("breakdown_by_day", "breakdown_hidden_hosts", "breakdown_false_flag_hosts"), bd):
        t.to_csv(out / f"{name}.csv", index=False, float_format="%.6f")
    v = cfg["verdict"]["signal"]
    prim = res[res["subset"].str.contains("primary")].copy()
    prim["signal"] = (prim["recall_hidden"] >= v["min_recall_hidden"]) & (prim["precision"] >= v["min_precision"])
    ok = prim.groupby(["rule", "window_min"])["signal"].agg(["all", "any"])
    verdict = "context_signal" if ok["all"].any() else ("dataset_specific" if ok["any"].any() else "no_context_signal")
    save_json({"cicids2017": i17, "cicids2018": i18, "verdict": verdict,
               "minutes": (time.perf_counter() - t0) / 60}, out / "summary.json")

    L = []
    w = L.append
    w("# DIRECTION A - Diagnosis A-D4: Can the host/time context retrieve the systemically missed attack \n")
    w(f">Researcher selected on 2026-09-26. Generation time{time.strftime('%Y-%m-%d %H:%M')};Code commit `{git_commit()[:7]}`."
      "The interpretation standard `configs/direction_a_d4_context.yaml` is submitted before calculation. Based on rules, no model is trained; the rules only use original tags, IP and time. \n")
    interp = ROOT / "reports/direction_a_d4_interpretation.md"
    if interp.exists():
        w(interp.read_text().strip() + "\n")
    w("## D4-a Rule effect \n")
    w(f"V1 = the source host was the destination host of the original attack flow within the previous T minutes; V2 = the source host was present in the original attack flow (source or destination) within the previous T minutes."
      f"Candidates are flows originally labeled benign. The criterion is recall ≥ {v['min_recall_hidden']} and precision ≥ {v['min_precision']} on the primary subset."
      f"**determination:{verdict}**.\n")
    base = ["dataset", "subset", "rule", "window_min", "candidates", "hidden", "flagged", "flagged_share_of_candidates",
            "precision", "recall_hidden"]
    w(md_table(res[base].rename(columns={
        "dataset": "Dataset", "subset": "Subset", "rule": "Rule", "window_min": "Window (min)",
        "candidates": "Benign-labelled rows", "hidden": "Hidden attacks", "flagged": "Flagged",
        "flagged_share_of_candidates": "Flagged share", "precision": "Precision", "recall_hidden": "Recall (hidden)"})))
    per = [c for c in res.columns if c.startswith("recall_") and c != "recall_hidden"]
    if per:
        w("\n## D4-b Recall rate by category (number of missed rows ≥ 100) \n")
        w(md_table(res[["dataset", "subset", "rule", "window_min"] + per]))
    w("\n## D4-c Teardown: V2, 60 minutes, main subset (descriptive, not involved in judgment) \n")
    w("By day (false_flags = flagged flows whose corrected label is benign or attempted):\n")
    w(md_table(bd[0]))
    w("\n is the source host with the highest concentration of missing tag attacks. attack_flows_as_src / as_dst = The number of times this host appears as a source/destination in the original attack flow that day;"
      "main_evidence_label = the label with the most in these original attack flows; top_peer_as_src / top_peer_share = the original attack flow where this host is the source"
      "The most common destination host, its proportion and the most common destination port; evidence_confirmed_share = Among the original attack flows (those with correction tags) involved in the host that day,"
      "The correction label is also the proportion of attacks (non-attempts). \n")
    w(md_table(bd[1]))
    w("\n The source host with the highest concentration of false positives (column meanings are the same as above): \n")
    w(md_table(bd[2]))
    w("\n## file \n")
    w("`results/directionA_d4/`: context_rules.csv, breakdown_*.csv, summary.json. Code: `scripts/directionA_d4_context.py`.")
    (ROOT / "reports/DIRECTION_A_D4.md").write_text("\n".join(L) + "\n")
    print(res[base].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("verdict:", verdict, f"({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
