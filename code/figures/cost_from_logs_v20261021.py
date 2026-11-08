"""Training cost of the controlled comparison, read from the original run logs (no new runs).
Logs: nids_failure_discovery/logs/directionA_run.log (CIC-IDS2017) and directionA2018_run.log (CSE-CIC-IDS2018).
XGBoost: boosting rounds run and train_time as logged. Neural models: epochs run and the sum of the logged epoch times.
Each model has six runs per dataset (two label versions x three seeds). Standard deviations use ddof=1.
"""
import collections
import re
import statistics as st
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "logs"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def parse(path):
    runs, cur = [], None
    for line in open(path):
        m = re.search(r"\b(xgboost) seed=(\d+) rounds_run=(\d+) best_iteration=(\d+) train_time=([\d.]+)s", line)
        if m:
            runs.append(["xgboost", int(m.group(2)), int(m.group(3)), float(m.group(5))])
            continue
        m = re.search(r"\b(mlp|cnn|bigru|transformer) seed=(\d+) ep=(\d+) .* ([\d.]+)s\s*\*?\s*$", line)
        if m:
            mod, seed, ep, sec = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
            if ep == 1 or cur is None or cur[0] != mod or cur[1] != seed:
                cur = [mod, seed, 0, 0.0]
                runs.append(cur)
            cur[2] = ep
            cur[3] += sec
    return runs


rows = []
for ds, name in [("cicids2017", "directionA_run.log"), ("cicids2018", "directionA2018_run.log")]:
    runs = parse(LOGS / name)
    by = collections.defaultdict(list)
    for mod, seed, ep, sec in runs:
        by[mod].append((ep, sec))
    for mod in ["xgboost", "mlp", "cnn", "bigru", "transformer"]:
        v = by[mod]
        assert len(v) == 6, (ds, mod, len(v))
        eps, secs = [e for e, _ in v], [s for _, s in v]
        rows.append((ds, mod, st.mean(eps), st.stdev(eps), st.mean(secs), st.stdev(secs)))
with open(OUT / "cost_table.csv", "w") as f:
    f.write("dataset,model,epochs_mean,epochs_sd,time_s_mean,time_s_sd\n")
    for r in rows:
        f.write("%s,%s,%.3f,%.3f,%.3f,%.3f\n" % r)
        print("%s %-12s epochs %6.1f +- %5.1f   time %7.1f +- %6.1f s" % r)
