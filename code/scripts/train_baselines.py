#!/usr/bin/env python
"""Steps 6-13: train + evaluate the five baselines for the given seeds.

  python scripts/train_baselines.py                    # formal runs, all models x 5 seeds
  python scripts/train_baselines.py --smoke --seeds 42 # smoke test -> smoke/
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA init

import argparse  # noqa: E402
import copy  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.evaluation.metrics import write_split_outputs  # noqa: E402
from src.utils import (MODEL_ORDER, environment_info, load_yaml, save_json,  # noqa: E402
                       sha256_file)


def load_data(common: dict, label_condition: str | None = None) -> dict:
    d = np.load(ROOT / common["dataset"]["processed_file"], allow_pickle=False)
    data = {k: d[k] for k in d.files}
    data["class_names"] = [str(c) for c in data["class_names"]]
    if "y_train_clean" in data:  # direction A: same rows, two label versions
        assert label_condition in ("clean", "noisy"), "--label-condition required for this dataset"
        data["y_train"] = data[f"y_train_{label_condition}"]
        data["y_val"] = data[f"y_val_{label_condition}"]
        data["y_test"] = data["y_test_clean"]  # evaluation always against the corrected labels
    return data


def get_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(str(path))
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(path, mode="w"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def run_one(model: str, seed: int, common: dict, data: dict, base: Path, smoke: bool,
            env: dict, class_weights: np.ndarray | None = None, data_meta: dict | None = None,
            label_condition: str | None = None, method: dict | None = None) -> None:
    method = method or {}
    if method.get("exclude_train"):  # A-D1 Confident Learning: drop flagged training rows
        drop = np.load(ROOT / method["exclude_train"].format(seed=seed))
        assert drop.shape == data["y_train"].shape
        data = {**data, "X_train": data["X_train"][~drop], "y_train": data["y_train"][~drop],
                "id_train": data["id_train"][~drop]}
        method = {**method, "excluded_rows": int(drop.sum())}
    run_dir = base / "results/raw" / model / f"seed{seed}"
    ckpt_dir = base / "checkpoints" / model / f"seed{seed}"
    log = get_logger(base / "logs" / model / f"seed{seed}.log")
    model_cfg = load_yaml(ROOT / f"configs/{model}.yaml")
    train_cfg = copy.deepcopy(common["dl_training"])
    if method.get("loss") == "gce":
        train_cfg.update(loss_name="gce", gce_q=method["gce_q"])
    if method.get("fixed_epochs"):
        train_cfg["fixed_epochs"] = method["fixed_epochs"]
    if smoke:
        train_cfg["max_epochs"] = common["smoke_overrides"]["max_epochs"]
        if model == "xgboost":
            model_cfg["params"]["n_estimators"] = common["smoke_overrides"]["xgb_n_estimators"]

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if model == "xgboost":
        from src.training.xgb_trainer import train_xgb
        out = train_xgb(model_cfg, data, seed, ckpt_dir, log, class_weights=class_weights)
    else:
        import torch
        torch.use_deterministic_algorithms(True)
        if method.get("coteaching_forget_rate") is not None:
            from src.training.noise_robust import train_coteaching
            out = train_coteaching(model, model_cfg, train_cfg, data, seed, ckpt_dir, log,
                                   method["coteaching_forget_rate"], method["coteaching_tk"])
        else:
            from src.training.dl_trainer import train_dl
            out = train_dl(model, model_cfg, train_cfg, data, seed, ckpt_dir, log,
                           class_weights=class_weights)

    names = data["class_names"]
    mv = write_split_outputs(run_dir, "val", data["y_val"], out["proba_val"], data["id_val"],
                             names, model, seed)
    mt = write_split_outputs(run_dir, "test", data["y_test"], out["proba_test"],
                             data["id_test"], names, model, seed)
    write_split_outputs(run_dir, "test", data["y_test"], out["proba_test_last"],
                        data["id_test"], names, model, seed, tag="_lastckpt")
    out["history"].to_csv(base / "logs" / model / f"seed{seed}_history.csv", index=False)
    save_json(out["efficiency"], run_dir / "efficiency.json")
    save_json({
        "model": model, "seed": seed, "smoke": smoke, "started": started,
        "class_weights": None if class_weights is None else [float(v) for v in class_weights],
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": model_cfg,
        "training_config": train_cfg if model != "xgboost" else "see model_config",
        "common_config": common,
        "data_metadata": data_meta,
        "label_condition": label_condition,
        "method": method or None,
        "environment": env,
    }, run_dir / "config.json")
    log.info("DONE %s seed=%d | val macroF1=%.4f | test macroF1=%.4f acc=%.4f",
             model, seed, mv["macro_f1"], mt["macro_f1"], mt["accuracy"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                    help="balanced = D3 diagnostic (N / (C * n_c) on the training partition)")
    ap.add_argument("--base", default=None,
                    help="output root relative to the project (default: project root, or smoke/)")
    ap.add_argument("--common", default="configs/directionA_common.yaml", help="dataset / protocol config")
    ap.add_argument("--label-condition", choices=["clean", "noisy"], default=None,
                    help="direction A only: which label version to train and validate on")
    ap.add_argument("--loss", choices=["ce", "gce"], default="ce", help="A-D1: gce = Zhang & Sabuncu 2018")
    ap.add_argument("--gce-q", type=float, default=0.7)
    ap.add_argument("--coteaching-forget-rate", type=float, default=None, help="A-D1: enables co-teaching")
    ap.add_argument("--coteaching-tk", type=int, default=10)
    ap.add_argument("--fixed-epochs", type=int, default=None,
                    help="A-D1: train exactly N epochs and evaluate the last one (DL only)")
    ap.add_argument("--exclude-train", default=None,
                    help="A-D1: .npy boolean mask of training rows to drop; may contain {seed}")
    args = ap.parse_args()
    method = {}
    if args.loss == "gce":
        method.update(loss="gce", gce_q=args.gce_q)
    if args.coteaching_forget_rate is not None:
        method.update(coteaching_forget_rate=args.coteaching_forget_rate, coteaching_tk=args.coteaching_tk)
    if args.exclude_train:
        method.update(exclude_train=args.exclude_train)
    if args.fixed_epochs:
        method.update(fixed_epochs=args.fixed_epochs)

    common = load_yaml(ROOT / args.common)
    seeds = args.seeds or common["seeds"]
    if args.base:
        base = ROOT / args.base
    else:
        base = ROOT / "smoke" if args.smoke else ROOT
    data = load_data(common, args.label_condition)
    processed = ROOT / common["dataset"]["processed_file"]
    data_meta = {"processed_file": str(processed.relative_to(ROOT)), "sha256": sha256_file(processed)}
    class_weights = None
    if args.class_weight == "balanced":
        counts = np.bincount(data["y_train"], minlength=len(data["class_names"]))
        class_weights = len(data["y_train"]) / (len(counts) * counts)
    env = environment_info()
    save_json(env, base / "logs/environment.json")
    for seed in seeds:
        for model in args.models:
            done = base / "results/raw" / model / f"seed{seed}" / "metrics_test.json"
            if args.skip_existing and done.exists():
                print(f"skip {model} seed={seed}")
                continue
            if model == "xgboost" and (method.get("loss") or method.get("coteaching_forget_rate") is not None):
                raise SystemExit("GCE / co-teaching apply to the DL backbones only")
            run_one(model, seed, common, data, base, args.smoke, env, class_weights, data_meta,
                    args.label_condition, method)


if __name__ == "__main__":
    main()
