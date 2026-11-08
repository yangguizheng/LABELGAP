"""B1 XGBoost baseline. Early stopping on validation mlogloss (the tree
counterpart of the DL validation cross-entropy rule)."""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import set_seed


def train_xgb(model_cfg: dict, data: dict, seed: int, ckpt_dir: Path,
              log: logging.Logger, class_weights: np.ndarray | None = None) -> dict:
    """class_weights (D3 diagnostic only) become sample weights for training and
    for the early-stopping validation set; None reproduces Stage I exactly."""
    import xgboost as xgb

    set_seed(seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    params = dict(model_cfg["params"])
    clf = xgb.XGBClassifier(**params, random_state=seed)
    weights = {} if class_weights is None else {
        "sample_weight": class_weights[data["y_train"]],
        "sample_weight_eval_set": [class_weights[data["y_val"]]]}

    # XGBoost needs labels 0..k-1: encode the classes present in the training labels and map the
    # probabilities back to the full class space below (the identity whenever they are already 0..k-1).
    n_classes = len(data["class_names"])
    present = np.unique(data["y_train"])
    assert np.isin(np.unique(data["y_val"]), present).all(), "validation classes absent from training"
    enc = np.full(n_classes, -1)
    enc[present] = np.arange(len(present))

    t0 = time.perf_counter()
    clf.fit(data["X_train"], enc[data["y_train"]], eval_set=[(data["X_val"], enc[data["y_val"]])],
            verbose=False, **weights)
    train_time = time.perf_counter() - t0

    best_it = int(clf.best_iteration)
    rng = (0, best_it + 1)
    booster = clf.get_booster()
    booster.save_model(ckpt_dir / "last.ubj")
    booster[: best_it + 1].save_model(ckpt_dir / "best.ubj")
    n_rounds_run = len(clf.evals_result()["validation_0"]["mlogloss"])
    log.info("xgboost seed=%d rounds_run=%d best_iteration=%d train_time=%.1fs",
             seed, n_rounds_run, best_it, train_time)

    with warnings.catch_warnings():  # numpy input on a CUDA booster -> DMatrix fallback
        warnings.simplefilter("ignore")
        proba_val = clf.predict_proba(data["X_val"], iteration_range=rng)
        proba_test_last = clf.predict_proba(data["X_test"], iteration_range=(0, n_rounds_run))

        def infer():  # end-to-end: host numpy array -> host probabilities
            return clf.predict_proba(data["X_test"], iteration_range=rng)

        proba_test = infer()
        times = []
        for _ in range(3):
            t = time.perf_counter()
            infer()
            times.append(time.perf_counter() - t)
    infer_s = float(np.median(times))

    # Classes absent from the training labels (possible with corrected CSE-CIC-IDS2018 labels) get
    # probability 0; XGBoost's columns are the encoded present classes, in order.
    if len(present) < n_classes:
        assert np.array_equal(clf.classes_, np.arange(len(present))), clf.classes_

        def pad(q):
            full = np.zeros((len(q), n_classes), dtype=q.dtype)
            full[:, present] = q
            return full
        proba_val, proba_test, proba_test_last = pad(proba_val), pad(proba_test), pad(proba_test_last)

    trees = booster[: best_it + 1].trees_to_dataframe()
    history = pd.DataFrame({"iteration": np.arange(n_rounds_run),
                            "val_mlogloss": clf.evals_result()["validation_0"]["mlogloss"]})
    efficiency = {
        "params": None,
        "flops_per_sample": None,
        "n_trees": int((best_it + 1) * n_classes),
        "n_leaves": int((trees["Feature"] == "Leaf").sum()),
        "model_size_mb": (ckpt_dir / "best.ubj").stat().st_size / 2 ** 20,
        "train_time_s": train_time,
        "train_wall_time_incl_val_s": train_time,
        "epochs_run": n_rounds_run,
        "best_epoch": best_it + 1,
        "time_per_epoch_s": train_time / n_rounds_run,
        "test_inference_time_s": infer_s,
        "test_inference_us_per_sample": infer_s / len(data["X_test"]) * 1e6,
        "peak_gpu_memory_mb": None,
        "inference_batch_size": "full",
    }
    return {"proba_val": proba_val, "proba_test": proba_test,
            "proba_test_last": proba_test_last, "history": history,
            "efficiency": efficiency}
