"""Metric computation and per-run artifact writing (protocol sections 9, 22)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from src.utils import save_json


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, class_names: list[str]):
    n_classes = len(class_names)
    labels = np.arange(n_classes)
    y_pred = proba.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    onehot = np.eye(n_classes, dtype=int)[y_true]
    ap = np.array([average_precision_score(onehot[:, c], proba[:, c]) if onehot[:, c].any()
                   else np.nan for c in labels])

    # benign class: "Normal" in UNSW-NB15, "BENIGN" in CICIDS2017, "Benign" in CSE-CIC-IDS2018
    normal = class_names.index(next(n for n in ("Normal", "BENIGN", "Benign") if n in class_names))
    is_normal, pred_normal = y_true == normal, y_pred == normal
    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted",
                                zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "macro_roc_auc_ovr": macro_auc(y_true, proba, labels),
        "macro_pr_auc": float(np.nanmean(ap)),
        "log_loss": log_loss(y_true, proba, labels=labels),
        # binary NIDS view derived from the 10-class prediction
        "normal_false_alarm_rate": float((~pred_normal[is_normal]).mean()),
        "attack_missed_as_normal_rate": float(pred_normal[~is_normal].mean()),
    }
    overall = {k: float(v) for k, v in overall.items()}
    per_class = pd.DataFrame({"class": class_names, "support": s.astype(int), "precision": p,
                              "recall": r, "f1": f, "pr_auc": ap})
    return overall, per_class, cm


def macro_auc(y_true: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> float:
    """sklearn's macro one-vs-rest AUC; when some classes are absent from y_true (possible
    for corrected CSE-CIC-IDS2018 labels) the mean runs over the classes present."""
    if all((y_true == c).any() for c in labels):
        return roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
    return float(np.mean([roc_auc_score(y_true == c, proba[:, c]) for c in labels if (y_true == c).any()]))


def normalise_rows(cm: np.ndarray) -> np.ndarray:
    tot = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, tot, out=np.zeros_like(cm, dtype=float), where=tot > 0)


def write_split_outputs(run_dir: Path, split: str, y_true: np.ndarray, proba: np.ndarray,
                        sample_ids: np.ndarray, class_names: list[str], model: str,
                        seed: int, tag: str = "") -> dict:
    """Write metrics_<split><tag>.json, per_class, confusion matrices and
    predictions for one evaluation split of one run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    overall, per_class, cm = compute_metrics(y_true, proba, class_names)
    save_json(overall, run_dir / f"metrics_{split}{tag}.json")
    if tag:  # secondary evaluations (e.g. last checkpoint) keep only the summary
        return overall
    per_class.to_csv(run_dir / f"per_class_{split}.csv", index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        run_dir / f"confusion_raw_{split}.csv", index_label="true\\pred")
    pd.DataFrame(normalise_rows(cm), index=class_names, columns=class_names).to_csv(
        run_dir / f"confusion_norm_{split}.csv", index_label="true\\pred", float_format="%.6f")

    y_pred = proba.argmax(1)
    names = np.asarray(class_names)
    pred = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": names[y_true],
        "pred_label": names[y_pred],
        "true_idx": y_true.astype(np.int16),
        "pred_idx": y_pred.astype(np.int16),
        "confidence": proba.max(1).astype(np.float32),
        "seed": seed,
        "model": model,
    })
    for c, name in enumerate(class_names):
        pred[f"prob_{name}"] = proba[:, c].astype(np.float32)
    pred.to_parquet(run_dir / f"{split}_predictions.parquet", index=False)
    return overall
