"""Stage II diagnostics: input-space overlap (D1) and partition shift (D2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def knn_label_counts(xq: np.ndarray, xr: np.ndarray, yr: np.ndarray, ks: list[int],
                     n_classes: int, device: str = "cuda", chunk: int = 1024):
    """Label counts in the k-neighbourhood of every query row (Euclidean, float64).

    All reference points tied with the k-th distance are included, so a query
    that exactly matches a large group of identical training rows sees that
    group's whole label distribution rather than an arbitrary subset of it.
    Returns ({k: (n_query, n_classes) counts}, nearest-neighbour distance)."""
    ref = torch.from_numpy(xr).to(device, torch.float64)
    ref_sq = (ref * ref).sum(1)
    onehot = torch.nn.functional.one_hot(torch.from_numpy(yr).long().to(device),
                                         n_classes).to(torch.float64)
    kmax = max(ks)
    counts = {k: np.zeros((len(xq), n_classes)) for k in ks}
    nn_dist = np.zeros(len(xq))
    for s in range(0, len(xq), chunk):
        q = torch.from_numpy(xq[s:s + chunk]).to(device, torch.float64)
        d2 = ((q * q).sum(1, keepdim=True) + ref_sq[None, :] - 2.0 * q @ ref.T).clamp_(min=0)
        top, _ = torch.topk(d2, kmax, dim=1, largest=False, sorted=True)
        nn_dist[s:s + chunk] = top[:, 0].sqrt().cpu().numpy()
        for k in ks:
            mask = (d2 <= top[:, k - 1:k] + 1e-9).to(torch.float64)
            counts[k][s:s + chunk] = (mask @ onehot).cpu().numpy()
    return counts, nn_dist


def kdn(counts: np.ndarray, y: np.ndarray) -> np.ndarray:
    """k-Disagreeing Neighbours (Smith et al., Machine Learning 2014): share of
    the neighbourhood whose label differs from the row's own label."""
    return 1.0 - counts[np.arange(len(y)), y] / counts.sum(1)


def group_majority_ceiling(hashes: np.ndarray, y: np.ndarray, class_names: list[str]):
    """Exact-duplicate view of the training set. Any deterministic function of
    the 42 features must give every row of an identical-vector group the same
    prediction; predicting the group's majority label is the accuracy-optimal
    choice. Returns, per class, the recall that rule reaches (ties share credit
    equally) and the majority labels of the conflict groups the class sits in."""
    df = pd.DataFrame({"h": hashes, "y": y})
    tab = pd.crosstab(df["h"], df["y"]).reindex(columns=range(len(class_names)), fill_value=0)
    top = tab.max(axis=1)
    is_max = tab.eq(top, axis=0)
    credit = is_max.div(is_max.sum(axis=1), axis=0)          # tie-sharing
    n_labels = (tab > 0).sum(axis=1)
    rows = []
    for c, name in enumerate(class_names):
        in_c = tab[c] > 0
        total = tab[c].sum()
        conflict = in_c & (n_labels > 1)
        maj_of_conflict = tab[conflict].idxmax(axis=1)
        w = tab.loc[conflict, c]
        maj_share = (w.groupby(maj_of_conflict).sum() / w.sum()).sort_values(ascending=False) \
            if w.sum() else pd.Series(dtype=float)
        rows.append({
            "class": name,
            "train_rows": int(total),
            "rows_in_conflict_groups": float(tab.loc[conflict, c].sum() / total),
            "majority_rule_recall_ceiling": float((tab[c] * credit[c]).sum() / total),
            "conflict_group_majority_label": ", ".join(
                f"{class_names[int(k)]} {v:.0%}" for k, v in maj_share.head(3).items()),
        })
    overall = float(top.sum() / len(df))
    return pd.DataFrame(rows), overall


def adversarial_validation(x: np.ndarray, origin: np.ndarray, groups: np.ndarray,
                           params: dict, n_splits: int, seed: int):
    """Out-of-fold P(origin = 1 | x) with grouped, stratified folds."""
    import xgboost as xgb

    oof = np.zeros(len(origin))
    aucs, gains = [], []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(cv.split(x, origin, groups)):
        clf = xgb.XGBClassifier(**params, objective="binary:logistic", random_state=seed + fold)
        clf.fit(x[tr], origin[tr], verbose=False)
        oof[te] = clf.predict_proba(x[te])[:, 1]
        aucs.append(roc_auc_score(origin[te], oof[te]))
        g = clf.get_booster().get_score(importance_type="total_gain")
        gains.append(np.array([g.get(f"f{i}", 0.0) for i in range(x.shape[1])]))
    return oof, np.array(aucs), np.mean(gains, axis=0)
