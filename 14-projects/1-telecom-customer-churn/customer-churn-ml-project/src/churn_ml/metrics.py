"""Evaluation helpers: threshold choice, threshold metrics, capacity and segment metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_arrays(y_true: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have equal length")
    if y.shape[0] == 0:
        raise ValueError("y_true and scores must not be empty")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y, s


def threshold_metrics(
    y_true: Any, scores: Any, threshold: float, fn_cost: float = 5.0, fp_cost: float = 1.0
) -> dict[str, float]:
    """Confusion counts, precision/recall/F1 and cost when flagging score >= threshold."""
    y, s = _as_arrays(y_true, scores)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "cost": float(fn_cost * fn + fp_cost * fp),
    }


def select_threshold(
    y_true: Any, scores: Any, fn_cost: float, fp_cost: float
) -> tuple[float, float]:
    """Exact cost-minimising threshold over every distinct score; returns (threshold, cost).

    Sorting once and using cumulative sums evaluates all candidate thresholds in
    O(n log n). Ties go to the higher threshold (fewer contacts for the same cost).
    """
    y, s = _as_arrays(y_true, scores)
    order = np.argsort(-s, kind="stable")
    s_desc, y_desc = s[order], y[order]
    tp = np.cumsum(y_desc)
    fp = np.cumsum(1 - y_desc)
    last_of_each_value = np.r_[s_desc[1:] != s_desc[:-1], True]
    thresholds = s_desc[last_of_each_value]
    positives = int(y.sum())
    costs = fn_cost * (positives - tp[last_of_each_value]) + fp_cost * fp[last_of_each_value]

    best = int(np.argmin(costs))
    flag_nothing_cost = fn_cost * positives
    if flag_nothing_cost < costs[best]:
        return float(np.nextafter(s.max(), np.inf)), float(flag_nothing_cost)
    return float(thresholds[best]), float(costs[best])


def classification_metrics(
    y_true: Any, scores: Any, threshold: float, fn_cost: float = 5.0, fp_cost: float = 1.0
) -> dict[str, float]:
    """Threshold metrics plus ranking (AP, ROC-AUC) and probability (Brier) quality."""
    y, s = _as_arrays(y_true, scores)
    both_classes = len(np.unique(y)) == 2
    pred = (s >= threshold).astype(int)
    return {
        **threshold_metrics(y, s, threshold, fn_cost, fp_cost),
        "prevalence": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "average_precision": float(average_precision_score(y, s)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if both_classes else float("nan"),
        "brier_score": float(brier_score_loss(y, s)),
    }


def top_k_metrics(y_true: Any, scores: Any, fraction: float = 0.1) -> dict[str, float]:
    """Precision/recall when the team can only act on the top `fraction` of scores."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    y, s = _as_arrays(y_true, scores)
    k = max(1, int(np.ceil(len(y) * fraction)))
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(s)[-k:]] = True
    tp = int(((y == 1) & flagged).sum())
    positives = int((y == 1).sum())
    return {
        "k": k,
        "precision_at_k": tp / k,
        "recall_at_k": tp / max(positives, 1),
        "positives_found": tp,
        "total_positives": positives,
    }


def segment_metrics(
    segments: pd.DataFrame, y_true: Any, scores: Any, threshold: float
) -> pd.DataFrame:
    """Per-segment size, churn rate, precision and recall, to spot groups the model fails."""
    y, s = _as_arrays(y_true, scores)
    pred = (s >= threshold).astype(int)
    rows = []
    for column in segments.columns:
        values = segments[column].astype("string").fillna("<missing>").to_numpy()
        for value in sorted(set(values)):
            mask = values == value
            rows.append(
                {
                    "feature": column,
                    "segment": value,
                    "customers": int(mask.sum()),
                    "churn_rate": float(y[mask].mean()),
                    "precision": float(precision_score(y[mask], pred[mask], zero_division=0)),
                    "recall": float(recall_score(y[mask], pred[mask], zero_division=0)),
                    "mean_probability": float(s[mask].mean()),
                }
            )
    return pd.DataFrame(rows)
