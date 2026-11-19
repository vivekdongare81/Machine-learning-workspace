import numpy as np
import pytest

from fraud_ml.metrics import (
    classification_metrics,
    select_threshold,
    threshold_metrics,
    top_k_metrics,
)


def test_threshold_metrics_counts_and_cost():
    result = threshold_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5, fn_cost=10, fp_cost=1)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["cost"] == 11


def test_threshold_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        threshold_metrics([0, 1], [0.3], 0.5)


def test_select_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    scores = np.round(rng.random(300) * 0.6 + y * 0.3, 2)  # rounding creates ties
    threshold, cost = select_threshold(y, scores, fn_cost=7, fp_cost=1)
    brute = min(threshold_metrics(y, scores, t, 7, 1)["cost"] for t in np.unique(scores))
    assert cost == brute
    assert threshold_metrics(y, scores, threshold, 7, 1)["cost"] == cost


def test_select_threshold_can_flag_nothing():
    # A false alert costs more than a missed positive, so flagging nothing is optimal.
    threshold, cost = select_threshold([0, 0, 1], [0.9, 0.8, 0.1], fn_cost=1, fp_cost=5)
    assert threshold > 0.9 and cost == 1


def test_classification_metrics_include_ranking_quality():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert result["average_precision"] == 1.0 and result["roc_auc"] == 1.0
    assert result["prevalence"] == 0.5


def test_top_k_metrics():
    result = top_k_metrics([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9], fraction=0.25)
    assert result["k"] == 1 and result["precision_at_k"] == 1.0 and result["recall_at_k"] == 1.0
    with pytest.raises(ValueError):
        top_k_metrics([0, 1], [0.2, 0.8], fraction=0)
