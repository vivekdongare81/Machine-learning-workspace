import numpy as np
import pandas as pd
import pytest

from churn_ml.metrics import (
    classification_metrics,
    segment_metrics,
    select_threshold,
    threshold_metrics,
    top_k_metrics,
)


def test_threshold_metrics_counts_and_cost():
    result = threshold_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5, fn_cost=5, fp_cost=1)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["cost"] == 6


def test_select_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    scores = np.round(rng.random(300) * 0.6 + y * 0.3, 2)  # rounding creates ties
    threshold, cost = select_threshold(y, scores, fn_cost=5, fp_cost=1)
    brute = min(threshold_metrics(y, scores, t, 5, 1)["cost"] for t in np.unique(scores))
    assert cost == brute
    assert threshold_metrics(y, scores, threshold, 5, 1)["cost"] == cost


def test_select_threshold_can_flag_nothing():
    threshold, cost = select_threshold([0, 0, 1], [0.9, 0.8, 0.1], fn_cost=1, fp_cost=5)
    assert threshold > 0.9 and cost == 1


def test_classification_metrics_include_ranking_quality():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert result["average_precision"] == 1.0 and result["roc_auc"] == 1.0


def test_top_k_metrics():
    result = top_k_metrics([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9], fraction=0.25)
    assert result["k"] == 1 and result["precision_at_k"] == 1.0
    with pytest.raises(ValueError):
        top_k_metrics([0, 1], [0.2, 0.8], fraction=0)


def test_segment_metrics_per_group():
    segments = pd.DataFrame({"Contract": ["A", "A", "B", "B"]})
    report = segment_metrics(segments, [1, 0, 1, 1], [0.9, 0.2, 0.1, 0.8], threshold=0.5)
    by_segment = report.set_index("segment")
    assert by_segment.loc["A", "recall"] == 1.0
    assert by_segment.loc["B", "recall"] == 0.5
    assert by_segment["customers"].sum() == 4


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        threshold_metrics([0, 1], [0.3], 0.5)
