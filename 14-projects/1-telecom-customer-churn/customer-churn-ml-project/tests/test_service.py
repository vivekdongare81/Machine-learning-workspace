from pathlib import Path
from typing import Any

import pytest

from churn_ml.service import ChurnScoringService


def test_scores_a_valid_row(trained_artifacts: Path, feature_row: dict[str, Any]):
    service = ChurnScoringService(trained_artifacts)
    result = service.score(feature_row)
    assert 0 <= result["churn_probability"] <= 1
    assert isinstance(result["predicted_churn"], bool)
    assert result["model_version"] == service.model_version


def test_rejects_schema_mismatch(trained_artifacts: Path, feature_row: dict[str, Any]):
    service = ChurnScoringService(trained_artifacts)
    with pytest.raises(ValueError, match="missing=\\['tenure'\\]"):
        service.score({k: v for k, v in feature_row.items() if k != "tenure"})
    with pytest.raises(ValueError, match="extra=\\['customerID'\\]"):
        service.score({**feature_row, "customerID": "C-1"})


def test_rejects_wrong_value_types(trained_artifacts: Path, feature_row: dict[str, Any]):
    service = ChurnScoringService(trained_artifacts)
    with pytest.raises(ValueError, match="finite numbers"):
        service.score({**feature_row, "tenure": "twelve"})
    with pytest.raises(ValueError, match="text"):
        service.score({**feature_row, "Contract": 1.0})


def test_accepts_nulls_and_unseen_categories(trained_artifacts: Path, feature_row: dict[str, Any]):
    service = ChurnScoringService(trained_artifacts)
    row = {**feature_row, "TotalCharges": None, "PaymentMethod": None, "Contract": "Weekly"}
    assert 0 <= service.score(row)["churn_probability"] <= 1


def test_missing_artifacts_raise(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ChurnScoringService(tmp_path)
