from pathlib import Path

import pytest

from fraud_ml.service import FraudScoringService


def test_scores_a_valid_row(trained_artifacts: Path, feature_row: dict[str, float]):
    service = FraudScoringService(trained_artifacts)
    result = service.score(feature_row)
    assert 0 <= result["fraud_score"] <= 1
    assert isinstance(result["review_flag"], bool)
    assert result["model_version"] == service.model_version


def test_rejects_schema_mismatch(trained_artifacts: Path, feature_row: dict[str, float]):
    service = FraudScoringService(trained_artifacts)
    with pytest.raises(ValueError, match="missing=\\['Amount'\\]"):
        service.score({k: v for k, v in feature_row.items() if k != "Amount"})
    with pytest.raises(ValueError, match="extra=\\['unexpected'\\]"):
        service.score({**feature_row, "unexpected": 1.0})


def test_rejects_non_finite_values(trained_artifacts: Path, feature_row: dict[str, float]):
    service = FraudScoringService(trained_artifacts)
    with pytest.raises(ValueError, match="finite"):
        service.score({**feature_row, "V1": float("nan")})


def test_missing_artifacts_raise(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        FraudScoringService(tmp_path)
