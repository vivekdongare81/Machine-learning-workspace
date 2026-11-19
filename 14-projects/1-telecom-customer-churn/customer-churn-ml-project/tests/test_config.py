from pathlib import Path

import pytest
from pydantic import ValidationError

from churn_ml.config import TrainConfig, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
MINIMAL = {
    "data": {"url": "https://example.invalid/x.csv", "path": "x.csv"},
    "models": [{"name": "lr", "type": "logistic_regression"}],
}


def test_repository_config_is_valid():
    cfg = load_config(REPO_CONFIG)
    assert cfg.data.sha256 and len(cfg.data.sha256) == 64
    assert {m.type for m in cfg.models} <= {"logistic_regression", "decision_tree", "random_forest"}
    assert cfg.evaluation.segment_columns


def test_split_must_leave_training_data():
    with pytest.raises(ValidationError, match="leave training data"):
        TrainConfig.model_validate(
            {**MINIMAL, "split": {"validation_fraction": 0.5, "test_fraction": 0.5}}
        )


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({**MINIMAL, "sead": 1})


def test_model_names_must_be_unique():
    models = [{"name": "a", "type": "logistic_regression"}, {"name": "a", "type": "decision_tree"}]
    with pytest.raises(ValidationError, match="unique"):
        TrainConfig.model_validate({**MINIMAL, "models": models})


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({**MINIMAL, "models": [{"name": "x", "type": "xgboost"}]})
