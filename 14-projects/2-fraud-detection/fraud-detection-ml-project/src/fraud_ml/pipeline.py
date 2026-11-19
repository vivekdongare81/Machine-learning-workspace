"""Model construction from config: preprocessing + estimator in one sklearn Pipeline.

Keeping preprocessing inside the pipeline means the exact same transformations
run in training, batch scoring and the API (no training/serving skew).
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_ml.config import ModelSpec

ESTIMATORS: dict[str, Any] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
}


def make_preprocessor() -> Pipeline:
    """All benchmark features are numeric; statistics are learned from training data only."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def build_estimator(spec: ModelSpec, seed: int) -> Any:
    return ESTIMATORS[spec.type](**{"random_state": seed, **spec.params})


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
