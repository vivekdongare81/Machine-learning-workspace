"""Model construction from config: preprocessing + estimator in one sklearn Pipeline.

Keeping preprocessing inside the pipeline means the exact same transformations
run in training, batch scoring and the API (no training/serving skew).
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from churn_ml.config import ModelSpec
from churn_ml.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ESTIMATORS: dict[str, Any] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def make_preprocessor() -> ColumnTransformer:
    """Statistics and category levels are learned from training data only.

    handle_unknown="ignore" lets the model score a category it never saw in
    training (all-zero one-hot) instead of failing the request.
    """
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def build_estimator(spec: ModelSpec, seed: int) -> Any:
    return ESTIMATORS[spec.type](**{"random_state": seed, **spec.params})


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
