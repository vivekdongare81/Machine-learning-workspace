"""Shared fixtures: a small synthetic dataset with the real schema, and a model trained on it.

Tests train through the real `run_training`, so the artifact format that serving
reads is exactly what training writes (the train/serve contract is tested).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from churn_ml.config import TrainConfig
from churn_ml.data import FEATURE_COLUMNS, NUMERIC_FEATURES
from churn_ml.train import run_training

_SERVICE = ["Yes", "No", "No internet service"]
LEVELS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _SERVICE,
    "OnlineBackup": _SERVICE,
    "DeviceProtection": _SERVICE,
    "TechSupport": _SERVICE,
    "StreamingTV": _SERVICE,
    "StreamingMovies": _SERVICE,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}  # fmt: skip


def make_synthetic_frame(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Raw-format rows: Yes/No target and TotalCharges as text with blanks, like the real CSV."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({col: rng.choice(levels, n_rows) for col, levels in LEVELS.items()})
    frame["SeniorCitizen"] = rng.integers(0, 2, n_rows)
    frame["tenure"] = rng.integers(0, 73, n_rows)
    frame["MonthlyCharges"] = rng.uniform(18, 120, n_rows).round(2)
    total = (frame["tenure"] * frame["MonthlyCharges"]).round(2).astype(str)
    frame["TotalCharges"] = total.where(frame["tenure"] > 0, " ")
    logit = (
        -1.0
        + 1.5 * (frame["Contract"] == "Month-to-month")
        + 0.8 * (frame["InternetService"] == "Fiber optic")
        - 0.04 * frame["tenure"]
    )
    frame["Churn"] = np.where(rng.random(n_rows) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    frame.insert(0, "customerID", [f"C{seed}-{i:05d}" for i in range(n_rows)])
    return frame


def make_config(tmp_dir: Path, data_path: Path, **overrides: object) -> TrainConfig:
    raw: dict[str, object] = {
        "data": {"url": "https://example.invalid/unused.csv", "path": str(data_path)},
        "cv": {"n_splits": 3},
        "models": [
            {
                "name": "logistic_regression",
                "type": "logistic_regression",
                "params": {"max_iter": 1000, "class_weight": "balanced"},
            },
            {"name": "decision_tree", "type": "decision_tree", "params": {"max_depth": 4}},
        ],
        "evaluation": {"segment_columns": ["Contract", "InternetService"]},
        "output": {
            "artifact_dir": str(tmp_dir / "artifacts"),
            "report_dir": str(tmp_dir / "reports"),
        },
    }
    raw.update(overrides)
    return TrainConfig.model_validate(raw)


def to_feature_row(raw_row: pd.Series) -> dict[str, Any]:
    """A raw CSV row as the API expects it: numbers for numeric features, None for blanks."""
    row: dict[str, Any] = {}
    for name in FEATURE_COLUMNS:
        value = raw_row[name]
        if name in NUMERIC_FEATURES:
            number = pd.to_numeric(value, errors="coerce")
            row[name] = None if pd.isna(number) else float(number)
        else:
            row[name] = str(value)
    return row


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "churn.csv"
    path.parent.mkdir(parents=True)
    make_synthetic_frame().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train once per test session and share the published artifact directory."""
    root = tmp_path_factory.mktemp("trained")
    data_path = root / "churn.csv"
    make_synthetic_frame().to_csv(data_path, index=False)
    cfg = make_config(root, data_path)
    run_training(cfg)
    return cfg.output.artifact_dir


@pytest.fixture
def feature_row() -> dict[str, Any]:
    frame = make_synthetic_frame(n_rows=20, seed=1)
    return to_feature_row(frame[frame["tenure"] > 0].iloc[0])
