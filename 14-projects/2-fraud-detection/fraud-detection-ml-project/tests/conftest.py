"""Shared fixtures: a small synthetic dataset with the real schema, and a model trained on it.

Tests train through the real `run_training`, so the artifact format that serving
reads is exactly what training writes (the train/serve contract is tested).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from fraud_ml.config import TrainConfig
from fraud_ml.data import FEATURE_COLUMNS, TARGET
from fraud_ml.train import run_training


def make_synthetic_frame(n_rows: int = 800, seed: int = 0) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_rows,
        n_features=len(FEATURE_COLUMNS),
        n_informative=8,
        weights=[0.9],
        random_state=seed,
    )
    frame = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    frame["Time"] = np.sort(np.random.default_rng(seed).uniform(0, 10_000, n_rows))
    frame["Amount"] = np.abs(frame["Amount"]) * 50
    frame[TARGET] = y
    return frame


def make_config(tmp_dir: Path, data_path: Path, **overrides: object) -> TrainConfig:
    raw: dict[str, object] = {
        "data": {"url": "https://example.invalid/unused.csv", "path": str(data_path)},
        "cv": {"strategy": "time_series", "n_splits": 2},
        "models": [
            {
                "name": "logistic_regression",
                "type": "logistic_regression",
                "params": {"max_iter": 500, "class_weight": "balanced"},
            },
            {
                "name": "random_forest",
                "type": "random_forest",
                "params": {"n_estimators": 20, "min_samples_leaf": 2},
            },
        ],
        "output": {
            "artifact_dir": str(tmp_dir / "artifacts"),
            "report_dir": str(tmp_dir / "reports"),
        },
    }
    raw.update(overrides)
    return TrainConfig.model_validate(raw)


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "creditcard.csv"
    path.parent.mkdir(parents=True)
    make_synthetic_frame().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train once per test session and share the published artifact directory."""
    root = tmp_path_factory.mktemp("trained")
    data_path = root / "creditcard.csv"
    make_synthetic_frame().to_csv(data_path, index=False)
    cfg = make_config(root, data_path)
    run_training(cfg)
    return cfg.output.artifact_dir


@pytest.fixture
def feature_row() -> dict[str, float]:
    row = make_synthetic_frame(n_rows=50, seed=1).iloc[0]
    return {name: float(row[name]) for name in FEATURE_COLUMNS}
