"""Scoring service shared by the online API and batch scoring.

It loads one trusted artifact (joblib is pickle-based: never load untrusted
files) and enforces the feature contract recorded in its metadata.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class ChurnScoringService:
    def __init__(self, artifact_dir: str | Path = "artifacts") -> None:
        artifact_dir = Path(artifact_dir)
        model_path, metadata_path = artifact_dir / MODEL_FILE, artifact_dir / METADATA_FILE
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"no model in {artifact_dir}; run training first (make train)")
        self.model = joblib.load(model_path)
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.features: list[str] = list(self.metadata["features"])
        self.numeric_features: list[str] = list(self.metadata["numeric_features"])
        self.threshold = float(self.metadata["decision_threshold"])
        self.model_version = str(self.metadata.get("model_version", "unknown"))

    def check_row(self, row: Mapping[str, Any]) -> None:
        """Exact feature set; numbers for numeric features, text for categorical. Nulls allowed
        (the pipeline imputes them), and unseen categories are allowed (ignored by the encoder)."""
        missing = sorted(set(self.features) - set(row))
        extra = sorted(set(row) - set(self.features))
        if missing or extra:
            raise ValueError(f"feature schema mismatch; missing={missing}, extra={extra}")
        numeric = set(self.numeric_features)
        bad_numeric = [k for k in numeric if row[k] is not None and not _is_finite_number(row[k])]
        bad_text = [
            k for k in self.features if k not in numeric and row[k] is not None
            and not isinstance(row[k], str)
        ]  # fmt: skip
        if bad_numeric:
            raise ValueError(f"features must be finite numbers or null: {sorted(bad_numeric)}")
        if bad_text:
            raise ValueError(f"features must be text or null: {bad_text}")

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Vectorised scoring for batches; `frame` must contain every model feature."""
        missing = sorted(set(self.features) - set(frame.columns))
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        probabilities = self.model.predict_proba(frame[self.features])[:, 1]
        return pd.DataFrame(
            {
                "churn_probability": probabilities,
                "predicted_churn": probabilities >= self.threshold,
            },
            index=frame.index,
        )

    def score_many(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            self.check_row(row)
        frame = pd.DataFrame([{k: r[k] for k in self.features} for r in rows])
        # JSON null -> NaN, which the pipeline's imputers treat as missing (None is not).
        frame = frame.astype(object).where(frame.notna(), np.nan)
        frame[self.numeric_features] = frame[self.numeric_features].astype(float)
        scored = self.score_frame(frame)
        return [
            {
                "churn_probability": float(p),
                "predicted_churn": bool(flag),
                "threshold": self.threshold,
                "model_version": self.model_version,
            }
            for p, flag in zip(scored["churn_probability"], scored["predicted_churn"], strict=True)
        ]

    def score(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self.score_many([row])[0]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
