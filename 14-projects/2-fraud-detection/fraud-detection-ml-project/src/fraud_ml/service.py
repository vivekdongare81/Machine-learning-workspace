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
import pandas as pd

MODEL_FILE = "fraud_pipeline.joblib"
METADATA_FILE = "model_metadata.json"


class FraudScoringService:
    def __init__(self, artifact_dir: str | Path = "artifacts") -> None:
        artifact_dir = Path(artifact_dir)
        model_path, metadata_path = artifact_dir / MODEL_FILE, artifact_dir / METADATA_FILE
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"no model in {artifact_dir}; run training first (make train)")
        self.model = joblib.load(model_path)
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.features: list[str] = list(self.metadata["features"])
        self.threshold = float(self.metadata["decision_threshold"])
        self.model_version = str(self.metadata.get("model_version", "unknown"))

    def check_row(self, row: Mapping[str, Any]) -> None:
        missing = sorted(set(self.features) - set(row))
        extra = sorted(set(row) - set(self.features))
        if missing or extra:
            raise ValueError(f"feature schema mismatch; missing={missing}, extra={extra}")
        bad = [k for k in self.features if not _is_finite_number(row[k])]
        if bad:
            raise ValueError(f"features must be finite numbers: {bad}")

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Vectorised scoring for batches; `frame` must contain every model feature."""
        missing = sorted(set(self.features) - set(frame.columns))
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        scores = self.model.predict_proba(frame[self.features])[:, 1]
        return pd.DataFrame(
            {"fraud_score": scores, "review_flag": scores >= self.threshold}, index=frame.index
        )

    def score_many(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            self.check_row(row)
        frame = pd.DataFrame([{k: float(r[k]) for k in self.features} for r in rows])
        scored = self.score_frame(frame)
        return [
            {
                "fraud_score": float(score),
                "review_flag": bool(flag),
                "threshold": self.threshold,
                "model_version": self.model_version,
            }
            for score, flag in zip(scored["fraud_score"], scored["review_flag"], strict=True)
        ]

    def score(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self.score_many([row])[0]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
