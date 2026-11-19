"""Typed training configuration loaded from YAML.

Pydantic validates the whole file before any work starts, so a typo or an
impossible value (e.g. split fractions that leave no training data) fails fast
instead of halfway through a training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    # Unknown keys are errors: a misspelt option must not be silently ignored.
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    url: str
    path: Path
    sha256: str | None = None


class SplitConfig(_Strict):
    validation_fraction: float = Field(0.16, gt=0, lt=1)
    test_fraction: float = Field(0.20, gt=0, lt=1)

    @model_validator(mode="after")
    def _leave_room_for_training(self) -> SplitConfig:
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError(
                "validation_fraction + test_fraction must be < 1 to leave training data"
            )
        return self


class CVConfig(_Strict):
    n_splits: int = Field(5, ge=2)


class ThresholdConfig(_Strict):
    false_negative_cost: float = Field(5.0, gt=0)
    false_positive_cost: float = Field(1.0, gt=0)


class ModelSpec(_Strict):
    name: str
    type: Literal["logistic_regression", "decision_tree", "random_forest"]
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(_Strict):
    segment_columns: list[str] = Field(default_factory=list)
    capacity_fraction: float = Field(0.10, gt=0, le=1)


class QualityGates(_Strict):
    min_test_average_precision: float = Field(0.0, ge=0, le=1)
    min_test_recall: float = Field(0.0, ge=0, le=1)


class OutputConfig(_Strict):
    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")


class TrackingConfig(_Strict):
    enabled: bool = False
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "customer-churn"


class TrainConfig(_Strict):
    seed: int = 42
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    models: list[ModelSpec] = Field(min_length=1)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    quality_gates: QualityGates = Field(default_factory=QualityGates)
    output: OutputConfig = Field(default_factory=OutputConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def _unique_model_names(self) -> TrainConfig:
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"model names must be unique, got {names}")
        return self


def load_config(path: str | Path) -> TrainConfig:
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return TrainConfig.model_validate(raw)
