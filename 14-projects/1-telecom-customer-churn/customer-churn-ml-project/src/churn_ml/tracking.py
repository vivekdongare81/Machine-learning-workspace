"""Optional experiment tracking.

Training talks to a small Tracker interface. With tracking disabled it is a
no-op, so MLflow stays an optional dependency (`pip install -e ".[tracking]"`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from churn_ml.config import TrackingConfig


class NullTracker:
    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifacts(self, directory: Path) -> None:
        pass

    def set_tags(self, tags: dict[str, str]) -> None:
        pass


class MlflowTracker(NullTracker):
    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_artifacts(self, directory: Path) -> None:
        self._mlflow.log_artifacts(str(directory))

    def set_tags(self, tags: dict[str, str]) -> None:
        self._mlflow.set_tags(tags)


@contextmanager
def tracking_run(config: TrackingConfig, run_name: str) -> Iterator[NullTracker]:
    if not config.enabled:
        yield NullTracker()
        return
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            'tracking.enabled is true but MLflow is not installed: pip install -e ".[tracking]"'
        ) from exc
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=run_name):
        yield MlflowTracker(mlflow)
