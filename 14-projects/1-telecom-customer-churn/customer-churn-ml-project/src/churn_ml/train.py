"""Training pipeline.

config -> load + validate data (checksum) -> stratified train/validation/test split ->
CV on training data -> pick model on validation -> pick threshold on validation ->
evaluate once on test (incl. segments and capacity) -> quality gates -> publish artifacts.

Usage:
    python -m churn_ml.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from churn_ml import __version__
from churn_ml.config import QualityGates, TrainConfig, load_config
from churn_ml.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET,
    DataValidationError,
    load_data,
)
from churn_ml.logging_utils import configure_logging
from churn_ml.metrics import (
    classification_metrics,
    segment_metrics,
    select_threshold,
    threshold_metrics,
    top_k_metrics,
)
from churn_ml.pipeline import build_estimator, make_pipeline
from churn_ml.tracking import tracking_run

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class QualityGateError(RuntimeError):
    """The trained model did not meet the release criteria; it was not published."""


@dataclass
class TrainingResult:
    run_id: str
    selected_model: str
    threshold: float
    test_metrics: dict[str, float]
    run_dir: Path


def stratified_split(
    frame: pd.DataFrame, validation_fraction: float, test_fraction: float, seed: int
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Random split that keeps the churn rate equal across train, validation and test."""
    dev_idx, test_idx = train_test_split(
        frame.index, test_size=test_fraction, stratify=frame[TARGET], random_state=seed
    )
    train_idx, valid_idx = train_test_split(
        dev_idx,
        test_size=validation_fraction / (1 - test_fraction),
        stratify=frame.loc[dev_idx, TARGET],
        random_state=seed,
    )
    return pd.Index(train_idx), pd.Index(valid_idx), pd.Index(test_idx)


def check_quality_gates(test_metrics: dict[str, float], gates: QualityGates) -> list[str]:
    failures = []
    if test_metrics["average_precision"] < gates.min_test_average_precision:
        failures.append(
            f"average_precision {test_metrics['average_precision']:.4f} "
            f"< {gates.min_test_average_precision}"
        )
    if test_metrics["recall"] < gates.min_test_recall:
        failures.append(f"recall {test_metrics['recall']:.4f} < {gates.min_test_recall}")
    return failures


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_training(cfg: TrainConfig) -> TrainingResult:
    run_id = _new_run_id()
    logger.info("training run started", extra={"run_id": run_id})
    costs = (cfg.threshold.false_negative_cost, cfg.threshold.false_positive_cost)

    frame, data_sha256 = load_data(cfg.data.path, cfg.data.url, cfg.data.sha256)
    X, y = frame[FEATURE_COLUMNS], frame[TARGET].astype(int)
    train_idx, valid_idx, test_idx = stratified_split(
        frame, cfg.split.validation_fraction, cfg.split.test_fraction, cfg.seed
    )
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    split_summary = {
        name: {"rows": len(part), "churn_rate": float(part.mean())}
        for name, part in (("train", y_train), ("validation", y_valid), ("test", y_test))
    }
    logger.info("stratified split", extra={"split": split_summary})

    with tracking_run(cfg.tracking, run_name=run_id) as tracker:
        tracker.set_tags({"run_id": run_id, "data_sha256": data_sha256})
        tracker.log_params(
            {
                "seed": cfg.seed,
                "cv_n_splits": cfg.cv.n_splits,
                "fn_cost": costs[0],
                "fp_cost": costs[1],
                "candidates": ",".join(m.name for m in cfg.models),
            }
        )

        # A majority-class baseline gives context: decent accuracy, zero churners found.
        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        baseline_metrics = threshold_metrics(y_valid, baseline.predict(X_valid), 0.5)

        cv = StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=True, random_state=cfg.seed)
        cv_rows, val_rows, fitted = [], [], {}
        for spec in cfg.models:
            pipe = make_pipeline(build_estimator(spec, cfg.seed))
            scores = cross_validate(
                pipe, X_train, y_train, cv=cv, scoring={"ap": "average_precision", "auc": "roc_auc"}
            )
            cv_rows.append(
                {
                    "model": spec.name,
                    "cv_average_precision_mean": float(scores["test_ap"].mean()),
                    "cv_average_precision_std": float(scores["test_ap"].std()),
                    "cv_roc_auc_mean": float(scores["test_auc"].mean()),
                }
            )
            pipe.fit(X_train, y_train)
            valid_scores = pipe.predict_proba(X_valid)[:, 1]
            val = classification_metrics(y_valid, valid_scores, 0.5)
            val_rows.append(
                {
                    "model": spec.name,
                    "validation_average_precision": val["average_precision"],
                    "validation_roc_auc": val["roc_auc"],
                    "validation_brier": val["brier_score"],
                }
            )
            fitted[spec.name] = (pipe, valid_scores)
            logger.info("candidate evaluated", extra={**cv_rows[-1], **val_rows[-1]})

        cv_results = pd.DataFrame(cv_rows).sort_values("cv_average_precision_mean", ascending=False)
        val_results = pd.DataFrame(val_rows).sort_values(
            "validation_average_precision", ascending=False
        )
        best_name = str(val_results.iloc[0]["model"])
        model, valid_scores = fitted[best_name]

        threshold, validation_cost = select_threshold(y_valid, valid_scores, *costs)
        sweep = pd.DataFrame(
            [threshold_metrics(y_valid, valid_scores, t, *costs) for t in np.linspace(0, 1, 101)]
        )
        logger.info(
            "model and threshold selected",
            extra={"model": best_name, "threshold": threshold, "cost": validation_cost},
        )

        # The test set is touched exactly once, after every choice is fixed.
        test_scores = model.predict_proba(X_test)[:, 1]
        capacity = top_k_metrics(y_test, test_scores, cfg.evaluation.capacity_fraction)
        test_metrics = {
            **classification_metrics(y_test, test_scores, threshold, *costs),
            **{f"capacity_{k}": v for k, v in capacity.items()},
        }
        gate_failures = check_quality_gates(test_metrics, cfg.quality_gates)

        report_dir = cfg.output.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(report_dir / "cv_results.csv", index=False)
        val_results.to_csv(report_dir / "validation_results.csv", index=False)
        sweep.to_csv(report_dir / "threshold_sweep.csv", index=False)
        segment_metrics(
            X_test[cfg.evaluation.segment_columns], y_test, test_scores, threshold
        ).to_csv(report_dir / "segment_metrics.csv", index=False)
        predictions = frame.loc[test_idx, [ID_COLUMN]].assign(
            actual_churn=y_test.to_numpy(),
            churn_probability=test_scores,
            predicted_churn=(test_scores >= threshold).astype(int),
        )
        predictions["error_type"] = np.select(
            [
                (predictions.actual_churn == 1) & (predictions.predicted_churn == 0),
                (predictions.actual_churn == 0) & (predictions.predicted_churn == 1),
            ],
            ["false_negative", "false_positive"],
            default="correct",
        )
        predictions.to_csv(report_dir / "test_predictions.csv", index=False)
        _write_json(
            report_dir / "test_metrics.json", {"run_id": run_id, "model": best_name, **test_metrics}
        )
        tracker.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
        tracker.log_artifacts(report_dir)

        if gate_failures:
            tracker.set_tags({"quality_gates": "failed"})
            raise QualityGateError("; ".join(gate_failures))

        metadata = {
            "project": "customer-churn-ml",
            "package_version": __version__,
            "model_version": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "selected_model": best_name,
            "model_params": next(m.params for m in cfg.models if m.name == best_name),
            "features": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "excluded_columns": [ID_COLUMN],
            "target": TARGET,
            "decision_threshold": threshold,
            "threshold_cost_assumptions": cfg.threshold.model_dump(),
            "data": {
                "source": cfg.data.url,
                "sha256": data_sha256,
                "rows": len(frame),
                "churn_rate": float(y.mean()),
            },
            "split": {"strategy": "stratified random", **split_summary},
            "baseline_validation": baseline_metrics,
            "validation_cost_at_threshold": validation_cost,
            "test_metrics": test_metrics,
            "quality_gates": {**cfg.quality_gates.model_dump(), "passed": True},
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "limitations": [
                "Public sample dataset, not production customer data",
                "No scoring timestamp, so a random (not time-based) split is used",
                "Threshold costs are illustrative",
                "Predicts churn risk, not who a retention offer would persuade (uplift)",
            ],
        }

        # Immutable history under runs/<run_id>/; the top-level copy is what serving loads.
        # Rolling back = pointing CHURN_ARTIFACT_DIR at an earlier runs/<run_id>/.
        run_dir = cfg.output.artifact_dir / "runs" / run_id
        run_dir.mkdir(parents=True)
        joblib.dump(model, run_dir / MODEL_FILE)
        _write_json(run_dir / METADATA_FILE, metadata)
        for name in (MODEL_FILE, METADATA_FILE):
            shutil.copy2(run_dir / name, cfg.output.artifact_dir / name)
        tracker.log_artifacts(run_dir)
        tracker.set_tags({"quality_gates": "passed"})

    logger.info("model published", extra={"run_id": run_id, "path": str(run_dir)})
    return TrainingResult(run_id, best_name, threshold, test_metrics, run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and publish the churn risk model.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--data-path", type=Path, help="override data.path from the config")
    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override tracking.enabled from the config",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)

    configure_logging(args.log_level, args.log_format)
    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg.data.path = args.data_path
    if args.tracking is not None:
        cfg.tracking.enabled = args.tracking

    try:
        result = run_training(cfg)
    except DataValidationError as exc:
        logger.error("data validation failed", extra={"reason": str(exc)})
        return 3
    except QualityGateError as exc:
        logger.error("quality gates failed; model NOT published", extra={"reasons": str(exc)})
        return 2
    summary = {
        "run_id": result.run_id,
        "model": result.selected_model,
        "threshold": result.threshold,
        "test_metrics": result.test_metrics,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
