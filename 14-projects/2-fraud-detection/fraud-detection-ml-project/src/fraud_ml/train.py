"""Training pipeline.

config -> load + validate data (checksum) -> time-ordered split -> CV on the
training period -> pick model on validation -> pick threshold on validation ->
evaluate once on the latest (test) period -> quality gates -> publish artifacts.

Usage:
    python -m fraud_ml.train --config configs/train.yaml
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
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate

from fraud_ml import __version__
from fraud_ml.config import QualityGates, TrainConfig, load_config
from fraud_ml.data import FEATURE_COLUMNS, TARGET, DataValidationError, load_data
from fraud_ml.logging_utils import configure_logging
from fraud_ml.metrics import (
    classification_metrics,
    select_threshold,
    threshold_metrics,
    top_k_metrics,
)
from fraud_ml.pipeline import build_estimator, make_pipeline
from fraud_ml.tracking import tracking_run

logger = logging.getLogger(__name__)

MODEL_FILE = "fraud_pipeline.joblib"
METADATA_FILE = "model_metadata.json"


class QualityGateError(RuntimeError):
    """The trained model did not meet the release criteria; it was not published."""


@dataclass
class TrainingResult:
    run_id: str
    selected_model: str
    threshold: float
    test_metrics: dict[str, float]
    run_dir: Path


def time_ordered_split(
    frame: pd.DataFrame, time_column: str, train_fraction: float, validation_fraction: float
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Earliest rows train, the next block validates, the latest block tests."""
    ordered = frame.sort_values(time_column, kind="stable").index
    n = len(ordered)
    train_end = int(n * train_fraction)
    valid_end = int(n * (train_fraction + validation_fraction))
    if min(train_end, valid_end - train_end, n - valid_end) == 0:
        raise ValueError(f"{n} rows is too few for the configured split")
    return ordered[:train_end], ordered[train_end:valid_end], ordered[valid_end:]


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

    frame, data_sha256 = load_data(cfg.data.path, cfg.data.url, cfg.data.sha256)
    X, y = frame[FEATURE_COLUMNS], frame[TARGET].astype(int)
    train_idx, valid_idx, test_idx = time_ordered_split(
        frame, cfg.split.time_column, cfg.split.train_fraction, cfg.split.validation_fraction
    )
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    split_summary = {
        name: {"rows": len(part), "fraud_count": int(part.sum()), "fraud_rate": float(part.mean())}
        for name, part in (("train", y_train), ("validation", y_valid), ("test", y_test))
    }
    logger.info("time-ordered split", extra={"split": split_summary})

    with tracking_run(cfg.tracking, run_name=run_id) as tracker:
        tracker.set_tags({"run_id": run_id, "data_sha256": data_sha256})
        tracker.log_params(
            {
                "seed": cfg.seed,
                "cv_strategy": cfg.cv.strategy,
                "cv_n_splits": cfg.cv.n_splits,
                "fn_cost": cfg.threshold.false_negative_cost,
                "fp_cost": cfg.threshold.false_positive_cost,
                "candidates": ",".join(m.name for m in cfg.models),
            }
        )

        # A majority-class baseline gives context: high accuracy, zero fraud caught.
        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        baseline_metrics = threshold_metrics(y_valid, baseline.predict(X_valid), 0.5)

        cv = (
            TimeSeriesSplit(n_splits=cfg.cv.n_splits)
            if cfg.cv.strategy == "time_series"
            else StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=True, random_state=cfg.seed)
        )

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

        threshold, validation_cost = select_threshold(
            y_valid,
            valid_scores,
            cfg.threshold.false_negative_cost,
            cfg.threshold.false_positive_cost,
        )
        sweep = pd.DataFrame(
            [
                threshold_metrics(
                    y_valid,
                    valid_scores,
                    t,
                    cfg.threshold.false_negative_cost,
                    cfg.threshold.false_positive_cost,
                )
                for t in np.linspace(0.01, 0.99, 99)
            ]
        )
        logger.info(
            "model and threshold selected",
            extra={"model": best_name, "threshold": threshold, "cost": validation_cost},
        )

        # The test period is touched exactly once, after every choice is fixed.
        test_scores = model.predict_proba(X_test)[:, 1]
        test_metrics = {
            **classification_metrics(
                y_test,
                test_scores,
                threshold,
                cfg.threshold.false_negative_cost,
                cfg.threshold.false_positive_cost,
            ),
            **{f"top1pct_{k}": v for k, v in top_k_metrics(y_test, test_scores, 0.01).items()},
        }
        gate_failures = check_quality_gates(test_metrics, cfg.quality_gates)

        report_dir = cfg.output.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(report_dir / "cv_results.csv", index=False)
        val_results.to_csv(report_dir / "validation_results.csv", index=False)
        sweep.to_csv(report_dir / "threshold_sweep.csv", index=False)
        _write_json(
            report_dir / "test_metrics.json", {"run_id": run_id, "model": best_name, **test_metrics}
        )
        tracker.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
        tracker.log_artifacts(report_dir)

        if gate_failures:
            tracker.set_tags({"quality_gates": "failed"})
            raise QualityGateError("; ".join(gate_failures))

        metadata = {
            "project": "fraud-detection-ml",
            "package_version": __version__,
            "model_version": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "selected_model": best_name,
            "model_params": next(m.params for m in cfg.models if m.name == best_name),
            "features": FEATURE_COLUMNS,
            "target": TARGET,
            "decision_threshold": threshold,
            "threshold_cost_assumptions": cfg.threshold.model_dump(),
            "data": {
                "source": cfg.data.url,
                "sha256": data_sha256,
                "rows": len(frame),
                "fraud_rate": float(y.mean()),
            },
            "split": {"strategy": "time-ordered", **split_summary},
            "cv_strategy": cfg.cv.strategy,
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
                "Public benchmark, not production data",
                "Time is elapsed seconds, not a calendar timestamp",
                "Anonymized V features limit domain interpretation",
                "Threshold costs are illustrative",
            ],
        }

        # Immutable history under runs/<run_id>/; the top-level copy is what serving loads.
        # Rolling back = pointing FRAUD_ARTIFACT_DIR at an earlier runs/<run_id>/.
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
    parser = argparse.ArgumentParser(description="Train and publish the fraud risk model.")
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
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "model": result.selected_model,
                "threshold": result.threshold,
                "test_metrics": result.test_metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
