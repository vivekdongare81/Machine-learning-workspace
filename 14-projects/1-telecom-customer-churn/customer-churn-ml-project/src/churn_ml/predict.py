"""Batch (offline) scoring: score a CSV of customers in chunks and write the results.

Batch suits scheduled jobs such as a weekly churn-risk list for the retention
team; the online API suits one decision that needs an answer in milliseconds.
Both use the same ChurnScoringService, so their scores are identical.

Usage:
    python -m churn_ml.predict --input data/customers.csv --output reports/scores.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from churn_ml.logging_utils import configure_logging
from churn_ml.service import ChurnScoringService

logger = logging.getLogger(__name__)


def score_file(
    input_path: Path, output_path: Path, artifact_dir: Path, chunksize: int = 50_000
) -> int:
    """Score every row; non-feature columns (customerID, labels) pass through unchanged."""
    service = ChurnScoringService(artifact_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
        features = chunk[service.features].copy()
        # Same coercion as training: blank TotalCharges etc. become NaN and are imputed.
        features[service.numeric_features] = features[service.numeric_features].apply(
            pd.to_numeric, errors="coerce"
        )
        scored = service.score_frame(features)
        passthrough = chunk.drop(columns=service.features)
        out = pd.concat([passthrough, scored], axis=1).assign(model_version=service.model_version)
        out.to_csv(output_path, mode="w" if i == 0 else "a", header=i == 0, index=False)
        rows += len(chunk)
    logger.info(
        "batch scoring finished",
        extra={"rows": rows, "output": str(output_path), "model_version": service.model_version},
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a CSV file with the published model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    score_file(args.input, args.output, args.artifact_dir, args.chunksize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
