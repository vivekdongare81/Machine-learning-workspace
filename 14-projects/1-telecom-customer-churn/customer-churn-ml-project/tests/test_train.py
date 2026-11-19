import json
from pathlib import Path

import pytest
import yaml

from churn_ml.data import validate_frame
from churn_ml.train import (
    METADATA_FILE,
    MODEL_FILE,
    QualityGateError,
    main,
    run_training,
    stratified_split,
)
from tests.conftest import make_config, make_synthetic_frame


def test_stratified_split_preserves_churn_rate():
    frame = validate_frame(make_synthetic_frame(1000))
    train, valid, test = stratified_split(frame, 0.16, 0.20, seed=0)
    assert len(set(train) | set(valid) | set(test)) == len(frame)  # disjoint and complete
    rates = [frame.loc[idx, "Churn"].mean() for idx in (train, valid, test)]
    assert max(rates) - min(rates) < 0.02


def test_training_publishes_versioned_artifacts(tmp_path: Path, synthetic_csv: Path):
    cfg = make_config(tmp_path, synthetic_csv)
    result = run_training(cfg)

    artifact_dir = cfg.output.artifact_dir
    for name in (MODEL_FILE, METADATA_FILE):
        assert (artifact_dir / name).exists()  # "current" model served by the API
        assert (result.run_dir / name).exists()  # immutable copy for rollback
    metadata = json.loads((artifact_dir / METADATA_FILE).read_text())
    assert metadata["model_version"] == result.run_id
    assert metadata["excluded_columns"] == ["customerID"]
    assert len(metadata["data"]["sha256"]) == 64
    for report in (
        "cv_results.csv",
        "validation_results.csv",
        "threshold_sweep.csv",
        "segment_metrics.csv",
        "test_predictions.csv",
        "test_metrics.json",
    ):
        assert (cfg.output.report_dir / report).exists()


def test_failed_quality_gate_does_not_publish(tmp_path: Path, synthetic_csv: Path):
    cfg = make_config(
        tmp_path,
        synthetic_csv,
        quality_gates={"min_test_average_precision": 1.0, "min_test_recall": 1.0},
    )
    with pytest.raises(QualityGateError):
        run_training(cfg)
    assert not (cfg.output.artifact_dir / MODEL_FILE).exists()
    assert (cfg.output.report_dir / "test_metrics.json").exists()  # kept for investigation


def test_cli_returns_exit_codes(tmp_path: Path, synthetic_csv: Path):
    cfg = make_config(tmp_path, synthetic_csv)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    assert main(["--config", str(config_path)]) == 0

    cfg.quality_gates.min_test_recall = 1.0
    cfg.quality_gates.min_test_average_precision = 1.0
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    assert main(["--config", str(config_path)]) == 2

    cfg.data.sha256 = "0" * 64
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    assert main(["--config", str(config_path)]) == 3
