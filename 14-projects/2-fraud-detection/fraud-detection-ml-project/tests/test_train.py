import json
from pathlib import Path

import pytest
import yaml

from fraud_ml.train import (
    METADATA_FILE,
    MODEL_FILE,
    QualityGateError,
    main,
    run_training,
    time_ordered_split,
)
from tests.conftest import make_config, make_synthetic_frame


def test_time_ordered_split_keeps_future_out_of_training():
    frame = make_synthetic_frame(100).sample(frac=1, random_state=0)  # shuffled rows
    train, valid, test = time_ordered_split(frame, "Time", 0.6, 0.2)
    assert frame.loc[train, "Time"].max() <= frame.loc[valid, "Time"].min()
    assert frame.loc[valid, "Time"].max() <= frame.loc[test, "Time"].min()
    assert len(train) + len(valid) + len(test) == len(frame)


def test_training_publishes_versioned_artifacts(tmp_path: Path, synthetic_csv: Path):
    cfg = make_config(tmp_path, synthetic_csv)
    result = run_training(cfg)

    artifact_dir = cfg.output.artifact_dir
    for name in (MODEL_FILE, METADATA_FILE):
        assert (artifact_dir / name).exists()  # "current" model served by the API
        assert (result.run_dir / name).exists()  # immutable copy for rollback
    metadata = json.loads((artifact_dir / METADATA_FILE).read_text())
    assert metadata["model_version"] == result.run_id
    assert len(metadata["data"]["sha256"]) == 64
    assert metadata["quality_gates"]["passed"] is True
    for report in (
        "cv_results.csv",
        "validation_results.csv",
        "threshold_sweep.csv",
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
