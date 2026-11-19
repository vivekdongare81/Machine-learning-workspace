from pathlib import Path

import pandas as pd

from fraud_ml.predict import main, score_file
from fraud_ml.service import FraudScoringService
from tests.conftest import make_synthetic_frame


def test_batch_scoring_matches_online_scoring(tmp_path: Path, trained_artifacts: Path):
    frame = make_synthetic_frame(n_rows=120, seed=3).assign(transaction_id=range(120))
    input_path, output_path = tmp_path / "in.csv", tmp_path / "out" / "scores.csv"
    frame.to_csv(input_path, index=False)

    rows = score_file(input_path, output_path, trained_artifacts, chunksize=50)  # 3 chunks
    scored = pd.read_csv(output_path)

    assert rows == len(scored) == 120
    assert {"transaction_id", "Class", "fraud_score", "review_flag", "model_version"} <= set(
        scored.columns
    )
    service = FraudScoringService(trained_artifacts)
    online = service.score({k: float(frame.iloc[0][k]) for k in service.features})
    assert abs(online["fraud_score"] - scored.loc[0, "fraud_score"]) < 1e-9


def test_cli(tmp_path: Path, trained_artifacts: Path):
    input_path, output_path = tmp_path / "in.csv", tmp_path / "out.csv"
    make_synthetic_frame(n_rows=30).to_csv(input_path, index=False)
    args = [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--artifact-dir",
        str(trained_artifacts),
    ]
    assert main(args) == 0
    assert len(pd.read_csv(output_path)) == 30
