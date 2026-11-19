from churn_ml.config import ModelSpec
from churn_ml.data import FEATURE_COLUMNS, TARGET, validate_frame
from churn_ml.pipeline import build_estimator, make_pipeline
from tests.conftest import make_synthetic_frame


def _fitted_pipeline():
    frame = validate_frame(make_synthetic_frame(200))
    spec = ModelSpec(name="lr", type="logistic_regression", params={"max_iter": 500})
    model = make_pipeline(build_estimator(spec, seed=0))
    return model.fit(frame[FEATURE_COLUMNS], frame[TARGET]), frame


def test_identifier_and_target_are_not_features():
    assert "customerID" not in FEATURE_COLUMNS and TARGET not in FEATURE_COLUMNS


def test_pipeline_handles_unseen_categories():
    model, frame = _fitted_pipeline()
    row = frame[FEATURE_COLUMNS].iloc[[0]].copy()
    row.loc[row.index[0], "Contract"] = "Unseen contract type"
    probabilities = model.predict_proba(row)
    assert probabilities.shape == (1, 2) and 0 <= probabilities[0, 1] <= 1


def test_pipeline_imputes_missing_values():
    model, frame = _fitted_pipeline()
    row = frame[FEATURE_COLUMNS].iloc[[0]].copy()
    row.loc[row.index[0], ["TotalCharges", "PaymentMethod"]] = float("nan")
    assert 0 <= model.predict_proba(row)[0, 1] <= 1
