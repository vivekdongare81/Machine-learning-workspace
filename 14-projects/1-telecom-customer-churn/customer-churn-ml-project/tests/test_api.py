from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from churn_ml.api import create_app
from churn_ml.settings import ServingSettings


@pytest.fixture
def client(trained_artifacts: Path):
    with TestClient(create_app(ServingSettings(artifact_dir=trained_artifacts))) as c:
        yield c


def test_health_endpoints(client: TestClient):
    assert client.get("/health/live").json() == {"status": "alive"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["model_version"]


def test_predict_returns_contract(client: TestClient, feature_row: dict[str, Any]):
    response = client.post("/v1/predict", json={"features": feature_row})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "predicted_churn", "threshold", "model_version"}
    assert 0 <= body["churn_probability"] <= 1


def test_request_id_is_propagated(client: TestClient):
    response = client.get("/health/live", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert client.get("/health/live").headers["X-Request-ID"]  # generated when absent


@pytest.mark.parametrize(
    "change",
    [
        {"drop": "tenure"},  # missing feature
        {"set": ("tenure", "twelve")},  # wrong type
        {"set": ("customerID", "C-1")},  # identifier is not a feature
    ],
)
def test_invalid_features_return_422(
    client: TestClient, feature_row: dict[str, Any], change: dict[str, Any]
):
    row = dict(feature_row)
    if "drop" in change:
        row.pop(change["drop"])
    else:
        key, value = change["set"]
        row[key] = value
    assert client.post("/v1/predict", json={"features": row}).status_code == 422


def test_unknown_top_level_field_returns_422(client: TestClient, feature_row: dict[str, Any]):
    response = client.post("/v1/predict", json={"features": feature_row, "extra": 1})
    assert response.status_code == 422


def test_batch_and_size_limit(trained_artifacts: Path, feature_row: dict[str, Any]):
    settings = ServingSettings(artifact_dir=trained_artifacts, max_batch_size=2)
    with TestClient(create_app(settings)) as client:
        ok = client.post("/v1/predict/batch", json={"instances": [feature_row, feature_row]})
        assert ok.status_code == 200 and len(ok.json()["predictions"]) == 2
        too_big = client.post("/v1/predict/batch", json={"instances": [feature_row] * 3})
        assert too_big.status_code == 413


def test_model_info_and_metrics(client: TestClient, feature_row: dict[str, Any]):
    info = client.get("/v1/model").json()
    assert info["features"] and info["decision_threshold"] is not None
    client.post("/v1/predict", json={"features": feature_row})
    metrics = client.get("/metrics").text
    assert "churn_api_requests_total" in metrics and "churn_model_probability_bucket" in metrics


def test_not_ready_without_model(tmp_path: Path, feature_row: dict[str, Any]):
    with TestClient(create_app(ServingSettings(artifact_dir=tmp_path))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/predict", json={"features": feature_row}).status_code == 503
