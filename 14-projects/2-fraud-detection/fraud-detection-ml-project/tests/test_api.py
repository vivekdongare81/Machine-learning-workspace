from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_ml.api import create_app
from fraud_ml.settings import ServingSettings


@pytest.fixture
def client(trained_artifacts: Path):
    with TestClient(create_app(ServingSettings(artifact_dir=trained_artifacts))) as c:
        yield c


def test_health_endpoints(client: TestClient):
    assert client.get("/health/live").json() == {"status": "alive"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["model_version"]


def test_score_returns_contract(client: TestClient, feature_row: dict[str, float]):
    response = client.post("/v1/score", json={"features": feature_row})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"fraud_score", "review_flag", "threshold", "model_version"}
    assert 0 <= body["fraud_score"] <= 1


def test_request_id_is_propagated(client: TestClient):
    assert (
        client.get("/health/live", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"]
        == "abc123"
    )
    assert client.get("/health/live").headers["X-Request-ID"]  # generated when absent


@pytest.mark.parametrize(
    "payload",
    [
        {"features": {"V1": 1.0}},  # missing features
        {"features": {"V1": "not-a-number"}},  # wrong type
        {"features": {}, "unexpected": 1},  # unknown top-level field
    ],
)
def test_invalid_requests_return_422(client: TestClient, payload: dict):
    assert client.post("/v1/score", json=payload).status_code == 422


def test_batch_scoring_and_size_limit(trained_artifacts: Path, feature_row: dict[str, float]):
    settings = ServingSettings(artifact_dir=trained_artifacts, max_batch_size=2)
    with TestClient(create_app(settings)) as client:
        ok = client.post("/v1/score/batch", json={"instances": [feature_row, feature_row]})
        assert ok.status_code == 200 and len(ok.json()["predictions"]) == 2
        too_big = client.post("/v1/score/batch", json={"instances": [feature_row] * 3})
        assert too_big.status_code == 413


def test_model_info_and_metrics(client: TestClient, feature_row: dict[str, float]):
    info = client.get("/v1/model").json()
    assert info["features"] and info["decision_threshold"] is not None
    client.post("/v1/score", json={"features": feature_row})
    metrics = client.get("/metrics").text
    assert "fraud_api_requests_total" in metrics and "fraud_model_score_bucket" in metrics


def test_not_ready_without_model(tmp_path: Path, feature_row: dict[str, float]):
    with TestClient(create_app(ServingSettings(artifact_dir=tmp_path))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/score", json={"features": feature_row}).status_code == 503
