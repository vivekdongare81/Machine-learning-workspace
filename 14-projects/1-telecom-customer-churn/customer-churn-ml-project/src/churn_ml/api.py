"""Online prediction API.

- /health/live and /health/ready for the orchestrator (restart vs. route traffic)
- /v1/predict and /v1/predict/batch, versioned so the contract can evolve safely
- /metrics for Prometheus: request counts, latency, and the prediction distribution
- every response carries X-Request-ID, which also appears in the access log

Run: uvicorn churn_ml.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from churn_ml import __version__
from churn_ml.logging_utils import configure_logging
from churn_ml.service import ChurnScoringService
from churn_ml.settings import ServingSettings

logger = logging.getLogger("churn_ml.api")

REQUESTS = Counter("churn_api_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("churn_api_request_seconds", "HTTP request latency", ["method", "route"])
# Watching the probability distribution and positive rate is a cheap early drift signal.
PROBABILITIES = Histogram(
    "churn_model_probability",
    "Predicted churn probabilities",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
PREDICTIONS = Counter("churn_model_predictions_total", "Predictions", ["predicted_churn"])

FeatureValue = str | float | None


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, FeatureValue] = Field(description="Every model feature, by name.")


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[dict[str, FeatureValue]] = Field(min_length=1)


class PredictResponse(BaseModel):
    churn_probability: float
    predicted_churn: bool
    threshold: float
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    settings = settings or ServingSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_format)
        try:
            app.state.service = ChurnScoringService(settings.artifact_dir)
            logger.info(
                "model loaded",
                extra={
                    "model_version": app.state.service.model_version,
                    "artifact_dir": str(settings.artifact_dir),
                },
            )
        except FileNotFoundError:
            # Stay up so liveness passes, but report not-ready until a model exists.
            app.state.service = None
            logger.warning(
                "no model artifact; readiness will fail",
                extra={"artifact_dir": str(settings.artifact_dir)},
            )
        yield

    app = FastAPI(
        title="Customer Churn Prediction API",
        version=__version__,
        description="Portfolio demonstration; not approved for real customer decisions.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start, status = time.perf_counter(), 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - start
            route = getattr(request.scope.get("route"), "path", "unmatched")  # low cardinality
            REQUESTS.labels(request.method, route, str(status)).inc()
            LATENCY.labels(request.method, route).observe(elapsed)
            logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status": status,
                    "latency_ms": round(elapsed * 1000, 2),
                },
            )
        response.headers["X-Request-ID"] = request_id
        return response

    def get_service(request: Request) -> ChurnScoringService:
        service: ChurnScoringService | None = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return service

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready(request: Request) -> dict[str, str]:
        return {"status": "ready", "model_version": get_service(request).model_version}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model")
    def model_info(request: Request) -> dict[str, Any]:
        meta = get_service(request).metadata
        keys = (
            "model_version", "selected_model", "created_at_utc", "decision_threshold",
            "features", "numeric_features", "categorical_features", "test_metrics", "git_commit",
        )  # fmt: skip
        return {key: meta.get(key) for key in keys}

    def _predict(request: Request, rows: list[dict[str, FeatureValue]]) -> list[PredictResponse]:
        service = get_service(request)
        try:
            results = service.score_many(rows)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        for result in results:
            PROBABILITIES.observe(result["churn_probability"])
            PREDICTIONS.labels(str(result["predicted_churn"]).lower()).inc()
        return [PredictResponse(**r) for r in results]

    @app.post("/v1/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        return _predict(request, [body.features])[0]

    @app.post("/v1/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
        if len(body.instances) > settings.max_batch_size:
            raise HTTPException(
                status_code=413, detail=f"batch larger than {settings.max_batch_size} instances"
            )
        return BatchPredictResponse(predictions=_predict(request, body.instances))

    return app


app = create_app()
