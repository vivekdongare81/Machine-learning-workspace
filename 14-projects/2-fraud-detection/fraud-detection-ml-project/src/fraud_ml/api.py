"""Online scoring API.

- /health/live and /health/ready for the orchestrator (restart vs. route traffic)
- /v1/score and /v1/score/batch, versioned so the contract can evolve safely
- /metrics for Prometheus: request counts, latency, and the score distribution
- every response carries X-Request-ID, which also appears in the access log

Run: uvicorn fraud_ml.api:app --host 127.0.0.1 --port 8000
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

from fraud_ml import __version__
from fraud_ml.logging_utils import configure_logging
from fraud_ml.service import FraudScoringService
from fraud_ml.settings import ServingSettings

logger = logging.getLogger("fraud_ml.api")

REQUESTS = Counter("fraud_api_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("fraud_api_request_seconds", "HTTP request latency", ["method", "route"])
# Watching the score distribution and flag rate is a cheap early drift signal.
SCORES = Histogram(
    "fraud_model_score",
    "Model fraud scores",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)
FLAGS = Counter("fraud_model_review_flags_total", "Scored transactions", ["flagged"])


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(description="Every model feature, by name.")


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[dict[str, float]] = Field(min_length=1)


class ScoreResponse(BaseModel):
    fraud_score: float
    review_flag: bool
    threshold: float
    model_version: str


class BatchScoreResponse(BaseModel):
    predictions: list[ScoreResponse]


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    settings = settings or ServingSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_format)
        try:
            app.state.service = FraudScoringService(settings.artifact_dir)
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

    app = FastAPI(title="Fraud Risk Scoring API", version=__version__, lifespan=lifespan)

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

    def get_service(request: Request) -> FraudScoringService:
        service: FraudScoringService | None = getattr(request.app.state, "service", None)
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
        return {
            key: meta.get(key)
            for key in (
                "model_version",
                "selected_model",
                "created_at_utc",
                "decision_threshold",
                "features",
                "test_metrics",
                "git_commit",
            )
        }

    def _score(request: Request, rows: list[dict[str, float]]) -> list[ScoreResponse]:
        service = get_service(request)
        try:
            results = service.score_many(rows)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        for result in results:
            SCORES.observe(result["fraud_score"])
            FLAGS.labels(str(result["review_flag"]).lower()).inc()
        return [ScoreResponse(**r) for r in results]

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(body: ScoreRequest, request: Request) -> ScoreResponse:
        return _score(request, [body.features])[0]

    @app.post("/v1/score/batch", response_model=BatchScoreResponse)
    def score_batch(body: BatchScoreRequest, request: Request) -> BatchScoreResponse:
        if len(body.instances) > settings.max_batch_size:
            raise HTTPException(
                status_code=413, detail=f"batch larger than {settings.max_batch_size} instances"
            )
        return BatchScoreResponse(predictions=_score(request, body.instances))

    return app


app = create_app()
