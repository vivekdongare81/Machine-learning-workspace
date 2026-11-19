# Project Plan

## Implemented
- Config-driven training (`configs/train.yaml`) validated by pydantic
- Dataset version pinning (SHA-256), data-contract checks, atomic download
- Time-ordered 60/20/20 split and time-aware cross-validation (`TimeSeriesSplit`)
- Majority baseline and supervised candidate comparison
- Exact cost-optimal threshold selection on validation data
- Held-out latest-period evaluation, including review-capacity (top-k) metrics
- Quality gates that block publishing a weak model
- Versioned, immutable artifacts (`artifacts/runs/<run_id>/`) with lineage metadata
  (data checksum, git commit, config, library versions) and rollback by version
- Optional MLflow experiment tracking
- Batch scoring CLI and online FastAPI service sharing one scoring path
- Health/readiness probes, versioned routes, batch endpoint with size limit
- Structured JSON logs with request IDs; Prometheus request, latency and score metrics
- Pinned dependency lock files; ruff, mypy and pre-commit; CI on Python 3.10 and 3.12
- Tests: config, data, metrics, end-to-end training, train/serve contract, API, batch
- Non-root Docker image with healthcheck; docker-compose for local runs; model card

## Next enhancements
1. Probability calibration (`CalibratedClassifierCV`) and a calibration quality gate.
2. Gradient-boosted candidate (HistGradientBoosting / LightGBM) with tuned hyperparameters.
3. Drift monitoring job: compare live feature and score distributions with training (PSI).
4. Delayed-label join: evaluate live precision/recall once chargebacks mature.
5. Load test (e.g. Locust) with p95/p99 latency budgets under concurrency.
6. Model registry (MLflow) with staged promotion and a shadow/canary release step.
7. CI job that builds and smoke-tests the Docker image against a small trained model.
