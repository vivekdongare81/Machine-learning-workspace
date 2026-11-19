# Architecture

## Training path
`configs/train.yaml` → download / SHA-256 check → data-contract validation → chronological
60/20/20 split → `TimeSeriesSplit` CV on the training period → model selection on validation →
exact cost-based threshold on validation → single evaluation on the latest (test) period →
quality gates → publish to `artifacts/runs/<run_id>/` and `artifacts/` (current) → reports,
optional MLflow run.

## Serving path
Caller → FastAPI (`/v1/score`, `/v1/score/batch`) → request schema validation (pydantic) →
feature-contract validation (`FraudScoringService`) → fitted preprocessing/model pipeline →
risk score + review flag + model version → downstream review queue.
Batch scoring (`fraud_ml.predict`) uses the same service, so scores match the API.

## Configuration and deployment
Training settings come from YAML; serving settings from `FRAUD_*` environment variables.
The Docker image bakes in the current model, runs as non-root, and exposes `/health/live`,
`/health/ready` and `/metrics`. Scale by running more replicas; each holds its own model copy.

## Monitoring path
Structured logs (request ID, route, status, latency) + Prometheus metrics (request rate,
latency, score distribution, flag rate) + input drift checks + delayed outcome joins →
evaluation and release gates → investigate / retrain / roll back to an earlier run ID.

## Important limitations
The benchmark is not a live payment dataset. Its `Time` is elapsed seconds, not calendar time;
features are anonymised; labels and feature availability do not fully describe a production
authorisation process. This project is an educational reference implementation, not a
deployable payment decision product.
