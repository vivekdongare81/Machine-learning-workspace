# Customer Churn Prediction — Risk Scoring for Retention

An end-to-end ML system that scores customers by churn risk so a retention team
can prioritise outreach: config-driven training with release gates, segment-level
evaluation, versioned model artifacts, batch scoring, and an observable online API.
Built on the public IBM Telco churn dataset as a portfolio project.

```text
             configs/train.yaml
                    │
 raw CSV ─► checksum + schema ─► stratified split ─► CV (StratifiedKFold) ─► pick model
 (pinned)    validation           train│valid│test       on training data       on validation
                                                                                  │
      artifacts/runs/<run_id>/ ◄─ publish ◄─ quality gates ◄─ test once ◄─ pick threshold
      artifacts/ (current)           (fail = not published)  + segments    (exact, cost-based)
                │
      ┌─────────┴──────────┐
 FastAPI /v1/predict   batch scoring CLI     ──►  logs (JSON, request IDs) + Prometheus /metrics
```

## Project layout

```text
configs/train.yaml        every tunable: data version, split, CV, models, costs, segments, gates
src/churn_ml/
  config.py               typed config (pydantic) — invalid config fails before training starts
  data.py                 download, SHA-256 check, data contract (columns, IDs, target, ranges)
  pipeline.py             ColumnTransformer (impute/scale numeric, impute/one-hot categorical)
  metrics.py              exact cost-optimal threshold, capacity (top-k) and segment metrics
  train.py                training pipeline + CLI (python -m churn_ml.train)
  tracking.py             optional MLflow tracking behind a no-op interface
  service.py              scoring + feature-contract checks, shared by API and batch
  predict.py              batch scoring CLI (python -m churn_ml.predict)
  api.py                  FastAPI app: health, /v1/predict, /v1/predict/batch, /v1/model, /metrics
  settings.py             serving settings from environment variables
  logging_utils.py        text or JSON structured logs
tests/                    unit, contract, end-to-end training, API and batch tests
docs/                     architecture, model card, operations runbook, project checklist
requirements.lock         pinned runtime dependencies (used by Docker)
requirements-dev.lock     pinned runtime + dev dependencies (used by CI and `make install`)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install                 # pinned deps + editable package
make check                   # lint + type-check + tests (what CI runs)
make train                   # downloads data on first run, trains, gates, publishes
make serve                   # http://127.0.0.1:8000/docs
```

`make help` lists every task. Without `make`, run the commands shown in the [Makefile](Makefile).

## Configuration

Everything that changes between experiments lives in [configs/train.yaml](configs/train.yaml):
dataset URL and its pinned SHA-256, split fractions, CV folds, candidate models and their
hyperparameters, threshold cost assumptions, segments to evaluate, outreach capacity,
quality gates, and tracking. Unknown keys and impossible values are rejected up front.

```bash
make train CONFIG=configs/my-experiment.yaml
python -m churn_ml.train --config configs/train.yaml --data-path /path/to/churn.csv --tracking
```

## Training pipeline

1. **Data version check.** The raw file's SHA-256 must match `data.sha256`.
2. **Data contract.** Exact columns, unique `customerID`, `Yes`/`No` target, numeric and
   non-negative `tenure`/`MonthlyCharges`. Blank `TotalCharges` (brand-new customers)
   become missing values that the pipeline imputes.
3. **Leakage control.** `customerID` and the target are never features. All
   preprocessing is fitted inside the pipeline, on training folds only.
4. **Stratified 64/16/20 split.** The dataset has no scoring timestamp; with one, a
   time-based split (train on past customers, test on later ones) would be preferred.
5. **Model comparison.** 5-fold stratified CV on training data, then the validation set
   picks the model with the best average precision.
6. **Threshold.** The exact cost-minimising threshold on validation data (missed churner
   = 5, unnecessary contact = 1, both illustrative).
7. **Test once, including segments.** Test metrics, the retention team's capacity view
   (precision and recall when contacting the top 10% riskiest), and precision/recall per
   contract type, internet service and payment method.
8. **Quality gates.** If test average precision or recall is below the minimum, the run
   exits with code 2 and **nothing is published**. Reports are still written.
9. **Publish.** The model and metadata go to `artifacts/runs/<run_id>/` (immutable)
   and are copied to `artifacts/` (served). Metadata records the data checksum, git
   commit, feature contract, metrics and library versions.

Exit codes: `0` published, `2` quality gates failed, `3` data validation failed.

Reports in `reports/`: `cv_results.csv`, `validation_results.csv`, `threshold_sweep.csv`,
`segment_metrics.csv`, `test_predictions.csv` (with error type, for error analysis) and
`test_metrics.json`.

### Model versioning and rollback

Each run has an ID like `20261014T104632Z-2785a8`. Its model and metadata are kept in
`artifacts/runs/<run_id>/` and never modified. The API reports the version it serves
(`/health/ready`, `/v1/model`, every prediction). To roll back, serve an earlier run:

```bash
CHURN_ARTIFACT_DIR=artifacts/runs/<previous_run_id> make serve
```

## Serving API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | Process is up (orchestrator restarts the container if this fails) |
| GET | `/health/ready` | Model loaded; `503` until then (load balancer routes traffic only when ready) |
| POST | `/v1/predict` | Predict for one customer |
| POST | `/v1/predict/batch` | Predict for up to `CHURN_MAX_BATCH_SIZE` customers (`413` above it) |
| GET | `/v1/model` | Served model version, threshold, feature contract and test metrics |
| GET | `/metrics` | Prometheus metrics |

```bash
curl -s localhost:8000/v1/predict -H 'Content-Type: application/json' -d '{"features": {
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1,
  "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
  "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85, "TotalCharges": 29.85}}'
# {"churn_probability": 0.80, "predicted_churn": true, "threshold": 0.41, "model_version": "2026..."}
```

Requests must contain exactly the model's features. Numeric features must be numbers
(or `null`), categorical features text (or `null`). Unseen categories are accepted and
ignored by the encoder. Anything else returns `422` with the reason. Settings are
`CHURN_*` environment variables (see [.env.example](.env.example)).

### Batch scoring

```bash
make score-batch INPUT=data/customers.csv OUTPUT=reports/scores.csv
```

The file is read in chunks. `customerID` and any other non-feature columns pass
through. Batch and online scoring share one service, so they return identical
probabilities for the same customer.

## Observability

- **Logs:** `CHURN_LOG_FORMAT=json` emits one JSON object per line, with
  `request_id`, route, status and latency for every request. Clients can send
  `X-Request-ID`; it is echoed in the response header.
- **Metrics:** `churn_api_requests_total` and `churn_api_request_seconds` by route and
  status, plus `churn_model_probability` (distribution) and `churn_model_predictions_total`.
  A shift in the predicted-churn rate is an early drift signal, long before outcomes are known.

## Quality and CI

- `ruff` (lint and format) and `mypy` (types) run in CI and as pre-commit hooks (`make hooks`).
- Tests cover config validation, the data contract and checksum, threshold maths (checked
  against brute force), segment metrics, unseen categories and missing values, end-to-end
  training (publish and gate failure), the train/serve artifact contract, API behaviour,
  and batch/online parity.
- CI runs on Python 3.10 and 3.12 from the lock file, with an 80% coverage floor.
- Dependencies are ranged in `pyproject.toml` and pinned in `requirements*.lock`.
  Update them deliberately with `make lock`.

## Docker

```bash
make train && make docker-build && make docker-run
```

The image installs only pinned runtime dependencies, runs as a non-root user, logs
JSON, has a `HEALTHCHECK` on `/health/ready`, and contains only the current model.

## Important design notes
- The threshold objective uses illustrative costs. Real costs and campaign capacity
  must be agreed with stakeholders.
- A churn model estimates risk, not who a retention offer would persuade. Measuring
  business value needs an uplift model or a controlled experiment.
- `joblib` artifacts are pickle-based: load only trusted model files.
- The public dataset may not represent current customers. This project is not approved
  for real customer decisions. See the [model card](docs/model-card.md).
- Report the metrics you actually obtain; do not invent performance results.
- Describe this as a portfolio project, not work deployed at Zoho unless that is true.

## Interview readiness
Be able to explain the business goal, data limitations, leakage prevention, model choice, threshold trade-offs, measured test metrics, false-positive/false-negative patterns, API failure handling, monitoring, rollback and retraining policy.
