# Fraud Detection — Time-Aware Risk Scoring

An end-to-end ML system for scoring card transactions by fraud risk: config-driven
training with release gates, versioned model artifacts, batch scoring, and an
observable online API. Built on a public benchmark dataset as a portfolio project.

```text
             configs/train.yaml
                    │
 raw CSV ─► checksum + schema ─► time-ordered split ─► CV (TimeSeriesSplit) ─► pick model
 (pinned)    validation            train│valid│test        on train period        on validation
                                                                                     │
         artifacts/runs/<run_id>/ ◄─ publish ◄─ quality gates ◄─ test once ◄─ pick threshold
         artifacts/ (current)              (fail = not published)              (exact, cost-based)
                │
      ┌─────────┴──────────┐
 FastAPI /v1/score     batch scoring CLI      ──►  logs (JSON, request IDs) + Prometheus /metrics
```

## Project layout

```text
configs/train.yaml        every tunable: data version, split, CV, models, costs, gates, tracking
src/fraud_ml/
  config.py               typed config (pydantic) — invalid config fails before training starts
  data.py                 download, SHA-256 check, data-contract validation
  pipeline.py             preprocessing + estimator built from config
  metrics.py              exact cost-optimal threshold, threshold / ranking / top-k metrics
  train.py                training pipeline + CLI (python -m fraud_ml.train)
  tracking.py             optional MLflow tracking behind a no-op interface
  service.py              scoring + feature-contract checks, shared by API and batch
  predict.py              batch scoring CLI (python -m fraud_ml.predict)
  api.py                  FastAPI app: health, /v1/score, /v1/score/batch, /v1/model, /metrics
  settings.py             serving settings from environment variables
  logging_utils.py        text or JSON structured logs
tests/                    unit, contract, end-to-end training, API and batch tests
docs/                     architecture, model card, operations runbook, project plan
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
dataset URL and its pinned SHA-256, split fractions, CV strategy, candidate models and
their hyperparameters, threshold cost assumptions, quality gates, and tracking.
Unknown keys and impossible values are rejected up front.

```bash
make train CONFIG=configs/my-experiment.yaml
python -m fraud_ml.train --config configs/train.yaml --data-path /path/to/creditcard.csv --tracking
```

## Training pipeline

1. **Data version check.** The raw file's SHA-256 must match `data.sha256`, so a
   changed upstream file stops training instead of silently changing the model.
2. **Data contract.** Exact columns, numeric types, binary target, non-negative
   `Time`/`Amount`. Unexpected columns are dropped and logged.
3. **Time-ordered split, 60/20/20.** Train on the past, validate on the next period,
   test on the latest. A random split would leak future patterns into training.
4. **Model comparison.** Every candidate is cross-validated with `TimeSeriesSplit`
   (each fold validates on data later than it trains on), then fitted and scored on
   the validation period. The best validation average precision wins.
5. **Threshold.** The exact cost-minimising threshold on validation data, evaluated
   over every distinct score in O(n log n) rather than a coarse grid.
6. **Test once.** The latest period is scored only after the model and threshold are fixed.
7. **Quality gates.** If test average precision or recall is below the configured
   minimum, the run exits with code 2 and **nothing is published**. Reports are
   still written for investigation.
8. **Publish.** The model and metadata go to `artifacts/runs/<run_id>/` (immutable)
   and are copied to `artifacts/` (the version serving loads). Metadata records the
   data checksum, git commit, config, metrics, split sizes and library versions.

Exit codes: `0` published, `2` quality gates failed, `3` data validation failed.

### Model versioning and rollback

Each run has an ID like `20261014T103651Z-19ffa9`. Its model and metadata are kept in
`artifacts/runs/<run_id>/` and never modified. The API reports the version it serves
(`/health/ready`, `/v1/model`, and every score response). To roll back, serve an earlier run:

```bash
FRAUD_ARTIFACT_DIR=artifacts/runs/<previous_run_id> make serve
```

## Serving API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | Process is up (orchestrator restarts the container if this fails) |
| GET | `/health/ready` | Model loaded; `503` until then (load balancer routes traffic only when ready) |
| POST | `/v1/score` | Score one transaction |
| POST | `/v1/score/batch` | Score up to `FRAUD_MAX_BATCH_SIZE` transactions (`413` above it) |
| GET | `/v1/model` | Served model version, threshold, features and test metrics |
| GET | `/metrics` | Prometheus metrics |

```bash
curl -s localhost:8000/v1/score -H 'Content-Type: application/json' \
  -d '{"features": {"Time": 0, "V1": -1.36, "...": 0, "Amount": 149.62}}'
# {"fraud_score": 0.07, "review_flag": false, "threshold": 0.9996, "model_version": "2026..."}
```

Requests must contain exactly the model's features as finite numbers. Missing,
extra or non-numeric features return `422` with the reason. Settings are environment
variables with the `FRAUD_` prefix (see [.env.example](.env.example)).

### Batch scoring

```bash
make score-batch INPUT=data/new_transactions.csv OUTPUT=reports/scores.csv
```

The file is read in chunks, so memory stays flat for large inputs. Non-feature
columns (IDs, labels) pass through. Batch and online scoring share one service,
so they return identical scores for the same row.

## Observability

- **Logs:** `FRAUD_LOG_FORMAT=json` emits one JSON object per line, with
  `request_id`, route, status and latency for every request. Clients can send
  `X-Request-ID`; it is echoed in the response header so a request can be traced.
- **Metrics:** `fraud_api_requests_total` and `fraud_api_request_seconds` by route and
  status, plus `fraud_model_score` (score distribution) and `fraud_model_review_flags_total`.
  A shift in the score distribution or flag rate is an early drift signal, before labels arrive.

## Quality and CI

- `ruff` (lint and format) and `mypy` (types) run in CI and as pre-commit hooks (`make hooks`).
- Tests cover config validation, the data contract and checksum, threshold maths
  (checked against brute force), end-to-end training on synthetic data (publish
  and gate failure), the train/serve artifact contract, API behaviour, and batch/online parity.
- CI runs on Python 3.10 and 3.12 from the lock file, with an 80% coverage floor.
- Dependencies are ranged in `pyproject.toml` and pinned in `requirements*.lock`.
  Update them deliberately with `make lock`.

## Docker

```bash
make train && make docker-build && make docker-run
```

The image installs only pinned runtime dependencies, runs as a non-root user,
logs JSON, has a `HEALTHCHECK` on `/health/ready`, and contains only the current
model (data, tests and run history are excluded by `.dockerignore`).

## Dataset and limitations

Credit-card fraud benchmark: 284,807 transactions, 492 frauds (0.17%), features
`Time`, `Amount` and anonymised `V1`–`V28`, target `Class`. The CSV is downloaded
from a public mirror; check the source terms before reuse. `Time` is elapsed
seconds rather than a calendar timestamp, the features are anonymised, and the
cost assumptions are illustrative. See the [model card](docs/model-card.md).

## Responsible presentation

Describe this honestly as a personal portfolio project. Do not imply it was deployed at Zoho or another employer. Do not invent performance metrics or business savings.
