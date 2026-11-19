# Architecture and design decisions

```mermaid
flowchart TD
 A[Public CSV] --> B[SHA-256 check + data contract]
 Z[configs/train.yaml] --> B
 B --> C[Stratified train / validation / test split]
 C --> D[Preprocessing + model pipelines]
 D --> E[CV model comparison]
 E --> F[Validation: model + threshold selection]
 F --> G[Held-out test: metrics, capacity, segments]
 G --> Q{Quality gates}
 Q -- fail --> R[Reports only, nothing published]
 Q -- pass --> H[artifacts/runs/run_id + current artifact]
 H --> I[FastAPI service]
 H --> J[Batch scoring job]
 I --> K[Client]
 I --> L[JSON logs + Prometheus metrics]
 L --> M[Operational review: drift, rollback, retrain]
```

- Keep preprocessing in the fitted model pipeline to reduce training-serving skew.
- Remove customer identifiers to reduce memorization risk.
- Pin the dataset version by checksum; fail loudly if it changes.
- Choose models using cross-validation on training data only.
- Select the threshold on validation data; use the test set only for final evaluation.
- Gate publishing on test metrics; keep every published version immutable for rollback.
- Evaluate by segment, not just overall: aggregate metrics can hide groups the model fails.
- Training settings come from YAML; serving settings from `CHURN_*` environment variables.
- Batch and online scoring share one service, so they cannot drift apart.
- Treat threshold costs as assumptions until validated with business stakeholders.
- Load `joblib` artifacts only from trusted sources.
