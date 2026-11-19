# Operations Runbook (portfolio template)

## Readiness
- Confirm the intended model artifact and metadata are present.
- Confirm model version, feature schema, threshold assumptions, and evaluation report.
- Run unit and API smoke tests.
- Verify latency and resource budgets in the target environment.

## Monitoring
- Service: request count, errors, timeouts, p95/p99 latency, CPU/memory.
- Data: missing fields, invalid numeric values, schema change, distribution drift.
- Model: precision/recall/average precision, calibration, cost and false-positive impact after labels mature.
- Fraud outcomes can be delayed; do not treat immature labels as legitimate transactions.

## Incident response
1. Acknowledge alert and identify affected model/version/time window.
2. Check service health, input schema, feature pipeline, and label maturity.
3. If the model is causing harm or service reliability is below policy, switch to the approved fallback or previous known-good version.
4. Preserve sanitized logs and evidence; avoid retaining sensitive transaction payloads unnecessarily.
5. Document root cause, impact, remediation, and follow-up tests.

## Rollback
- Keep previous artifact and application image immutable.
- Make model version explicit.
- Test rollback before a production release.
- This repository does not implement automated rollback; it documents the operational policy.
