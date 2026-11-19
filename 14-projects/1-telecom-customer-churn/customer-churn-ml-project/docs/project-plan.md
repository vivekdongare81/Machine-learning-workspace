# Project checklist

## Problem and data
- [ ] Define user, business action and success metrics
- [ ] Inspect class balance, missing values, duplicates and distributions
- [ ] Document dataset source and limitations
- [ ] Check target and identifier leakage

## Modeling
- [ ] Explain baseline and split design
- [ ] Explain preprocessing transformations
- [ ] Compare CV scores and variability
- [ ] Review threshold, PR-AUC, ROC-AUC and confusion matrix
- [ ] Analyze false positives and false negatives by segment
- [ ] Record measured results, not assumed scores

## Engineering
- [ ] Run tests from a clean environment
- [ ] Train and inspect artifacts/reports
- [ ] Test unknown categories and invalid API payloads
- [ ] Build/run Docker and check health/metrics endpoints
- [ ] Load test and record p50/p95/p99 and throughput

## Operations and interviews
- [ ] Explain drift versus model-quality monitoring
- [ ] Define delayed-label collection and retraining triggers
- [ ] Write incident response and rollback plan
- [ ] Document privacy, access-control and security requirements
- [ ] Prepare a two-minute walkthrough and defend key trade-offs
