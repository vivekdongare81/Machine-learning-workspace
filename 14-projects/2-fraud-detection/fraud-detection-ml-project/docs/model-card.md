# Model Card — Fraud Risk Scorer

## Model details
- **Type:** binary classifier inside an sklearn Pipeline (median imputation → standard scaling → estimator).
- **Selected estimator (latest run):** logistic regression, `class_weight="balanced"`.
- **Candidates compared:** logistic regression, random forest, extra trees (see `configs/train.yaml`).
- **Version:** each training run gets a `model_version` (run ID); full metadata in `artifacts/model_metadata.json`.

## Intended use
Rank transactions by fraud risk so a human review team can prioritise. The output is a
**review flag, not an automatic decline**. Not intended for use on real payment traffic
without validation on representative data, fairness review and a defined decision policy.

## Training data
Public credit-card fraud benchmark: 284,807 transactions over about two days, 492 frauds
(0.17%). Pinned by SHA-256 in config. Features are `Time` (elapsed seconds), `Amount` and
anonymised PCA components `V1`–`V28`.

**Split (time-ordered):**

| Period | Rows | Frauds | Fraud rate |
|---|---|---|---|
| Train (earliest 60%) | 170,884 | 360 | 0.21% |
| Validation (next 20%) | 56,961 | 57 | 0.10% |
| Test (latest 20%) | 56,962 | 75 | 0.13% |

## Evaluation (run of 2026-10-14; re-run `make train` and update before quoting)

Model selection on the validation period:

| Model | CV average precision (TimeSeriesSplit) | Validation average precision | Validation Brier |
|---|---|---|---|
| Logistic regression | 0.736 ± 0.100 | **0.772** | 0.0282 |
| Extra trees | 0.764 ± 0.090 | 0.765 | 0.0004 |
| Random forest | 0.788 ± 0.070 | 0.760 | 0.0004 |

Held-out test period, threshold 0.9996 (chosen on validation with false negative = 10, false positive = 1):

| Metric | Value |
|---|---|
| Average precision | 0.744 |
| ROC-AUC | 0.982 |
| Precision | 0.724 (55 of 76 alerts were fraud) |
| Recall | 0.733 (55 of 75 frauds caught) |
| False alerts | 21 out of 56,887 legitimate transactions |
| Top 1% reviewed | recall 0.867, precision 0.114 |
| Majority-class baseline | 99.9% accuracy, 0 frauds caught |

## Caveats and limitations
- **Calibration:** balanced class weights inflate logistic-regression probabilities
  (Brier 0.028 vs. 0.0004 for the tree models), which is why the threshold is near 1.
  Treat scores as a ranking; calibrate (e.g. `CalibratedClassifierCV`) before using them as probabilities.
- **Model selection disagreement:** cross-validation ranks random forest first, but the
  later validation period favours logistic regression. The gap is within fold variability;
  more time periods would make the choice more robust.
- **Short time span:** two days of data cannot show weekly or seasonal patterns or long-term drift.
- **Anonymised features** prevent domain-level explanation of individual scores.
- **Costs are illustrative,** not measured business costs.
- **Label delay:** in production, fraud labels (chargebacks) arrive weeks later, so live
  quality monitoring must wait for labels to mature.

## Monitoring
Request and error rates, latency, the score distribution and the flag rate (`/metrics`);
input drift against training statistics; precision and recall once labels mature.
See [operations.md](operations.md).
