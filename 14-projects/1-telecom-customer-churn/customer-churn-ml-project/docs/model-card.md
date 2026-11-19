# Model Card — Customer Churn Risk

## Model details
- **Type:** binary classifier inside an sklearn Pipeline. Numeric features: median
  imputation and scaling. Categorical features: most-frequent imputation and one-hot
  encoding (unseen categories ignored).
- **Selected estimator (latest run):** logistic regression, `class_weight="balanced"`.
- **Candidates compared:** logistic regression, decision tree, random forest (see `configs/train.yaml`).
- **Version:** each training run gets a `model_version` (run ID); full metadata in `artifacts/metadata.json`.

## Intended use
Rank customers by churn risk so a retention team can prioritise outreach. The output
supports a human decision; it does not trigger offers automatically. A risk score is
not a measure of who an offer would persuade: estimating the value of outreach needs
an uplift model or a controlled experiment.

## Training data
IBM Telco Customer Churn sample: 7,043 customers, 26.5% churned, 19 features
(account, services, billing). `customerID` is excluded. Pinned by SHA-256 in config.
11 customers with zero tenure have blank `TotalCharges`, which are imputed.

**Split (stratified random, 64/16/20):** train 4,507, validation 1,127, test 1,409;
the churn rate is 26.5% in each.

## Evaluation (run of 2026-10-14; re-run `make train` and update before quoting)

Model selection:

| Model | CV average precision (5-fold) | Validation average precision | Validation ROC-AUC |
|---|---|---|---|
| Logistic regression | 0.657 ± 0.022 | **0.666** | 0.848 |
| Random forest | 0.657 ± 0.026 | 0.650 | 0.845 |
| Decision tree | 0.603 ± 0.030 | 0.615 | 0.825 |

Held-out test set, threshold 0.41 (chosen on validation; missed churner = 5, unnecessary contact = 1):

| Metric | Value |
|---|---|
| Average precision | 0.632 |
| ROC-AUC | 0.842 |
| Recall | 0.861 (322 of 374 churners flagged) |
| Precision | 0.471 (322 of 684 flagged customers churned) |
| Accuracy | 0.706 |
| Top 10% contacted (141 customers) | precision 0.745, recall 0.281 |
| Majority-class baseline | 73.5% accuracy, 0 churners found |

### Segment performance (test set)

| Segment | Customers | Churn rate | Recall | Precision |
|---|---|---|---|---|
| Month-to-month contract | 773 | 42.6% | 0.93 | 0.49 |
| One-year contract | 300 | 12.0% | 0.42 | 0.28 |
| Two-year contract | 336 | 2.7% | **0.00** | 0.00 |
| Fiber optic internet | 613 | 41.1% | 0.93 | 0.51 |
| No internet service | 312 | 8.0% | 0.36 | 0.33 |
| Electronic check payment | 474 | 43.5% | 0.95 | 0.52 |

Full table: `reports/segment_metrics.csv`.

## Caveats and limitations
- **Segment gap:** the model misses every churner on two-year contracts and most on
  one-year contracts. Overall recall hides this. If those customers matter to the
  business, consider segment-specific thresholds or features that capture their churn drivers.
- **Precision vs. capacity:** at the cost-chosen threshold roughly half of the contacted
  customers would not have churned. If outreach capacity is the real constraint, choose
  the operating point from the capacity (top-k) view instead.
- **Random split:** without a scoring timestamp, the test set is not "later" than training.
  Performance on future customers may be lower.
- **Costs are illustrative,** not measured campaign economics.
- **Sensitive attributes:** `gender` and `SeniorCitizen` are model inputs. Real use needs a
  legal and fairness review, and possibly removing them or monitoring outcomes by group.

## Monitoring
Request and error rates, latency, the prediction distribution and the predicted-churn rate
(`/metrics`); input drift against training statistics; precision and recall by segment
once churn outcomes mature (weeks to months). See [operations.md](operations.md).
