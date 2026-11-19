# Machine Learning Workspace

Hands-on notes and code for the classical ML algorithms, the fundamentals around them, and two end-to-end projects.
Most notebooks follow the same pattern: **why → intuition → formula (where needed) → code → experiments → interview notes**.

Interview questions and where each one is answered: [INTERVIEW-MAP.md](INTERVIEW-MAP.md)

---

## Topics

| Folder | What's inside |
|---|---|
| [`00-numpy-and-pandas/`](00-numpy-and-pandas) | NumPy for ML (shapes and axes, broadcasting, vectorization, matrix shapes, numerical stability) and pandas for ML (dtypes, group-by, joins, reshaping, time series without leakage, vectorized code, memory, building a feature table) |
| [`01-linear-regression/`](01-linear-regression) | NumPy and vectorization, linear regression from scratch (single and multiple features), gradient descent, feature scaling, learning rate, polynomial features, scikit-learn |
| [`02-logistic-regression/`](02-logistic-regression) | Why not MSE (MLE, cross-entropy), sigmoid and decision boundary, logistic loss, gradient descent, overfitting and regularization, scikit-learn |
| [`03-decision-trees-and-ensembles/`](03-decision-trees-and-ensembles) | Decision tree, random forest, AdaBoost, gradient boosting (from scratch), XGBoost, LightGBM, CatBoost |
| [`04-SVM/`](04-SVM) | Margin, support vectors, hard vs soft margin, C, hinge loss, SVM from scratch, kernel trick |
| [`05-KNN/`](05-KNN) | Distance metrics, choosing K, uniform vs distance weights, feature scaling, KNN classification and regression |
| [`06-probabilistic-models/`](06-probabilistic-models) | Naive Bayes (Gaussian, Multinomial, Bernoulli, Laplace smoothing), LDA, QDA, Gaussian Mixture Models (EM algorithm) |
| [`07-clustering-models/`](07-clustering-models) | K-Means (inertia, elbow, silhouette), hierarchical clustering (linkage, dendrograms), DBSCAN |
| [`08-dimensionality-reduction/`](08-dimensionality-reduction) | PCA from scratch vs scikit-learn, LDA for dimensionality reduction, t-SNE, UMAP |
| [`09-anomaly-detection/`](09-anomaly-detection) | Isolation Forest, Local Outlier Factor, One-Class SVM |
| [`10-recommendation-system/`](10-recommendation-system) | User/item collaborative filtering, content-based filtering, matrix factorization, implicit feedback, ranking metrics, two-tower models, retrieval + ranking |
| [`11-timeseries/`](11-timeseries) | Stationarity, ACF/PACF, AR/MA/ARIMA/SARIMA/SARIMAX, VAR, ML-based forecasting, backtesting, RNN/LSTM/GRU and transformers (overview) |
| [`12-all-algo-sklearn-cheatsheet/`](12-all-algo-sklearn-cheatsheet) | One-page scikit-learn usage for every algorithm above |
| [`13-ml-fundamentals/`](13-ml-fundamentals) | See below |
| [`14-projects/`](14-projects) | Two end-to-end projects (see below) |

### 13 — ML fundamentals

| Notebook | Topics |
|---|---|
| `1-ml-fundamentals` | Splits, bias-variance, regularization (L1/L2), cross-validation, data leakage, class imbalance, baselines, learning curves |
| `2-ml-metrics` | Regression, classification, ROC and precision-recall, multiclass, ranking and clustering metrics, choosing a metric |
| `3-ml-feature-engineering` | Missing values, encoding, scaling, outliers, date/time and cyclical features, feature selection, leakage-safe pipelines |
| `4-ml-pipelines-and-hyperparameter-tuning` | Pipelines, ColumnTransformer, grid and random search, nested CV, early stopping, saving the full pipeline |
| `5-ml-explainability` | Feature and permutation importance, SHAP (global and local), PDP and ICE, explainability vs causality |
| `6-ml-deployments` | Serialization (joblib / ONNX), FastAPI serving, batch vs online, monitoring, versioning, Docker, canary and shadow deployments |
| `7-statistics-and-ab-testing` | Confidence intervals, bootstrap, hypothesis tests, A/B test design and sample size, peeking and multiple testing, comparing two models on one test set |
| `8-text-classification-basics` | Bag of words, TF-IDF, text classifiers, error analysis, TF-IDF search, BM25 from scratch, when to use a classical model vs an LLM |

---

## Projects

Both projects use public datasets. Each has an exploration notebook (`notebook/`) and a separate production-style repository with
config-driven training, quality gates, tests, CI, Docker and a FastAPI service.

| Project | Dataset | Highlights |
|---|---|---|
| [Customer churn](14-projects/1-telecom-customer-churn/customer-churn-ml-project) | IBM Telco Customer Churn | Stratified CV, cost-based threshold, segment-level evaluation, batch scoring + online API |
| [Fraud detection](14-projects/2-fraud-detection/fraud-detection-ml-project) | Credit Card Fraud (0.13% fraud) | Time-ordered split, precision-recall focus, threshold chosen from review capacity, monitoring plan |

The fraud dataset (~100 MB) is not stored in git. The notebook downloads it on first run.

---

## Setup

Python **3.10**. Library versions are in [`requirements.txt`](requirements.txt).

```bash
# 1. Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv310
source .venv310/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register the kernel with Jupyter
python -m ipykernel install --user --name=venv310 --display-name "Python 3.10 (ml)"
```
