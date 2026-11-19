# Interview Map

ML interview questions and where each one is answered in this repo.
`sec 3` = section 3 in the notebook's contents list, `Q5` = question 5 in its interview questions.
Notebooks without numbered sections are linked by section name.

Revision order before an interview: [13/1][f1], [13/2][f2], [KNN][knn], then the algorithms in the job description.

## Core topics

### ML fundamentals

| Question | Where |
|---|---|
| Supervised vs unsupervised vs self-supervised vs reinforcement learning? | [13/1][f1] sec 1, Q14 |
| Classification vs regression? Parameters vs hyperparameters? | [13/1][f1] sec 1 |
| Parametric vs non-parametric models? | [13/1][f1] sec 1, Q12 |
| Discriminative vs generative models? | [13/1][f1] sec 1, Q15 |
| Why train, validation and test sets? When is a random split wrong? | [13/1][f1] sec 2, Q7 |
| Explain overfitting, underfitting and the bias-variance trade-off. | [13/1][f1] sec 3, Q1, Q2 |
| How do you reduce overfitting? | [13/1][f1] sec 3, Q3 |
| How do bagging and boosting relate to bias and variance? | [13/1][f1] sec 3, Q13 |
| Will more data fix my model? What does a learning curve tell you? | [13/1][f1] sec 8, Q4 |
| L1 vs L2? Why does L1 give exact zeros? What is Elastic Net? | [13/1][f1] sec 4, Q5 |
| What do `alpha` and `C` mean in scikit-learn? | [13/1][f1] sec 4, Q6 |
| Why cross-validation? What are LOOCV, repeated and nested CV? | [13/1][f1] sec 5, Q10; [13/4][f4] |
| What is data leakage? Give examples. | [13/1][f1] sec 6, Q8; [13/3][f3] "Data leakage" |
| Your model has 99% accuracy. What do you check? | [13/1][f1] sec 7, Q9 |
| How do you handle class imbalance? Why build a baseline first? | [13/1][f1] sec 7 |
| Walk me through an ML project end to end. | [13/1][f1] sec 9, Q11; [project notebook][p0] |
| Walk me through your project. | [churn][p1], [fraud][p2] |
| How do you do error analysis? | [13/1][f1] sec 9; [13/8][f8] "Interpreting the model" |
| Good score offline, bad in production. Why? | [13/1][f1] sec 10; [13/6][f6] monitoring |

### Gradient descent and linear models

| Question | Where |
|---|---|
| How does gradient descent work? | [01/2][l2] "Gradient descent loop"; [01/3][l3] |
| How do you choose the learning rate? What happens if it is too large? | [01/4][l4] "Choosing the learning rate α"; [01/3][l3] last section |
| Why does feature scaling speed up gradient descent? | [01/4][l4] "Feature scaling", "Cost contours" |
| How can a linear model fit a curve? | [01/5][l5] polynomial features |
| Why not use linear regression for classification? | [02/1][g1] |
| What is the sigmoid? What is the decision boundary? | [02/2][g2] |
| Why log loss (cross-entropy) and not MSE for logistic regression? | [02/1][g1], [02/3][g3] |
| What is the gradient of logistic regression? | [02/4][g4] |
| How does regularization change the cost and the gradient? | [02/6][g6] |

### Metrics

| Question | Where |
|---|---|
| How do you choose an evaluation metric? | [13/2][f2] sec 1, Q1 |
| MAE vs MSE vs RMSE? | [13/2][f2] sec 2, Q2 |
| What does R² mean? Can it be negative? Why adjusted R²? | [13/2][f2] sec 2, Q3 |
| What is wrong with MAPE? What are SMAPE and WAPE? | [13/2][f2] sec 2, Q4 |
| What is a confusion matrix? Type I vs type II errors? | [13/2][f2] sec 3, Q5 |
| Precision vs recall, with an example of each? | [13/2][f2] sec 3, Q5 |
| Why is F1 a harmonic mean? What is Fβ? | [13/2][f2] sec 3, Q6 |
| Sensitivity vs specificity? | [13/2][f2] sec 3, Q7 |
| Why is accuracy misleading on imbalanced data? Balanced accuracy, MCC? | [13/2][f2] sec 5, Q8 |
| How do you choose the classification threshold? | [13/2][f2] sec 4, Q11 |
| What does ROC-AUC measure? What does an AUC below 0.5 mean? | [13/2][f2] sec 5, Q9 |
| ROC-AUC or PR-AUC for fraud with 1% positives? | [13/2][f2] sec 5, Q10 |
| What are log loss, Brier score and calibration? | [13/2][f2] sec 6, Q12 |
| Macro vs micro vs weighted F1? | [13/2][f2] sec 7, Q13 |
| Precision@K, Recall@K, MRR, MAP, NDCG: which one when? | [13/2][f2] sec 8, Q14 |
| How do you evaluate the retrieval step of a RAG system? | [13/2][f2] sec 8, Q15 |
| How do you evaluate a clustering? | [13/2][f2] sec 9 |

### Feature engineering, pipelines and tuning

| Question | Where |
|---|---|
| How do you handle missing values? When do you add a missing indicator? | [13/3][f3] "Missing values", "Missing indicators" |
| One-hot vs ordinal vs target encoding? | [13/3][f3] "One-hot / ordinal encoding"; [CatBoost][b7] sec 7 |
| Which models need scaling? Standard vs MinMax vs Robust scaler? | [13/3][f3] "Scaling"; [KNN][knn] sec 4; [SVM][svm] sec 6 |
| How do you treat outliers and skewed features? | [13/3][f3] "Outliers", "Log transforms" |
| How do you encode dates and cyclical features (hour, month)? | [13/3][f3] "Date/time features", "Cyclical features" |
| How do you select features? What is multicollinearity? | [13/3][f3] "Feature selection", "Correlation / multicollinearity" |
| Why put preprocessing inside a Pipeline? What is ColumnTransformer? | [13/3][f3] "ColumnTransformer + Pipeline"; [13/4][f4] |
| GridSearchCV vs RandomizedSearchCV? | [13/4][f4] |
| Why is the best CV score after tuning optimistic? What is nested CV? | [13/4][f4]; [13/1][f1] sec 5 |
| What is early stopping? | [13/4][f4] |
| How do you save, load and reproduce a model? | [13/4][f4] |

### Explainability

| Question | Where |
|---|---|
| Global vs local explanations? | [13/5][f5] "Global vs local explanations" |
| Impurity-based vs permutation importance? What are their pitfalls? | [13/5][f5] "Tree feature importance", "Permutation importance" |
| What is SHAP, and how do you read a SHAP plot? | [13/5][f5] "SHAP", "SHAP global/local explanations" |
| PDP vs ICE? | [13/5][f5] "PDP", "ICE" |
| How do correlated features break explanations? Is importance causality? | [13/5][f5] "Correlated features", "Explainability vs causality" |

### Deployment and MLOps

| Question | Where |
|---|---|
| Training vs inference? How do you keep preprocessing the same in both? | [13/6][f6] |
| Batch vs online inference? | [13/6][f6] |
| How do you serve a model with FastAPI and validate the input? | [13/6][f6]; [app.py](13-ml-fundamentals/app.py) |
| Health and readiness checks? Latency vs throughput? | [13/6][f6] |
| How do you monitor a model? What is data drift? | [13/6][f6] |
| How do you version models? Canary, shadow deployment, rollback? | [13/6][f6] |
| Docker, ONNX, CPU vs GPU serving? | [13/6][f6] |
| How do you track experiments (MLflow)? | [churn project][p1], [fraud project][p2] |

### Statistics and A/B testing

| Question | Where |
|---|---|
| Central limit theorem, standard error, confidence interval? | [13/7][f7] "Population, sample, CLT", "Standard error" |
| How do you get a confidence interval for any metric? | [13/7][f7] "Bootstrap confidence intervals" |
| What is a p-value? Type I and II errors? Power? | [13/7][f7] "Hypothesis testing" |
| Which statistical test do you use when? | [13/7][f7] "Which test to use when" |
| How do you design an A/B test (metric, randomization unit, sample size)? | [13/7][f7] "A/B test design" |
| How do you analyse one (SRM check, z-test, lift with a CI)? | [13/7][f7] "A/B test analysis" |
| Peeking, multiple testing, novelty effect, Simpson's paradox? | [13/7][f7] "Pitfalls" |
| Is model A really better than model B on the same test set? | [13/7][f7] "Comparing two models" (McNemar, paired bootstrap) |
| Offline vs online evaluation, and how it carries over to LLM evals? | [13/7][f7] "Offline vs online evaluation" |

### Text and search basics

| Question | Where |
|---|---|
| Tokenization, bag of words, n-grams? | [13/8][f8] |
| What is TF-IDF? | [13/8][f8] "TF-IDF" |
| Naive Bayes vs logistic regression vs linear SVM for text? | [13/8][f8] "Comparing Naive Bayes, Logistic Regression and Linear SVM" |
| What is BM25? What is hybrid search? | [13/8][f8] "BM25 from scratch" |
| Why do embeddings beat sparse methods, and when don't they? | [13/8][f8] "Limits of sparse methods" |
| Classical model or LLM: when do you use which? | [13/8][f8] "Classical model vs LLM" |
| Character n-grams, the hashing trick? | [13/8][f8] "Practical tricks" |

### Embeddings and vector search

| Question | Where |
|---|---|
| Euclidean vs cosine vs Manhattan distance? | [KNN][knn] sec 3, Q5 |
| What is the curse of dimensionality? | [KNN][knn] sec 8, Q6 |
| Cosine similarity vs dot product on normalized embeddings? | [KNN][knn] sec 9 |
| How do you search millions of vectors fast (HNSW, IVF, PQ, FAISS)? | [KNN][knn] sec 9, Q8; [recsys][rec] sec 10 |
| How do you look at embeddings in 2-D? | [t-SNE][tsne] Q13; [UMAP][umap] |

## Algorithms

### Trees and ensembles

| Question | Where |
|---|---|
| How does a decision tree choose a split? What is Gini impurity? | [03/1][b1] sec 1-3 |
| How does a random forest work, and why does it beat one tree? | [03/2][b2] sec 3, 20; [13/1][f1] sec 3 |
| Random forest feature importance, number of trees, depth? | [03/2][b2] sec 8-12, 19 |
| How does AdaBoost work (sample weights, learner weights)? | [03/3][b3] sec 9, 10, 16 |
| How does gradient boosting work (fitting the gradients, learning rate)? | [03/4][b4] sec 3, 16, 17 |
| What does XGBoost add (second-order gain, leaf weight −G/(H+λ), regularization)? | [03/5][b5] sec 15-18 |
| What makes LightGBM fast (histograms, leaf-wise growth, GOSS)? | [03/6][b6] sec 5-7, 14 |
| How does CatBoost handle categories (ordered target statistics, ordered boosting)? | [CatBoost][b7] sec 6-9 |
| XGBoost vs LightGBM vs CatBoost vs random forest? | [CatBoost][b7] sec 16; [03/6][b6] sec 12, 13 |

### SVM

| Question | Where |
|---|---|
| Explain SVM. Why is the margin 2/‖w‖? | [SVM][svm] sec 1, 2, Q1, Q2 |
| Hard vs soft margin? What does C control? | [SVM][svm] sec 3, Q3, Q6 |
| What are support vectors? Primal vs dual? | [SVM][svm] sec 4, Q4, Q7 |
| Hinge loss vs log loss? | [SVM][svm] sec 2, Q5 |
| Kernel trick, Mercer's condition, RBF and gamma? | [SVM][svm] sec 5, Q8, Q9 |
| Multi-class (one-vs-one vs one-vs-rest) and probabilities (Platt scaling)? | [SVM][svm] sec 8, Q12 |
| How does SVR work? What would you use on 10 million rows? | [SVM][svm] Q13, 9, Q14 |
| SVM vs logistic regression vs trees? | [SVM][svm] sec 10, Q15 |

### KNN

| Question | Where |
|---|---|
| Explain KNN. Why is it called lazy? | [KNN][knn] sec 1, Q1, Q2 |
| How do you choose K? Why does KNN need scaling? | [KNN][knn] sec 5, 4, Q3, Q4 |
| Time complexity? How does class imbalance affect it? | [KNN][knn] sec 2, Q7, Q9 |
| KNN vs K-Means? KNN vs logistic regression? Other uses (imputation, LOF)? | [KNN][knn] sec 11, Q10-Q12 |

### Probabilistic models

| Question | Where |
|---|---|
| Bayes' theorem? Why is Naive Bayes "naive", and why does it still work? | [Naive Bayes][nb] sec 3, 4, 24 |
| Gaussian vs multinomial vs Bernoulli NB? Laplace smoothing? | [Naive Bayes][nb] sec 8, 16, 25 |
| Naive Bayes vs logistic regression? | [Naive Bayes][nb] sec 27; [13/1][f1] Q15 |
| LDA: Fisher's criterion and assumptions? LDA vs QDA vs logistic regression? | [LDA][lda] sec 7, 24-26; [QDA][qda] sec 5, 8, 19 |
| GMM: soft clustering, the EM algorithm (E-step, M-step)? | [GMM][gmm] sec 6, 7, 9-11 |
| Covariance types? Choosing the number of components with AIC or BIC? | [GMM][gmm] sec 22, 24-28 |
| GMM vs K-Means? | [GMM][gmm] sec 32-35 |

### Clustering

| Question | Where |
|---|---|
| How does K-Means work, and what does it minimize (inertia)? | [K-Means][km] sec 6, 10, 18 |
| How do you choose K (elbow, silhouette)? | [K-Means][km] sec 19-21 |
| K-Means++ and `n_init`? | [K-Means][km] sec 23, 26 |
| K-Means assumptions, outliers, scaling? | [K-Means][km] sec 31-33 |
| Hierarchical clustering: linkage types, dendrogram, where to cut? | [hierarchical][hc] sec 4-7, 14, 18 |
| DBSCAN: eps, min_samples, core/border/noise points, choosing eps? | [DBSCAN][db] sec 5-7, 23, 24 |
| K-Means vs hierarchical vs DBSCAN vs GMM? | [DBSCAN][db] sec 33; [K-Means][km] sec 36-38 |

### Dimensionality reduction

| Question | Where |
|---|---|
| PCA step by step (standardize, covariance, eigenvectors, explained variance)? | [PCA][pca] sec 9-17 |
| PCA with SVD? How many components? | [PCA][pca] sec 23, 24, 26 |
| PCA limitations (linear, outliers, scaling)? | [PCA][pca] sec 30, 35, 36 |
| LDA for dimensionality reduction? Why at most K−1 components? PCA vs LDA? | [LDA (dim. red.)][lda2] sec 8, 9, 26, 31 |
| How does t-SNE work (Gaussian vs Student-t, KL divergence, perplexity)? | [t-SNE][tsne] sec 2, 4, Q2-Q5 |
| What can you read from a t-SNE plot? Why no `transform()`? | [t-SNE][tsne] sec 5, 6, Q7, Q9 |
| UMAP: n_neighbors, min_dist, new data? | [UMAP][umap] sec 6, 7, 22 |
| PCA vs t-SNE vs UMAP? | [t-SNE][tsne] sec 7, Q14; [UMAP][umap] sec 25 |

## Specialized topics

### Anomaly detection

| Question | Where |
|---|---|
| Supervised vs semi-supervised vs unsupervised anomaly detection? | [Isolation Forest][if] sec 2 |
| How does Isolation Forest work (path length, contamination)? | [Isolation Forest][if] sec 5-8, 11 |
| `decision_function` vs `score_samples`? Evaluating without labels? | [Isolation Forest][if] sec 20, 21, 24 |
| How does LOF work (k-distance, reachability, local density)? `novelty=True`? | [LOF][lof] sec 4-9, 19 |
| One-Class SVM: what does ν control? | [One-Class SVM][ocsvm] sec 7, 10, 24 |
| Isolation Forest vs LOF vs One-Class SVM? | [Isolation Forest][if] sec 27, 28; [LOF][lof] sec 26 |

### Recommendation systems

| Question | Where |
|---|---|
| Explicit vs implicit feedback? | [recsys][rec] sec 2, 7 |
| Collaborative filtering vs content-based? | [recsys][rec] sec 4, 5 |
| Matrix factorization? | [recsys][rec] sec 6 |
| Two-tower model, retrieval + ranking architecture? | [recsys][rec] sec 9-13 |
| How do recommenders connect to RAG? | [recsys][rec] sec 15 |

### Time series

| Question | Where |
|---|---|
| Trend, seasonality, noise? Lag and rolling features? | [time series][ts] sec 3-5 |
| ACF vs PACF, and choosing p and q? | [time series][ts] sec 6, 7 |
| Stationarity, the ADF test, differencing? | [time series][ts] sec 8-10 |
| How do you split and validate (walk-forward)? | [time series][ts] sec 11, 41 |
| Baselines: naive, seasonal naive, moving average? | [time series][ts] sec 12, 13 |
| Exponential smoothing, Holt, Holt-Winters? | [time series][ts] sec 14-16 |
| AR, MA, ARIMA, SARIMA, SARIMAX? | [time series][ts] sec 17-22 |
| VAR and VARX? | [time series][ts] sec 23, 24 |
| Trees for forecasting? Recursive vs direct multi-step? | [time series][ts] sec 25, 27 |
| Forecast metrics? Point vs probabilistic forecasts? | [time series][ts] sec 26, 36 |
| RNN vs LSTM vs GRU? | [time series][ts] sec 28-31 |
| Transformers and foundation models (PatchTST, TimesFM, Chronos)? | [time series][ts] sec 32-35 |
| Leakage in time series? A production forecasting setup? | [time series][ts] sec 40, 42, 43 |

sklearn syntax for every algorithm: [cheatsheet][cs].

## Not in the notebooks yet

Short answers for now; these still need to go into the notebooks.

| Question | Short answer |
|---|---|
| Gini vs entropy? Information gain? | Both measure how mixed a node is: Gini = 1 − Σp², entropy = −Σp·log₂p. Information gain = parent impurity − weighted child impurity. They usually pick almost the same splits; Gini is a little faster. |
| How do you prune a tree? | Pre-pruning stops early (`max_depth`, `min_samples_leaf`). Post-pruning grows the full tree, then cuts back weak branches (cost-complexity pruning, `ccp_alpha` in sklearn). |
| Why don't trees need feature scaling? | A split compares one feature with a threshold. Scaling doesn't change the order of the values, so the splits stay the same. |
| What is the out-of-bag (OOB) score? | Each tree's bootstrap sample leaves out about 37% of the rows. Scoring each row with only the trees that never saw it gives a free validation score (`oob_score=True`). |
| What are the assumptions of linear regression? | A linear relationship, independent errors, constant error variance (homoscedasticity), normally distributed errors (needed for p-values and intervals), and no strong multicollinearity. |
| Normal equation vs gradient descent? | Normal equation: w = (XᵀX)⁻¹Xᵀy, exact in one step but O(d³) and fails if XᵀX is singular. Gradient descent scales to many features and rows. |
| Batch vs stochastic vs mini-batch gradient descent? | Batch uses all rows per step (stable, slow). Stochastic uses one row (fast, noisy). Mini-batch uses small batches and is the standard in deep learning. |
| How do you read a logistic regression coefficient? | w is the change in log-odds for a one-unit increase in the feature; e^w is the odds ratio. |
| What is softmax regression? | Logistic regression for K classes: softmax turns K scores into probabilities that sum to 1, trained with cross-entropy. |
| Bayesian optimization (Optuna)? | It models score as a function of the hyperparameters and tries next where improvement looks likely, so it needs fewer trials than grid or random search. |
| LIME vs SHAP? | LIME fits a simple model around one prediction using perturbed samples. SHAP uses Shapley values: consistent and additive, but slower except for trees. |
| What is CUPED? | It reduces the variance of an A/B test by adjusting the metric with a pre-experiment covariate (often the same metric before the test), so you need fewer users. |
| The cold-start problem in recommendations? | New users or items have no interactions. Use content features, popularity, onboarding questions, or a two-tower model with content features. |
| KPSS vs ADF? | ADF's null hypothesis is "not stationary"; KPSS's is "stationary". Running both gives a clearer answer. |
| What is MASE? | MAE divided by the in-sample MAE of a (seasonal) naive forecast. Below 1 beats naive. It compares across series of different scales and works with zeros. |
| What is Prophet? | An additive model (trend + seasonality + holidays) that is quick to fit. A good baseline for business series with strong seasonality. |
| Stemming vs lemmatization? | Stemming chops suffixes by rule; lemmatization maps to the dictionary form. Rarely needed with the subword tokenizers of modern models. |
| What is word2vec? | Static word vectors learned by predicting neighbouring words (skip-gram, CBOW). One vector per word in every context, unlike transformer embeddings. |

[f1]: 13-ml-fundamentals/1-ml-fundamentals.ipynb
[f2]: 13-ml-fundamentals/2-ml-metrics.ipynb
[f3]: 13-ml-fundamentals/3-ml-feature-engineering.ipynb
[f4]: 13-ml-fundamentals/4-ml-pipelines-and-hyperparameter-tuning.ipynb
[f5]: 13-ml-fundamentals/5-ml-explainability.ipynb
[f6]: 13-ml-fundamentals/6-ml-deployments.ipynb
[f7]: 13-ml-fundamentals/7-statistics-and-ab-testing.ipynb
[f8]: 13-ml-fundamentals/8-text-classification-basics.ipynb
[l2]: 01-linear-regression/2-linear-regression.ipynb
[l3]: 01-linear-regression/3-linear-regression-multi-features.ipynb
[l4]: 01-linear-regression/4-feature-scaling-&-learning-rate.ipynb
[l5]: 01-linear-regression/5-feature-engineering-&-polynomial-regression.ipynb
[g1]: 02-logistic-regression/1-try-linear-regg-on-classification.ipynb
[g2]: 02-logistic-regression/2-sigmoid-function-&-decision-boundary.ipynb
[g3]: 02-logistic-regression/3-logistic-loss-&-cost.ipynb
[g4]: 02-logistic-regression/4-logistic-gradient-descent.ipynb
[g6]: 02-logistic-regression/6-overfitting-regularization.ipynb
[b1]: 03-decision-trees-and-ensembles/1-decision-tree.ipynb
[b2]: 03-decision-trees-and-ensembles/2-random-forest.ipynb
[b3]: 03-decision-trees-and-ensembles/3-adaboost.ipynb
[b4]: 03-decision-trees-and-ensembles/4-gradient-boost.ipynb
[b5]: 03-decision-trees-and-ensembles/5-XgBoost.ipynb
[b6]: 03-decision-trees-and-ensembles/6-LightGBM.ipynb
[b7]: 03-decision-trees-and-ensembles/7-CatBoost.ipynb
[svm]: 04-SVM/1-SVMs.ipynb
[knn]: 05-KNN/1-KNN.ipynb
[nb]: 06-probabilistic-models/1-Naive-Bayes.ipynb
[lda]: 06-probabilistic-models/2-LDA.ipynb
[qda]: 06-probabilistic-models/3-QDA.ipynb
[gmm]: 06-probabilistic-models/4-GMM.ipynb
[km]: 07-clustering-models/1-k-means.ipynb
[hc]: 07-clustering-models/2-hierarchical.ipynb
[db]: 07-clustering-models/3-DBSCAN.ipynb
[pca]: 08-dimensionality-reduction/1-PCA.ipynb
[lda2]: 08-dimensionality-reduction/2-LDA.ipynb
[tsne]: 08-dimensionality-reduction/3-t-SNE.ipynb
[umap]: 08-dimensionality-reduction/4-UMAP.ipynb
[if]: 09-anomaly-detection/1-isolation-forest.ipynb
[lof]: 09-anomaly-detection/2-LOF.ipynb
[ocsvm]: 09-anomaly-detection/3-one-class-svm.ipynb
[rec]: 10-recommendation-system/1-recommendation-system.ipynb
[ts]: 11-timeseries/1-timeseries.ipynb
[cs]: 12-all-algo-sklearn-cheatsheet/all-algo-sklearn-cheatsheet.ipynb
[p0]: 14-projects/1-project.ipynb
[p1]: 14-projects/1-telecom-customer-churn
[p2]: 14-projects/2-fraud-detection
