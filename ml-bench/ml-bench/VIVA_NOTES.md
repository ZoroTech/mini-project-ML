# ML Bench — Project Explanation & Viva Preparation Notes

## One-paragraph project explanation

ML Bench is a local web application that turns any structured dataset into a complete machine
learning experiment. A user uploads a CSV or Excel file (or loads a built-in synthetic demo
dataset), selects the column to predict, and the app suggests whether the problem is
classification or regression. It then trains several standard algorithms — logistic/linear
regression, decision trees, random forests, support vector machines and gradient boosting —
through a scikit-learn Pipeline that handles missing values, categorical encoding and feature
scaling without leaking test information into training. Models are compared on a held-out test
set using task-appropriate metrics, the best model under a user-chosen metric is explained in
plain English, and new records can be predicted one at a time or in batch through the same
fitted pipeline.

## The pipeline, step by step (know this flow)

1. **Split first.** `train_test_split` (stratified for classification when possible) happens
   before any fitting. Nothing about the test set may influence training.
2. **Preprocessing inside a Pipeline.** Numeric columns: median imputation, plus standardisation
   for models that need it (logistic regression, SVM/SVR). Categorical columns: most-frequent
   imputation + one-hot encoding with `handle_unknown="ignore"`. Date columns: expanded into
   year/month/day/weekday numbers. `ColumnTransformer` applies each recipe to the right columns.
3. **Fit on train, evaluate on test.** Every reported metric is computed from predictions on the
   held-out test split. Cross-validation, when enabled, runs on the training split only and is
   reported separately.
4. **Predict through the same pipeline.** The encoder/scaler fitted in training are reused
   exactly — refitting them on prediction data would be a bug, not a feature.

## Likely viva questions

**Q: Why use a Pipeline instead of preprocessing the whole dataset first?**
Because preprocessing statistics (medians, category lists, means/standard deviations) are learned
from data. If they are computed on the full dataset, information from the test set leaks into
training and the evaluation becomes optimistically biased. A Pipeline guarantees the test set is
transformed with training-derived statistics only.

**Q: What is target leakage?**
When a feature contains information that is only available because the target happened — e.g.
predicting exam pass/fail using a "final grade" column. The app warns when a feature correlates
with the target above |r| = 0.98 or duplicates it.

**Q: Why scale features for SVM and logistic regression but not for trees?**
SVMs and logistic regression use distances/gradient magnitudes, so a feature measured in
thousands (salary) would dominate one measured in units (age). Tree splits only compare a
feature against its own thresholds, so monotone rescaling changes nothing.

**Q: Accuracy vs F1 — when is accuracy misleading?**
When classes are imbalanced. If 95% of customers don't churn, a model that always says "no
churn" scores 95% accuracy while being useless. F1 balances precision and recall; macro F1 treats
every class equally.

**Q: How does ROC-AUC work for multiclass problems?**
One-vs-rest: build one binary ROC curve per class (that class vs all others) and average the
AUCs (macro). It is not the same quantity as binary AUC and should not be compared directly.

**Q: Why hold out a test set AND offer cross-validation?**
Cross-validation (on training data) estimates how stable a model is across different subsets —
useful for model selection. The held-out test set is the final unbiased check, used once.
Selecting models on the test set would quietly turn it into training data.

**Q: How does Random Forest reduce overfitting compared to one tree?**
Bagging: each tree trains on a bootstrap sample and a random subset of features, so trees make
different errors; averaging/voting cancels variance while keeping the low bias of deep trees.

**Q: How does XGBoost differ from Random Forest?**
Boosting vs bagging: XGBoost builds trees sequentially, each one fitting the residual errors of
the current ensemble, with regularisation (shrinkage, subsampling, penalties) to control
overfitting. Random Forest trees are independent and averaged.

**Q: How are the demo datasets generated, and why does it matter?**
Programmatically with a fixed seed (`numpy.random.default_rng(42)`), with targets derived from
plausible feature relationships plus noise — for example churn probability is a logistic function
of contract type, charges, tenure and support calls. Fixed seeds make results reproducible;
plausible relationships make the learned patterns explainable. They are still synthetic, so demo
metrics prove the pipeline works, not real-world accuracy.

**Q: What happens with missing values / unseen categories at prediction time?**
Missing numerics are filled with the *training* median; missing categoricals with the training
mode. A category never seen in training produces an all-zeros one-hot row
(`handle_unknown="ignore"`), so prediction never crashes.

**Q: Limitations of the project?**
Single-file tabular data only; categorical columns above 50 distinct values are excluded;
dates are reduced to calendar components (no time-series models); no hyperparameter search
(deliberate — defaults keep training fast and the comparison fair); metrics on small datasets
are noisy, which the app warns about.

## Numbers you can quote (from the verified test run)

- 3 demo datasets: 600 / 800 / 700 records, reproducible from seed 42.
- 5 algorithms per task (including XGBoost when installed).
- Test split: 20% by default (adjustable 10–40%), stratified for classification.
- The automated smoke test trains every model on every demo dataset and asserts that metrics
  are computed from held-out predictions, batch prediction works, and the CSV upload path works.
