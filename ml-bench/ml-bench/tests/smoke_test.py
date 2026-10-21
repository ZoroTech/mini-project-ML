"""End-to-end smoke tests for ML Bench — no Streamlit UI required.

Run:  python tests/smoke_test.py

Covers the delivery checklist: all three demo datasets, real training for
classification and regression, metrics recomputed independently from stored
held-out predictions, target switching, the CSV upload path, batch
prediction (including unseen categories and missing values), identifier
exclusion, cross-validation, and the preprocessed-features export.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlbench import data_utils, evaluation  # noqa: E402
from mlbench.demo_data import DEMO_DATASETS, get_demo_dataset  # noqa: E402
from mlbench.pipeline import get_model_registry, train_models  # noqa: E402

PASS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    PASS.append(name)


def align(frame: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
    """Mirror of the app's prediction alignment (keeps the test honest)."""
    frame = frame.copy()
    for col in train_columns:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[train_columns]


# ---------------------------------------------------------------------------
# 1. Demo datasets: size, schema, reproducibility
# ---------------------------------------------------------------------------
for key, info in DEMO_DATASETS.items():
    df = get_demo_dataset(key)
    again = get_demo_dataset(key)
    check(f"demo '{key}' has >= 500 rows", len(df) >= 500, f"{len(df)} rows")
    check(f"demo '{key}' target present", info["target"] in df.columns)
    check(f"demo '{key}' is reproducible (fixed seed)", df.equals(again))
    suggestion, _ = data_utils.suggest_task(df, info["target"])
    check(f"demo '{key}' task auto-detect = {info['task']}", suggestion == info["task"], f"got {suggestion}")

# ---------------------------------------------------------------------------
# 2. Train every model on every demo; verify metrics are real
# ---------------------------------------------------------------------------
outcomes = {}
for key, info in DEMO_DATASETS.items():
    df = get_demo_dataset(key)
    registry = get_model_registry(info["task"])
    outcome = train_models(df, info["target"], info["task"], list(registry.keys()), test_size=0.2, random_state=42)
    outcomes[key] = outcome
    ok = [n for n, r in outcome.results.items() if r.status == "ok"]
    check(f"demo '{key}': at least 3 models trained", len(ok) >= 3, f"ok={ok}")
    check(
        f"demo '{key}': split sizes add up",
        outcome.n_train + outcome.n_test == len(df.dropna(subset=[info["target"]])),
        f"{outcome.n_train}+{outcome.n_test}",
    )
    for name, result in outcome.results.items():
        if result.status != "ok":
            print(f"       (model {name} failed: {result.error})")
            continue
        check(f"demo '{key}'/{name}: predictions cover the test set", len(result.y_pred) == outcome.n_test)
        if info["task"] == "classification":
            from sklearn.metrics import accuracy_score

            recomputed = accuracy_score(result.y_test, result.y_pred)
            check(
                f"demo '{key}'/{name}: stored accuracy == recomputed from held-out predictions",
                abs(recomputed - result.metrics["Accuracy"]) < 1e-9,
                f"{recomputed:.4f}",
            )
            check(f"demo '{key}'/{name}: accuracy is meaningful (> 0.4)", result.metrics["Accuracy"] > 0.4,
                  f"{result.metrics['Accuracy']:.3f}")
            if result.y_proba is not None:
                sums = result.y_proba.sum(axis=1)
                check(f"demo '{key}'/{name}: class probabilities sum to 1",
                      bool(np.allclose(sums, 1.0, atol=1e-4)))
                roc = evaluation.roc_curve_data(result.y_test, result.y_proba, result.classes)
                check(f"demo '{key}'/{name}: ROC data computable", roc is not None and len(roc["curves"]) >= 1)
        else:
            from sklearn.metrics import mean_absolute_error

            recomputed = mean_absolute_error(result.y_test, result.y_pred)
            check(
                f"demo '{key}'/{name}: stored MAE == recomputed from held-out predictions",
                abs(recomputed - result.metrics["MAE"]) < 1e-6,
                f"{recomputed:.2f}",
            )
            check(f"demo '{key}'/{name}: R2 is meaningful (> 0.5)", result.metrics["R2"] > 0.5,
                  f"{result.metrics['R2']:.3f}")

# ---------------------------------------------------------------------------
# 3. Cross-validation runs and stays separate from test metrics
# ---------------------------------------------------------------------------
df = get_demo_dataset("churn")
outcome_cv = train_models(df, "Churn", "classification", ["Random Forest"], cv_folds=3, random_state=42)
rf = outcome_cv.results["Random Forest"]
check("CV: Random Forest reports CV scores", rf.cv is not None and "F1 (macro)" in rf.cv, str(rf.cv))
check("CV: test metrics still present alongside CV", "F1 (macro)" in rf.metrics)

# ---------------------------------------------------------------------------
# 4. CSV upload path (bytes -> DataFrame -> train)
# ---------------------------------------------------------------------------
csv_bytes = get_demo_dataset("churn").to_csv(index=False).encode("utf-8")
loaded = data_utils.load_uploaded_file(csv_bytes, "churn_upload.csv")
check("upload: CSV parses to the same shape", loaded.shape == get_demo_dataset("churn").shape)
try:
    data_utils.load_uploaded_file(b"", "empty.csv")
    check("upload: empty file rejected", False)
except ValueError as exc:
    check("upload: empty file rejected", "empty" in str(exc).lower(), str(exc))
try:
    data_utils.load_uploaded_file(b"a,b\n1,2\n", "tiny.csv")
    check("upload: too-few-rows file rejected", False)
except ValueError as exc:
    check("upload: too-few-rows file rejected", "at least" in str(exc).lower(), str(exc))
try:
    data_utils.load_uploaded_file(b"hello", "notes.txt")
    check("upload: unsupported format rejected", False)
except ValueError as exc:
    check("upload: unsupported format rejected", "Unsupported" in str(exc), str(exc))

# Excel path
buf = io.BytesIO()
get_demo_dataset("house").to_excel(buf, index=False, engine="openpyxl")
loaded_xl = data_utils.load_uploaded_file(buf.getvalue(), "houses.xlsx")
check("upload: Excel parses", loaded_xl.shape == get_demo_dataset("house").shape)

# ---------------------------------------------------------------------------
# 5. Batch prediction: no target column, unseen category, missing values
# ---------------------------------------------------------------------------
outcome = outcomes["churn"]
rf = outcome.results["Random Forest"]
new_records = get_demo_dataset("churn").drop(columns=["Churn"]).head(12).copy()
new_records.loc[new_records.index[0], "Contract_Type"] = "Decade plan"  # unseen category
new_records.loc[new_records.index[1], "Monthly_Charges"] = np.nan      # missing value
aligned = align(new_records, outcome.train_columns)
preds = rf.pipeline.predict(aligned)
check("batch: predictions returned for all rows", len(preds) == 12)
decoded = outcome.label_encoder.inverse_transform(np.asarray(preds).astype(int))
check("batch: predictions decode to real class labels", set(decoded) <= {"Yes", "No"}, str(set(decoded)))
proba = rf.pipeline.predict_proba(aligned)
check("batch: probabilities sum to 1 per row", bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)))

# ---------------------------------------------------------------------------
# 6. Identifier exclusion + prediction without the identifier column
# ---------------------------------------------------------------------------
df_id = get_demo_dataset("student").copy()
df_id.insert(0, "Student_ID", [f"STU-{i:04d}" for i in range(len(df_id))])
plan = data_utils.plan_features(df_id, "Performance")
excluded_cols = [c for c, _ in plan.excluded]
check("plan: identifier column excluded", "Student_ID" in excluded_cols, str(plan.excluded))
outcome_id = train_models(df_id, "Performance", "classification", ["Random Forest"], random_state=42)
rf_id = outcome_id.results["Random Forest"]
check("plan: model trains with identifier present in data", rf_id.status == "ok", rf_id.error or "")
one = df_id.drop(columns=["Performance", "Student_ID"]).head(3)  # no ID at prediction time
pred_one = rf_id.pipeline.predict(align(one, outcome_id.train_columns))
check("plan: prediction works without the identifier column", len(pred_one) == 3)

# Leakage warning fires on a duplicated target
df_leak = get_demo_dataset("student").copy()
df_leak["Performance_Copy"] = df_leak["Performance"]
plan_leak = data_utils.plan_features(df_leak, "Performance")
leaks = data_utils.leakage_warnings(df_leak, "Performance", plan_leak)
check("leakage: duplicated-target feature is flagged", any("Performance_Copy" in w for w in leaks), str(leaks))

# ---------------------------------------------------------------------------
# 7. Changing the target changes the experiment (fingerprint + retrain)
# ---------------------------------------------------------------------------
fp_a = data_utils.dataset_fingerprint(get_demo_dataset("house"), "House_Price", "regression")
fp_b = data_utils.dataset_fingerprint(get_demo_dataset("house"), "Bedrooms", "classification")
check("fingerprint: changes when the target changes", fp_a != fp_b)
outcome_bed = train_models(get_demo_dataset("house"), "Bedrooms", "classification", ["Decision Tree"], random_state=42)
check("target switch: same dataset trains as classification on a new target",
      outcome_bed.results["Decision Tree"].status == "ok")

# ---------------------------------------------------------------------------
# 8. Missing values + heavy categoricals still train end-to-end
# ---------------------------------------------------------------------------
df_nan = get_demo_dataset("house")
outcome_nan = train_models(df_nan, "House_Price", "regression", ["Linear Regression", "Random Forest"], random_state=42)
check("robustness: regression trains on demo with categoricals",
      all(r.status == "ok" for r in outcome_nan.results.values()))

# ---------------------------------------------------------------------------
# 9. Preprocessed-features export (what the Comparison page downloads)
# ---------------------------------------------------------------------------
result = outcomes["house"].results["Random Forest"]
X_full = get_demo_dataset("house").drop(columns=["House_Price"])
transformed = result.pipeline[:-1].transform(X_full)
names_out = list(result.pipeline[:-1].get_feature_names_out())
check("export: transformed matrix has named columns", transformed.shape[1] == len(names_out),
      f"{transformed.shape[1]} cols")
check("export: no NaNs survive the fitted preprocessing", not bool(np.isnan(transformed).any()))

# ---------------------------------------------------------------------------
# 10. Feature importance: native (trees) and permutation (logistic regression)
# ---------------------------------------------------------------------------
imp_tree, note_tree = evaluation.feature_importance(outcomes["student"].results["Random Forest"], "classification")
check("importance: Random Forest native importance available", imp_tree is not None and len(imp_tree) > 0, note_tree[:60])
imp_lr, note_lr = evaluation.feature_importance(outcomes["churn"].results["Logistic Regression"], "classification")
check("importance: Logistic Regression permutation importance available", imp_lr is not None and len(imp_lr) > 0, note_lr[:60])

# ---------------------------------------------------------------------------
# 11. Quick-cleaned download keeps shape and removes NaNs from numeric cols
# ---------------------------------------------------------------------------
cleaned = data_utils.quick_clean(get_demo_dataset("student"))
check("quick clean: no NaNs left in numeric columns",
      int(cleaned.select_dtypes(include=[np.number]).isna().sum().sum()) == 0)

# ---------------------------------------------------------------------------
# 12. Joblib round-trip of a fitted pipeline
# ---------------------------------------------------------------------------
import joblib  # noqa: E402

buf = io.BytesIO()
joblib.dump(outcomes["student"].results["Random Forest"].pipeline, buf)
buf.seek(0)
restored = joblib.load(buf)
sample = align(get_demo_dataset("student").drop(columns=["Performance"]).head(5), outcomes["student"].train_columns)
check("joblib: restored pipeline predicts identically",
      bool((restored.predict(sample) == outcomes["student"].results["Random Forest"].pipeline.predict(sample)).all()))

print(f"\nAll {len(PASS)} checks passed.")
