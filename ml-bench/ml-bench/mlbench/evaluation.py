"""Evaluation: metrics, ROC data, feature importance, best-model selection.

All metrics here are computed from real held-out predictions. Nothing in
this module invents numbers: if a metric cannot be computed (for example
ROC-AUC for a model without probabilities), it is simply absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# True = higher is better, False = lower is better.
METRIC_DIRECTIONS: dict[str, bool] = {
    "Accuracy": True,
    "Precision (macro)": True,
    "Recall (macro)": True,
    "F1 (macro)": True,
    "F1 (weighted)": True,
    "ROC-AUC": True,
    "MAE": False,
    "MSE": False,
    "RMSE": False,
    "R2": True,
}

# Metrics shown in the comparison table, per task.
CLASSIFICATION_TABLE_METRICS = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)", "ROC-AUC"]
REGRESSION_TABLE_METRICS = ["MAE", "MSE", "RMSE", "R2"]


def classification_metrics(y_true, y_pred, y_proba: np.ndarray | None, classes: list) -> dict:
    """Full classification metric set from held-out predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (macro)": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall (macro)": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 (macro)": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 (weighted)": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    # ROC-AUC: binary uses the positive-class probability; multiclass uses
    # one-vs-rest averaging. Both need probabilities and >= 2 classes present.
    if y_proba is not None and len(classes) >= 2:
        try:
            if len(classes) == 2:
                class_index = {c: i for i, c in enumerate(classes)}
                y_true_bin = np.array([class_index[v] for v in y_true])
                metrics["ROC-AUC"] = float(roc_auc_score(y_true_bin, y_proba[:, 1]))
            else:
                y_bin = label_binarize(y_true, classes=classes)
                if y_bin.shape[1] == y_proba.shape[1]:
                    metrics["ROC-AUC"] = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
        except Exception:
            pass  # AUC not computable for this label/probability layout — omit it.

    metrics["_confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    metrics["_report"] = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    """Full regression metric set from held-out predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2),
    }


def roc_curve_data(y_true, y_proba: np.ndarray, classes: list) -> dict | None:
    """Data for a Plotly ROC chart. None when ROC is not applicable.

    Binary: a single curve for the positive class. Multiclass: one
    one-vs-rest curve per class plus a macro-average curve — the standard
    way to generalise ROC beyond two classes.
    """
    if y_proba is None or len(classes) < 2:
        return None
    y_true = np.asarray(y_true)
    try:
        if len(classes) == 2:
            class_index = {c: i for i, c in enumerate(classes)}
            y_bin = np.array([class_index[v] for v in y_true])
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, 1])
            auc = roc_auc_score(y_bin, y_proba[:, 1])
            return {"curves": [{"label": f"Class '{classes[1]}' (AUC = {auc:.3f})", "fpr": fpr.tolist(), "tpr": tpr.tolist()}]}
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] != y_proba.shape[1]:
            return None
        curves = []
        all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, i], y_proba[:, i])[0] for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            curves.append({"label": f"Class '{cls}' vs rest (AUC = {auc:.3f})", "fpr": fpr.tolist(), "tpr": tpr.tolist()})
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(classes)
        macro_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        curves.append(
            {"label": f"Macro average (AUC = {macro_auc:.3f})", "fpr": all_fpr.tolist(), "tpr": mean_tpr.tolist(), "macro": True}
        )
        return {"curves": curves}
    except Exception:
        return None


def feature_importance(result, task: str, max_features: int = 15):
    """Return (DataFrame of importances, method_note) or (None, reason).

    Tree-based models expose native impurity importance. Everything else
    falls back to permutation importance measured on held-out data with the
    full fitted pipeline — model-agnostic and honest about what it measures.
    """
    pipe = result.pipeline
    if pipe is None:
        return None, "Model did not train successfully."
    model = pipe.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        try:
            names = pipe[:-1].get_feature_names_out()
            values = np.asarray(model.feature_importances_, dtype=float)
            df = pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False)
            return df.head(max_features).reset_index(drop=True), (
                "Native tree-based importance: how much each feature reduced impurity "
                "across the trees, measured on the training data."
            )
        except Exception:
            pass  # fall through to permutation importance

    # Permutation importance with the full pipeline on held-out data.
    # result.X_test is the raw (pre-preprocessing) test split stored at
    # training time, aligned row-for-row with result.y_test.
    X_perm = getattr(result, "X_test", None)
    if X_perm is None or result.y_test is None:
        return None, "Held-out inputs were not retained for this result, so importance cannot be computed."
    try:
        scoring = "f1_macro" if task == "classification" else "r2"
        if task == "classification" and result.classes is not None:
            # The pipeline was fitted on label-encoded targets; map back.
            index = {c: i for i, c in enumerate(result.classes)}
            y_perm = np.array([index[v] for v in result.y_test])
        else:
            y_perm = np.asarray(result.y_test, dtype=float)
        if len(X_perm) > 500:  # keep the repeated scoring fast on big test sets
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_perm), size=500, replace=False)
            X_perm = X_perm.iloc[idx]
            y_perm = y_perm[idx]
        perm = permutation_importance(pipe, X_perm, y_perm, n_repeats=5, random_state=42, scoring=scoring)
        df = pd.DataFrame({"Feature": list(X_perm.columns), "Importance": perm.importances_mean}).sort_values(
            "Importance", ascending=False
        )
        return df.head(max_features).reset_index(drop=True), (
            "Permutation importance: how much the held-out score drops when one feature's "
            "values are shuffled. It measures reliance on the feature, not causation, and is "
            "computed on the original input columns."
        )
    except Exception as exc:
        return None, f"Feature importance could not be computed for this model ({type(exc).__name__})."


def public_metrics(metrics: dict) -> dict:
    """Strip private keys (confusion matrix, report) from a metrics dict."""
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def best_model_name(results: dict, metric: str) -> str | None:
    """Name of the best *successfully trained* model under one metric."""
    higher = METRIC_DIRECTIONS.get(metric, True)
    best_name, best_value = None, None
    for name, result in results.items():
        if result.status != "ok" or metric not in result.metrics:
            continue
        value = result.metrics[metric]
        if best_value is None or (value > best_value if higher else value < best_value):
            best_name, best_value = name, value
    return best_name


def comparison_table(results: dict, task: str) -> pd.DataFrame:
    """Sortable comparison table: one row per model, real metrics only."""
    wanted = CLASSIFICATION_TABLE_METRICS if task == "classification" else REGRESSION_TABLE_METRICS
    rows = []
    for name, result in results.items():
        row = {"Model": name, "Status": "Trained" if result.status == "ok" else "Failed"}
        for metric in wanted:
            row[metric] = round(float(result.metrics[metric]), 4) if metric in result.metrics else None
        row["Train time (s)"] = round(result.duration_s, 2)
        if result.status == "failed":
            row["Error"] = result.error
        rows.append(row)
    return pd.DataFrame(rows)
