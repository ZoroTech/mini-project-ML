"""Preprocessing, model registry and the training engine.

The training path is built on scikit-learn ``Pipeline`` + ``ColumnTransformer``
so that imputers, encoders and scalers are fitted on the *training split only*
and then applied unchanged to the test split and to future predictions.
That is what prevents data leakage between train and test.

This module is UI-free and fully testable from the command line.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .data_utils import FeaturePlan, plan_features

try:  # XGBoost is optional — the app works fully without it.
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Date/time expansion
# ---------------------------------------------------------------------------
class DateTimeExpander(BaseEstimator, TransformerMixin):
    """Expand date/time columns into numeric year/month/day/weekday features.

    Implemented as a proper sklearn transformer so it lives *inside* the
    pipeline: the exact same expansion is applied at prediction time, and
    the whole pipeline can be saved with joblib.
    """

    def __init__(self, datetime_cols: list[str] | None = None):
        # NOTE: the parameter must be stored unchanged — sklearn's clone()
        # (used by cross-validation) rejects estimators that modify their
        # constructor parameters in __init__.
        self.datetime_cols = datetime_cols

    @property
    def _cols(self) -> list[str]:
        return self.datetime_cols or []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        self.expanded_names_ = []
        for col in self.columns_:
            if col in self._cols:
                self.expanded_names_ += [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday"]
            else:
                self.expanded_names_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._cols:
            if col not in X.columns:
                continue
            parsed = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"] = parsed.dt.year
            X[f"{col}_month"] = parsed.dt.month
            X[f"{col}_day"] = parsed.dt.day
            X[f"{col}_weekday"] = parsed.dt.dayofweek
            X = X.drop(columns=[col])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.expanded_names_, dtype=object)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], object]
    needs_scaling: bool  # distance/margin-based models need scaled features
    kind: str  # 'linear' | 'tree' | 'ensemble' | 'svm' | 'boosting'


def get_model_registry(task: str) -> dict[str, ModelSpec]:
    """All algorithms available for a task, with laptop-friendly defaults."""
    if task == "classification":
        registry = {
            "Logistic Regression": ModelSpec(
                "Logistic Regression",
                lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                needs_scaling=True, kind="linear",
            ),
            "Decision Tree": ModelSpec(
                "Decision Tree",
                lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
                needs_scaling=False, kind="tree",
            ),
            "Random Forest": ModelSpec(
                "Random Forest",
                lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
                needs_scaling=False, kind="ensemble",
            ),
            "Support Vector Machine": ModelSpec(
                "Support Vector Machine",
                lambda: SVC(probability=True, random_state=RANDOM_STATE),
                needs_scaling=True, kind="svm",
            ),
        }
        if XGBOOST_AVAILABLE:
            registry["XGBoost"] = ModelSpec(
                "XGBoost",
                lambda: XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.08,
                    subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=RANDOM_STATE,
                ),
                needs_scaling=False, kind="boosting",
            )
        return registry

    registry = {
        "Linear Regression": ModelSpec(
            "Linear Regression", lambda: LinearRegression(), needs_scaling=True, kind="linear"
        ),
        "Decision Tree": ModelSpec(
            "Decision Tree",
            lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
            needs_scaling=False, kind="tree",
        ),
        "Random Forest": ModelSpec(
            "Random Forest",
            lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
            needs_scaling=False, kind="ensemble",
        ),
        "Support Vector Regression": ModelSpec(
            "Support Vector Regression",
            # SVR is sensitive to the *target's* scale as well (its epsilon tube is
            # in target units), so standardise y inside a TransformedTargetRegressor;
            # predictions are automatically returned in the original units.
            lambda: TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
            needs_scaling=True, kind="svm",
        ),
    }
    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = ModelSpec(
            "XGBoost",
            lambda: XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=RANDOM_STATE,
            ),
            needs_scaling=False, kind="boosting",
        )
    return registry


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
def build_preprocessor(plan: FeaturePlan, scale_numeric: bool) -> ColumnTransformer:
    """Median-impute + (optionally) scale numerics; mode-impute + one-hot categoricals.

    The ColumnTransformer only ever sees the *expanded* column names (after
    DateTimeExpander), so expanded date features are treated as numerics.
    """
    expanded_numeric = list(plan.numeric)
    for col in plan.datetime:
        expanded_numeric += [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday"]

    transformers = []
    if expanded_numeric:
        numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(numeric_steps), expanded_numeric))
    if plan.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(plan.categorical),
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns remain after exclusions. Choose a different target or dataset.")
    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    status: str  # 'ok' | 'failed'
    error: str | None = None
    pipeline: Pipeline | None = None
    metrics: dict = field(default_factory=dict)
    X_test: object | None = None            # raw held-out inputs (DataFrame), aligned with y_test
    y_test: np.ndarray | None = None        # original label/scale values
    y_pred: np.ndarray | None = None        # original label/scale values
    y_proba: np.ndarray | None = None       # aligned with `classes`
    classes: list | None = None             # classification class labels
    duration_s: float = 0.0
    cv: dict | None = None                  # {metric: (mean, std)} on training data
    cv_error: str | None = None             # set when CV itself could not run


@dataclass
class TrainOutcome:
    results: dict[str, ModelResult]
    plan: FeaturePlan
    warnings: list[str]
    label_encoder: LabelEncoder | None  # classification only
    n_train: int
    n_test: int
    feature_meta: dict  # per original input column: kind/options/range for the prediction form
    input_columns: list[str]  # original feature columns a prediction record must provide
    train_columns: list[str] = field(default_factory=list)  # every non-target column seen in training


def _feature_meta(df: pd.DataFrame, plan: FeaturePlan) -> tuple[dict, list[str]]:
    """Metadata used to build prediction input controls from training data."""
    meta: dict = {}
    inputs: list[str] = []
    for col in plan.numeric:
        series = pd.to_numeric(df[col], errors="coerce")
        meta[col] = {
            "kind": "numeric",
            "min": float(series.min()) if series.notna().any() else 0.0,
            "max": float(series.max()) if series.notna().any() else 0.0,
            "default": float(series.median()) if series.notna().any() else 0.0,
        }
        inputs.append(col)
    for col in plan.categorical:
        options = sorted({str(v) for v in df[col].dropna().unique()})
        meta[col] = {"kind": "categorical", "options": options, "default": options[0] if options else ""}
        inputs.append(col)
    for col in plan.datetime:
        parsed = pd.to_datetime(df[col], errors="coerce")
        meta[col] = {
            "kind": "datetime",
            "default": str(parsed.median().date()) if parsed.notna().any() else "2024-01-01",
        }
        inputs.append(col)
    return meta, inputs


def train_models(
    df: pd.DataFrame,
    target: str,
    task: str,
    model_names: list[str],
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    cv_folds: int = 0,
) -> TrainOutcome:
    """Train the selected models and evaluate them on a held-out test split.

    The split happens before any preprocessing is fitted. Cross-validation,
    when requested, runs on the training split only — the test set is never
    used for model selection.
    """
    from .evaluation import classification_metrics, regression_metrics  # local import: avoids a cycle

    work = df.dropna(subset=[target]).copy()
    plan = plan_features(work, target)
    if not (plan.numeric or plan.categorical or plan.datetime):
        raise ValueError("No usable features were found for this target. " + " ".join(plan.warnings))

    X = work.drop(columns=[target])
    y_raw = work[target]

    label_encoder: LabelEncoder | None = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str) if not pd.api.types.is_numeric_dtype(y_raw) else y_raw)
        classes_original = list(label_encoder.classes_)
    else:
        y = pd.to_numeric(y_raw, errors="raise").to_numpy(dtype=float)
        classes_original = None

    # --- split first; nothing is fitted before this line -------------------
    stratify = None
    split_warnings: list[str] = []
    if task == "classification":
        counts = pd.Series(y).value_counts()
        if counts.min() >= 2 and len(work) >= 20:
            stratify = y
        else:
            split_warnings.append(
                "Stratified splitting was not possible (a class has fewer than 2 samples), "
                "so a plain random split was used."
            )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    registry = get_model_registry(task)
    results: dict[str, ModelResult] = {}

    for name in model_names:
        spec = registry.get(name)
        if spec is None:
            results[name] = ModelResult(name=name, status="failed", error="Model is not available in this environment.")
            continue
        started = time.perf_counter()
        try:
            model = spec.factory()
            if XGBOOST_AVAILABLE and spec.kind == "boosting" and task == "classification":
                n_classes = len(np.unique(y_train))
                model.set_params(eval_metric="logloss" if n_classes == 2 else "mlogloss")
            pipe = Pipeline(
                [
                    ("dates", DateTimeExpander(plan.datetime)),
                    ("prep", build_preprocessor(plan, scale_numeric=spec.needs_scaling)),
                    ("model", model),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)

            y_pred_enc = pipe.predict(X_test)
            y_proba = None
            if task == "classification" and hasattr(pipe, "predict_proba"):
                try:
                    y_proba = pipe.predict_proba(X_test)
                except Exception:
                    y_proba = None

            if task == "classification":
                y_test_out = label_encoder.inverse_transform(y_test)
                y_pred_out = label_encoder.inverse_transform(np.asarray(y_pred_enc).astype(int))
                metrics = classification_metrics(y_test_out, y_pred_out, y_proba, classes_original)
            else:
                y_test_out = np.asarray(y_test, dtype=float)
                y_pred_out = np.asarray(y_pred_enc, dtype=float)
                metrics = regression_metrics(y_test_out, y_pred_out)

            cv_scores, cv_error = None, None
            if cv_folds and cv_folds >= 2:
                try:
                    cv_scores = _cross_validate(pipe, X_train, y_train, task, cv_folds, random_state)
                except Exception as cv_exc:  # CV failing must not fail the trained model
                    cv_error = f"{type(cv_exc).__name__}: {cv_exc}"

            results[name] = ModelResult(
                name=name,
                status="ok",
                pipeline=pipe,
                metrics=metrics,
                X_test=X_test,
                y_test=y_test_out,
                y_pred=y_pred_out,
                y_proba=y_proba,
                classes=classes_original,
                duration_s=time.perf_counter() - started,
                cv=cv_scores,
                cv_error=cv_error,
            )
        except Exception as exc:  # one bad model must not sink the others
            results[name] = ModelResult(
                name=name,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.perf_counter() - started,
            )

    meta, inputs = _feature_meta(work, plan)
    return TrainOutcome(
        results=results,
        plan=plan,
        warnings=split_warnings,
        label_encoder=label_encoder,
        n_train=len(X_train),
        n_test=len(X_test),
        feature_meta=meta,
        input_columns=inputs,
        train_columns=list(X.columns),
    )


def _cross_validate(pipe: Pipeline, X_train, y_train, task: str, folds: int, seed: int) -> dict:
    """K-fold CV of the *already structured* pipeline on training data only.

    Note: cross_validate clones the pipeline and refits it per fold, so the
    fitted pipeline used for test evaluation is untouched.
    """
    if task == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scoring = {"Accuracy": "accuracy", "F1 (macro)": "f1_macro"}
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scoring = {"R2": "r2", "RMSE": "neg_root_mean_squared_error"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = cross_validate(pipe, X_train, y_train, cv=splitter, scoring=scoring, n_jobs=None)
    summary = {}
    for label, scorer in scoring.items():
        # cross_validate names result arrays after the dict keys (labels).
        values = out[f"test_{label}"]
        if scorer.startswith("neg_"):
            values = -values
        summary[label] = (float(np.mean(values)), float(np.std(values)))
    return summary
