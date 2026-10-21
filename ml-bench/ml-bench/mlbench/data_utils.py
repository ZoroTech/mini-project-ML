"""Dataset loading, validation, profiling and feature planning.

Everything in this module is UI-free so it can be unit-tested without
Streamlit. Error messages are written for non-expert users: every
``ValueError`` raised here is safe to show directly in the interface.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

MAX_FILE_MB = 25
MIN_ROWS = 10
HIGH_CARDINALITY_LIMIT = 50

_ID_NAME_PATTERN = re.compile(r"^(id|uuid|guid|index|serial|roll_?no|reg_?no)$", re.IGNORECASE)
_DATE_NAME_PATTERN = re.compile(r"(date|time|day|timestamp)", re.IGNORECASE)
_DATE_VALUE_PATTERN = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV / Excel file into a DataFrame.

    Raises ValueError with a user-friendly message for empty files,
    unsupported formats, unparseable content and oversized uploads.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty. Please choose a file that contains data.")
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(
            f"The file is {size_mb:.1f} MB, which exceeds the {MAX_FILE_MB} MB limit for this app. "
            "Please upload a smaller dataset."
        )

    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        else:
            raise ValueError(
                f"Unsupported file format: '{filename}'. Please upload a .csv or .xlsx file."
            )
    except ValueError:
        raise
    except Exception as exc:  # parser errors, corrupt files, etc.
        raise ValueError(
            "The file could not be read. It may be corrupted or not a valid CSV/Excel file. "
            f"Details: {exc}"
        ) from exc

    if df.shape[1] == 0:
        raise ValueError("The file contains no columns. Please check the file and try again.")
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"The dataset has only {len(df)} rows. At least {MIN_ROWS} rows are needed to train "
            "and evaluate a model with a separate test set."
        )
    # Normalise column names to plain strings and drop fully-empty columns.
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------
def _looks_like_datetime(series: pd.Series) -> bool:
    """Heuristic: could this text column actually be dates?"""
    if is_datetime64_any_dtype(series):
        return True
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    sample = series.dropna().astype(str).head(25)
    if sample.empty:
        return False
    if _DATE_NAME_PATTERN.search(str(series.name or "")):
        hits = sample.str.match(_DATE_VALUE_PATTERN).mean()
        if hits >= 0.6:
            return True
    return bool(sample.str.match(_DATE_VALUE_PATTERN).mean() >= 0.85)


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into numeric / categorical / datetime buckets."""
    numeric, categorical, datetime_cols = [], [], []
    for col in df.columns:
        series = df[col]
        if is_datetime64_any_dtype(series) or _looks_like_datetime(series):
            datetime_cols.append(col)
        elif is_numeric_dtype(series) and not is_bool_dtype(series):
            numeric.append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols}


# ---------------------------------------------------------------------------
# Profiling (Dataset Explorer)
# ---------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return everything the Dataset Explorer page displays."""
    kinds = classify_columns(df)
    kind_of = {}
    for kind, cols in kinds.items():
        for c in cols:
            kind_of[c] = kind

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Detected as": [kind_of[c] for c in df.columns],
            "Missing values": [int(df[c].isna().sum()) for c in df.columns],
            "Missing %": [round(100 * df[c].isna().mean(), 1) for c in df.columns],
            "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    numeric_summary = None
    if kinds["numeric"]:
        numeric_summary = df[kinds["numeric"]].describe().T.round(3)
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "column_info": column_info,
        "numeric_summary": numeric_summary,
        "kinds": kinds,
    }


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------
def suggest_task(df: pd.DataFrame, target: str) -> tuple[str, str]:
    """Suggest classification vs regression for a target column.

    This is only a suggestion — the UI lets the user override it, because a
    numerical target is not automatically a regression target (e.g. a 0/1
    churn flag, or marks graded into a small set of score levels).
    """
    series = df[target]
    n = len(series)
    nunique = series.nunique(dropna=True)

    if is_bool_dtype(series) or not is_numeric_dtype(series):
        return "classification", f"'{target}' is a text/category column with {nunique} distinct values."
    if nunique <= 2:
        return "classification", f"'{target}' has only {nunique} distinct numerical value(s) — a category in disguise."
    if nunique <= 15 and nunique / max(n, 1) < 0.05:
        return (
            "classification",
            f"'{target}' takes only {nunique} distinct values across {n} rows, which usually means "
            "discrete classes (for example grades or rating levels) rather than a continuous quantity.",
        )
    return (
        "regression",
        f"'{target}' is numerical with {nunique} distinct values, which looks like a continuous quantity to predict.",
    )


def validate_target(df: pd.DataFrame, target: str, task: str) -> list[str]:
    """Return a list of blocking problems with the chosen target (empty = OK)."""
    problems: list[str] = []
    if target not in df.columns:
        problems.append(f"Target column '{target}' is not in the dataset.")
        return problems
    series = df[target]
    if series.isna().all():
        problems.append(f"Target column '{target}' contains only missing values.")
    elif series.isna().mean() > 0.2:
        problems.append(
            f"Target column '{target}' is missing in {series.isna().mean():.0%} of rows. "
            "Rows with a missing target cannot be used for training."
        )
    if task == "classification":
        nunique = series.nunique(dropna=True)
        if nunique < 2:
            problems.append("Classification needs at least 2 distinct target classes.")
        elif nunique > 50:
            problems.append(
                f"'{target}' has {nunique} distinct values — too many for a sensible classification "
                "target. Consider regression, or check that the right column was selected."
            )
    if task == "regression" and not is_numeric_dtype(series):
        problems.append(
            f"Regression needs a numerical target, but '{target}' is not numerical. "
            "Switch the task to classification or choose a different target."
        )
    return problems


# ---------------------------------------------------------------------------
# Feature planning
# ---------------------------------------------------------------------------
@dataclass
class FeaturePlan:
    """Which columns become features, which are excluded, and why."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)  # (column, reason)
    warnings: list[str] = field(default_factory=list)


def plan_features(df: pd.DataFrame, target: str) -> FeaturePlan:
    """Decide the feature set for a target, excluding unusable columns.

    Excluded: the target itself, identifier-like columns (recognised by name
    or because every row has a different text value), and categorical columns
    with too many distinct values to encode sensibly. Date/time columns are
    kept and expanded into year/month/day features by the pipeline.
    """
    plan = FeaturePlan()
    kinds = classify_columns(df)
    n_rows = len(df)

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        if series.isna().all():
            plan.excluded.append((col, "column contains only missing values"))
            continue
        if _ID_NAME_PATTERN.match(str(col)):
            plan.excluded.append((col, "name looks like an identifier"))
            continue
        if col in kinds["datetime"]:
            plan.datetime.append(col)
            continue
        if col in kinds["categorical"]:
            nunique = series.nunique(dropna=True)
            if nunique >= n_rows and n_rows > 20:
                plan.excluded.append((col, "every row has a different value — looks like an identifier or free text"))
                continue
            if nunique > HIGH_CARDINALITY_LIMIT:
                plan.excluded.append(
                    (col, f"high-cardinality categorical ({nunique} distinct values > limit of {HIGH_CARDINALITY_LIMIT})")
                )
                continue
            plan.categorical.append(col)
        else:
            if series.nunique(dropna=True) <= 1:
                plan.excluded.append((col, "constant column — carries no information"))
                continue
            plan.numeric.append(col)

    if plan.datetime:
        plan.warnings.append(
            "Date/time column(s) " + ", ".join(plan.datetime) + " will be expanded into "
            "year, month, day and weekday numeric features."
        )
    for col, reason in plan.excluded:
        plan.warnings.append(f"Column '{col}' excluded from features: {reason}.")
    return plan


def leakage_warnings(df: pd.DataFrame, target: str, plan: FeaturePlan) -> list[str]:
    """Heuristic checks for target leakage — features that 'know' the answer.

    These are warnings, not proof: a very high correlation can be legitimate.
    But a feature that duplicates the target, or tracks it almost perfectly,
    usually means information from the future leaked into the inputs and the
    reported metrics will be unrealistically good.
    """
    warnings: list[str] = []
    target_series = df[target]
    target_name = str(target).lower().replace(" ", "").replace("_", "")

    for col in plan.numeric + plan.categorical:
        series = df[col]
        if str(col).lower().replace(" ", "").replace("_", "") == target_name:
            warnings.append(f"'{col}' has the same name as the target — check for a duplicated target column.")
        if series.equals(target_series):
            warnings.append(f"'{col}' is identical to the target column — this is target leakage.")
            continue
        if col in plan.numeric and is_numeric_dtype(target_series):
            pair = pd.DataFrame({"x": series, "y": target_series}).dropna()
            if len(pair) > 10 and pair["x"].std() > 0 and pair["y"].std() > 0:
                corr = abs(pair["x"].corr(pair["y"]))
                if corr > 0.98:
                    warnings.append(
                        f"'{col}' correlates with the target at |r| = {corr:.3f}. Such a near-perfect "
                        "relationship often indicates target leakage — verify this feature would be "
                        "known at prediction time."
                    )
    return warnings


# ---------------------------------------------------------------------------
# Quick-cleaned download (Explorer)
# ---------------------------------------------------------------------------
def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Median/mode imputation over the full dataset.

    Used ONLY for the downloadable 'cleaned copy' in the Explorer. Model
    training never uses this — the training pipeline fits its imputers on
    the training split only, to avoid leaking test information.
    """
    cleaned = df.copy()
    kinds = classify_columns(cleaned)
    for col in kinds["numeric"]:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in kinds["categorical"]:
        mode = cleaned[col].mode(dropna=True)
        if not mode.empty:
            cleaned[col] = cleaned[col].fillna(mode.iloc[0])
    return cleaned


# ---------------------------------------------------------------------------
# Fingerprinting (stale-result invalidation)
# ---------------------------------------------------------------------------
def dataset_fingerprint(df: pd.DataFrame, target: str | None, task: str | None) -> str:
    """A cheap content hash: changes whenever data, target or task changes."""
    hasher = hashlib.sha256()
    hasher.update(f"{df.shape}|{list(df.columns)}|{target}|{task}".encode())
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False)
        hasher.update(str(int(row_hashes.sum())).encode())
    except Exception:
        hasher.update(df.head(50).to_csv(index=False).encode())
    return hasher.hexdigest()[:16]
