"""ML Bench — Machine Learning Experiment Studio.

A local Streamlit application for end-to-end machine learning experiments:
upload a dataset (or use a built-in synthetic demo), pick a target, train
several scikit-learn models through a leakage-safe pipeline, compare them
on a held-out test set, and predict on new records.

Run with:  python -m streamlit run app.py

All computation happens locally. No API keys, no external services, and
uploaded data never leaves this machine.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mlbench import data_utils, evaluation, explanations
from mlbench.demo_data import DEMO_DATASETS, get_demo_dataset
from mlbench.pipeline import get_model_registry, train_models

st.set_page_config(page_title="ML Bench — ML Experiment Studio", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Styling (dark navy / charcoal with a sky-teal accent)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
      [data-testid="stMetric"] {
          background: #111B2E; border: 1px solid #22314E; border-radius: 10px;
          padding: 0.7rem 0.9rem;
      }
      [data-testid="stMetricLabel"] { color: #9FB3CC; }
      [data-testid="stMetricValue"] { color: #E8EEF7; font-size: 1.35rem; }
      .mlbench-card {
          background: #111B2E; border: 1px solid #22314E; border-radius: 10px;
          padding: 1rem 1.2rem; margin-bottom: 1rem;
      }
      .mlbench-title {
          font-size: 2rem; font-weight: 700; color: #E8EEF7; margin-bottom: 0.1rem;
      }
      .mlbench-subtitle { color: #9FB3CC; margin-bottom: 1rem; }
      .synthetic-badge {
          display: inline-block; background: #3B2F12; color: #F5C04E;
          border: 1px solid #8A6D1F; border-radius: 6px; padding: 0.1rem 0.5rem;
          font-size: 0.8rem; font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = ["Dashboard", "Dataset Explorer", "Model Training", "Model Comparison", "Predictions", "About / Help"]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "page": "Dashboard",
    "df": None,
    "dataset_name": None,
    "dataset_source": None,  # 'upload' | 'demo'
    "is_synthetic": False,
    "target": None,
    "task_mode": "Auto-detect",
    "outcome": None,  # TrainOutcome from the last training run
    "outcome_fingerprint": None,
    "train_config": None,
    "importance_cache": {},
    "pending_guided_demo": False,
}
for _key, _value in _DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def reset_experiment() -> None:
    """Return the app to its initial state (Reset Demo / new experiment).

    'page' is a widget key and cannot be assigned mid-run, so navigation
    goes through 'goto_page', which is applied before the sidebar renders.
    """
    for key, value in _DEFAULTS.items():
        if key == "page":
            continue
        st.session_state[key] = value
    st.session_state.goto_page = "Dashboard"


def set_dataset(df: pd.DataFrame, name: str, source: str, synthetic: bool, default_target: str | None = None) -> None:
    """Install a new dataset and invalidate everything derived from the old one."""
    st.session_state.df = df
    st.session_state.dataset_name = name
    st.session_state.dataset_source = source
    st.session_state.is_synthetic = synthetic
    st.session_state.target = default_target
    st.session_state.task_mode = "Auto-detect"
    st.session_state.outcome = None
    st.session_state.outcome_fingerprint = None
    st.session_state.train_config = None
    st.session_state.importance_cache = {}


def resolved_task() -> str | None:
    """The effective task: the user's choice, or the auto-detected suggestion."""
    df, target = st.session_state.df, st.session_state.target
    if df is None or target is None or target not in df.columns:
        return None
    if st.session_state.task_mode == "Auto-detect":
        return data_utils.suggest_task(df, target)[0]
    return st.session_state.task_mode.lower()


def current_fingerprint() -> str | None:
    if st.session_state.df is None:
        return None
    return data_utils.dataset_fingerprint(st.session_state.df, st.session_state.target, resolved_task())


def valid_outcome():
    """The stored training outcome, or None if inputs changed since training."""
    outcome = st.session_state.outcome
    if outcome is None:
        return None
    if st.session_state.outcome_fingerprint != current_fingerprint():
        return None
    return outcome


@st.cache_data(show_spinner=False)
def _parse_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return data_utils.load_uploaded_file(file_bytes, filename)


@st.cache_data(show_spinner=False)
def _load_demo(key: str) -> pd.DataFrame:
    return get_demo_dataset(key)


def _parse_prediction_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse a batch-prediction file (no minimum-row rule — 1 record is fine)."""
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    raise ValueError("Batch prediction needs a .csv or .xlsx file.")


# ---------------------------------------------------------------------------
# Training driver (shared by the Training page and the guided demo)
# ---------------------------------------------------------------------------
def run_training(model_names: list[str], test_size: float, cv_folds: int, seed: int, progress=None) -> None:
    """Train models one at a time so the UI can show real progress."""
    df, target, task = st.session_state.df, st.session_state.target, resolved_task()
    merged_results = {}
    outcome = None
    for i, name in enumerate(model_names):
        if progress is not None:
            progress(i / len(model_names), f"Training {name}…")
        outcome = train_models(df, target, task, [name], test_size=test_size, random_state=seed, cv_folds=cv_folds)
        merged_results.update(outcome.results)
    if progress is not None:
        progress(1.0, "Training complete.")
    outcome.results = merged_results
    st.session_state.outcome = outcome
    st.session_state.outcome_fingerprint = current_fingerprint()
    st.session_state.train_config = {
        "models": model_names, "test_size": test_size, "cv_folds": cv_folds, "seed": seed,
    }
    st.session_state.importance_cache = {}


def run_guided_demo() -> None:
    """One-click demonstration: load Demo A, train all its models, show comparison."""
    with st.status("Running the complete demo…", expanded=True) as status:
        status.write("1/4 — Loading the Student Performance demo dataset (synthetic, 600 records).")
        info = DEMO_DATASETS["student"]
        set_dataset(_load_demo("student").copy(), info["title"], "demo", True, default_target=info["target"])
        st.session_state.task_mode = "Auto-detect"

        status.write(f"2/4 — Target set to '{info['target']}'; task auto-detected as classification.")
        registry = get_model_registry("classification")
        names = list(registry.keys())

        bar = status.progress(0.0, "Starting training…")
        run_training(names, test_size=0.2, cv_folds=0, seed=42,
                     progress=lambda frac, msg: bar.progress(frac, msg))
        status.write(f"3/4 — Trained {len(names)} models and evaluated them on the held-out test split.")
        status.write("4/4 — Opening the comparison page. Predictions are now enabled too.")
        status.update(label="Demo complete — real metrics computed from the demo data.", state="complete")
    st.session_state.goto_page = "Model Comparison"
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧪 ML Bench")
        st.caption("Machine Learning Experiment Studio")
        st.divider()

        st.radio("Navigation", PAGES, key="page")
        st.divider()

        st.markdown("### Current dataset")
        df = st.session_state.df
        if df is None:
            st.info("No dataset loaded yet. Load a demo below or upload a file in Dataset Explorer.")
        else:
            st.write(f"**{st.session_state.dataset_name}**")
            st.caption(f"{len(df):,} rows × {df.shape[1]} columns · source: {st.session_state.dataset_source}")
            if st.session_state.is_synthetic:
                st.markdown('<span class="synthetic-badge">SYNTHETIC DEMO DATA</span>', unsafe_allow_html=True)
            if st.session_state.target:
                st.caption(f"Target: **{st.session_state.target}** · Task: **{resolved_task()}**")
        st.divider()

        st.markdown("### Demo mode")
        demo_key = st.selectbox(
            "Built-in dataset",
            list(DEMO_DATASETS.keys()),
            format_func=lambda k: DEMO_DATASETS[k]["title"],
            key="demo_choice",
        )
        if st.button("Load Demo Dataset", width="stretch"):
            info = DEMO_DATASETS[demo_key]
            set_dataset(_load_demo(demo_key).copy(), info["title"], "demo", True, default_target=info["target"])
            st.session_state.goto_page = "Dataset Explorer"
            st.rerun()
        if st.button("▶ Run Complete Demo", width="stretch", type="primary"):
            st.session_state.pending_guided_demo = True
            st.session_state.goto_page = "Dashboard"
            st.rerun()
        if st.button("Reset Demo", width="stretch"):
            reset_experiment()
            st.rerun()
        st.divider()
        st.caption("Everything runs locally — no accounts, no API keys, uploaded data never leaves this machine.")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    st.markdown('<div class="mlbench-title">ML Bench — Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mlbench-subtitle">Upload a dataset or load a demo, train real models, '
        "compare them on held-out data, and predict.</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.pending_guided_demo:
        st.session_state.pending_guided_demo = False
        run_guided_demo()
        return

    df = st.session_state.df
    if df is None:
        st.markdown(
            """
            <div class="mlbench-card">
            <b>Getting started</b><br>
            1. Load a built-in demo dataset from the sidebar (no files needed), or upload a CSV/Excel in
            <b>Dataset Explorer</b>.<br>
            2. Pick the target column you want to predict.<br>
            3. Train models in <b>Model Training</b>, compare them in <b>Model Comparison</b>, and make
            predictions in <b>Predictions</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Tip: the **▶ Run Complete Demo** button in the sidebar does all of this in one click.")
        return

    outcome = valid_outcome()
    task = resolved_task()
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", st.session_state.dataset_name[:38])
    col2.metric("Rows", f"{len(df):,}")
    col3.metric("Columns", df.shape[1])
    col4, col5, col6 = st.columns(3)
    col4.metric("Target column", st.session_state.target or "— not selected —")
    col5.metric("Task type", task or "—")
    trained = [r for r in outcome.results.values() if r.status == "ok"] if outcome else []
    col6.metric("Models trained", len(trained))

    if st.session_state.is_synthetic:
        st.warning(
            "This is a **synthetic demo dataset**. Metrics computed on it demonstrate the pipeline — "
            "they are not evidence of real-world performance."
        )

    if outcome is not None:
        default_metric = "F1 (macro)" if task == "classification" else "RMSE"
        best = evaluation.best_model_name(outcome.results, default_metric)
        cfg = st.session_state.train_config or {}
        st.markdown("### Latest experiment")
        if best:
            value = outcome.results[best].metrics[default_metric]
            higher = evaluation.METRIC_DIRECTIONS[default_metric]
            st.success(
                f"Best model under **{default_metric}** ({'higher is better' if higher else 'lower is better'}): "
                f"**{best}** — {default_metric} = {value:.4f}. This is the best model *under this metric and "
                "this test split*, not a universal winner."
            )
        st.caption(
            f"Train/test split: {outcome.n_train:,} train / {outcome.n_test:,} test · "
            f"models: {', '.join(cfg.get('models', []))} · "
            f"cross-validation: {'off' if not cfg.get('cv_folds') else str(cfg['cv_folds']) + '-fold (training data)'}"
        )
        st.dataframe(evaluation.comparison_table(outcome.results, task), width="stretch", hide_index=True)
        failed = [r for r in outcome.results.values() if r.status == "failed"]
        for result in failed:
            st.error(f"**{result.name}** failed to train — {result.error}")
    elif st.session_state.target:
        st.info("Dataset and target are set. Head to **Model Training** to train your first models.")
    else:
        st.info("Now choose a target column in **Dataset Explorer**.")


# ---------------------------------------------------------------------------
# Dataset Explorer
# ---------------------------------------------------------------------------
def page_explorer() -> None:
    st.markdown('<div class="mlbench-title">Dataset Explorer</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a dataset (.csv or .xlsx, max 25 MB)", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        cache_key = (uploaded.name, len(file_bytes), hash(file_bytes) & 0xFFFFFFFF)
        if st.session_state.get("_upload_key") != cache_key:
            try:
                df = _parse_upload(file_bytes, uploaded.name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                set_dataset(df, uploaded.name, "upload", False)
                st.session_state._upload_key = cache_key
                st.success(f"Loaded '{uploaded.name}' — {len(df):,} rows × {df.shape[1]} columns.")
                st.rerun()

    df = st.session_state.df
    if df is None:
        st.info(
            "No dataset yet. Upload a CSV/Excel file above, or load a built-in demo from the sidebar — "
            "the demos need no download and are generated locally with a fixed seed."
        )
        for key, info in DEMO_DATASETS.items():
            with st.expander(info["title"]):
                st.markdown(info["description"])
                st.markdown("**Synthetic data** — generated locally, fully reproducible (seed 42).")
        return

    if st.session_state.is_synthetic:
        st.markdown('<span class="synthetic-badge">SYNTHETIC DEMO DATA</span>', unsafe_allow_html=True)
        for info in DEMO_DATASETS.values():
            if info["title"] == st.session_state.dataset_name:
                st.markdown(info["description"])
    profile = data_utils.profile_dataframe(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", profile["cols"])
    c3.metric("Duplicate rows", f"{profile['duplicates']:,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("### Preview (first 20 rows)")
    st.dataframe(df.head(20), width="stretch", hide_index=True)

    st.markdown("### Columns")
    st.dataframe(profile["column_info"], width="stretch", hide_index=True)

    if profile["numeric_summary"] is not None:
        st.markdown("### Numerical summary")
        st.dataframe(profile["numeric_summary"], width="stretch")

    kinds = profile["kinds"]
    if kinds["categorical"]:
        st.markdown("### Categorical value counts")
        cat_col = st.selectbox("Categorical column", kinds["categorical"], key="cat_counts_col")
        counts = df[cat_col].value_counts(dropna=False).head(20)
        st.bar_chart(counts)

    st.divider()
    st.markdown("### Target & task")
    target_options = list(df.columns)
    default_idx = target_options.index(st.session_state.target) if st.session_state.target in target_options else 0
    target = st.selectbox("Target column (what should the model predict?)", target_options, index=default_idx)
    st.session_state.target = target

    mode = st.radio(
        "Task type",
        ["Auto-detect", "Classification", "Regression"],
        index=["Auto-detect", "Classification", "Regression"].index(st.session_state.task_mode),
        horizontal=True,
        help="Auto-detect suggests a task from the target column's values. A numerical target is not "
             "automatically regression — you can always override the suggestion.",
    )
    st.session_state.task_mode = mode

    suggestion, reason = data_utils.suggest_task(df, target)
    task = resolved_task()
    if mode == "Auto-detect":
        st.info(f"**Auto-detect suggestion: {suggestion}.** {reason} You can override it above at any time.")
    else:
        st.info(f"Task set manually to **{task}**. (Auto-detect would have suggested: {suggestion} — {reason})")

    problems = data_utils.validate_target(df, target, task)
    for problem in problems:
        st.error(problem)
    if not problems:
        plan = data_utils.plan_features(df.dropna(subset=[target]), target)
        used = plan.numeric + plan.categorical + plan.datetime
        st.success(
            f"Ready: **{len(used)} feature column(s)** will be used "
            f"({len(plan.numeric)} numeric, {len(plan.categorical)} categorical, {len(plan.datetime)} date/time), "
            f"target = **{target}**, task = **{task}**."
        )
        for warning in plan.warnings:
            st.warning(warning)
        for warning in data_utils.leakage_warnings(df, target, plan):
            st.warning(f"⚠️ Possible target leakage — {warning}")

    st.divider()
    st.markdown("### Downloads")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download dataset as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="mlbench_dataset.csv",
        mime="text/csv",
        width="stretch",
    )
    cleaned = data_utils.quick_clean(df)
    d2.download_button(
        "Download cleaned copy (imputed)",
        cleaned.to_csv(index=False).encode("utf-8"),
        file_name="mlbench_dataset_cleaned.csv",
        mime="text/csv",
        width="stretch",
        help="Median/mode imputation applied to the full dataset. This is a convenience copy only — "
             "model training always fits imputers on the training split alone to avoid data leakage.",
    )
    st.caption("The original uploaded data is never modified — cleaning happens on copies.")


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
def page_training() -> None:
    st.markdown('<div class="mlbench-title">Model Training</div>', unsafe_allow_html=True)
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset first — from the sidebar demos or in Dataset Explorer.")
        return
    target, task = st.session_state.target, resolved_task()
    if target is None:
        st.warning("No target column selected yet. Choose one in **Dataset Explorer**.")
        return

    problems = data_utils.validate_target(df, target, task)
    for problem in problems:
        st.error(problem)
    if problems:
        return

    work = df.dropna(subset=[target])
    plan = data_utils.plan_features(work, target)
    st.markdown(
        f"**Dataset:** {st.session_state.dataset_name} · **Target:** {target} · **Task:** {task} · "
        f"**Usable rows:** {len(work):,} (rows with a missing target are dropped)"
    )
    feat_cols = st.columns(3)
    feat_cols[0].markdown("**Numeric features**\n\n" + ("\n".join(f"- {c}" for c in plan.numeric) or "_none_"))
    feat_cols[1].markdown("**Categorical features**\n\n" + ("\n".join(f"- {c}" for c in plan.categorical) or "_none_"))
    feat_cols[2].markdown("**Date/time features**\n\n" + ("\n".join(f"- {c}" for c in plan.datetime) or "_none_"))
    if plan.excluded:
        st.markdown("**Excluded columns**")
        for col, reason in plan.excluded:
            st.caption(f"• {col} — {reason}")
    for warning in plan.warnings:
        st.warning(warning)
    for warning in data_utils.leakage_warnings(df, target, plan):
        st.warning(f"⚠️ Possible target leakage — {warning}")
    if len(work) < 100:
        st.warning(
            f"This dataset has only {len(work)} usable rows. Models can still be trained, but test-set "
            "metrics will be noisy — treat them as rough indications, not precise measurements."
        )

    registry = get_model_registry(task)
    st.markdown("### Configuration")
    selected = st.multiselect(
        "Models to train", list(registry.keys()), default=list(registry.keys()), key="model_select"
    )
    cfg1, cfg2, cfg3 = st.columns(3)
    test_pct = cfg1.slider("Test set size (%)", 10, 40, 20, step=5, key="test_pct")
    use_cv = cfg2.checkbox("K-fold cross-validation (on training data)", value=False, key="use_cv")
    cv_folds = cfg3.selectbox("CV folds", [3, 5], index=1, key="cv_folds", disabled=not use_cv)
    with st.expander("Advanced"):
        seed = st.number_input("Random seed (reproducibility)", min_value=0, max_value=9999, value=42, key="seed")
        st.caption(
            "Preprocessing (imputation, one-hot encoding, scaling where a model needs it) is fitted "
            "inside an sklearn Pipeline on the training split only — the test split is never seen "
            "during fitting, so metrics are honest."
        )

    if st.button("🚀 Train models", type="primary", disabled=not selected):
        bar = st.progress(0.0, "Starting…")
        run_training(selected, test_size=test_pct / 100.0, cv_folds=cv_folds if use_cv else 0,
                     seed=int(seed), progress=lambda f, m: bar.progress(f, m))
        bar.empty()
        st.rerun()

    outcome = valid_outcome()
    if outcome is not None and st.session_state.outcome_fingerprint == current_fingerprint():
        st.markdown("### Last training run")
        st.caption(f"{outcome.n_train:,} training rows · {outcome.n_test:,} test rows")
        st.dataframe(evaluation.comparison_table(outcome.results, task), width="stretch", hide_index=True)
        for result in outcome.results.values():
            if result.status == "failed":
                st.error(f"**{result.name}** failed — {result.error}")
        st.info("Open **Model Comparison** for charts, confusion matrices, ROC curves and feature importance.")
    st.caption(explanations.GENERALISATION_WARNING)


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------
def _confusion_matrix_fig(result) -> go.Figure:
    cm = np.array(result.metrics["_confusion_matrix"])
    labels = [str(c) for c in result.classes]
    fig = px.imshow(
        cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual"}, title=f"Confusion matrix — {result.name}",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0A1120", plot_bgcolor="#0A1120")
    return fig


def _roc_fig(result) -> go.Figure | None:
    data = evaluation.roc_curve_data(result.y_test, result.y_proba, result.classes)
    if data is None:
        return None
    fig = go.Figure()
    for curve in data["curves"]:
        fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=curve["label"],
                                 line=dict(width=3 if curve.get("macro") else 2, dash="dash" if curve.get("macro") else "solid")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance (0.500)",
                             line=dict(color="#5B6B82", dash="dot")))
    fig.update_layout(
        title=f"ROC curves — {result.name}", xaxis_title="False positive rate", yaxis_title="True positive rate",
        template="plotly_dark", paper_bgcolor="#0A1120", plot_bgcolor="#0A1120",
    )
    return fig


def page_comparison() -> None:
    st.markdown('<div class="mlbench-title">Model Comparison</div>', unsafe_allow_html=True)
    outcome = valid_outcome()
    if outcome is None:
        if st.session_state.outcome is not None:
            st.warning("The dataset, target or task changed since the last training run, so those results were discarded. Train again in **Model Training**.")
        else:
            st.info("No models trained yet. Train some in **Model Training** first.")
        return
    task = resolved_task()
    results = outcome.results
    ok_results = {n: r for n, r in results.items() if r.status == "ok"}
    if not ok_results:
        st.error("Every selected model failed to train. See Model Training for the per-model errors.")
        return

    table_metrics = evaluation.CLASSIFICATION_TABLE_METRICS if task == "classification" else evaluation.REGRESSION_TABLE_METRICS
    available = [m for m in table_metrics if any(m in r.metrics for r in ok_results.values())]
    default_metric = "F1 (macro)" if task == "classification" else "RMSE"
    metric = st.selectbox(
        "Compare by metric", available,
        index=available.index(default_metric) if default_metric in available else 0,
        key="compare_metric",
    )
    higher = evaluation.METRIC_DIRECTIONS[metric]
    best_name = evaluation.best_model_name(results, metric)
    best_value = ok_results[best_name].metrics[metric]

    st.success(
        f"**Best under {metric}: {best_name}** ({metric} = {best_value:.4f}, "
        f"{'higher is better' if higher else 'lower is better'}). "
        + explanations.METRIC_DESCRIPTIONS.get(metric, "")
        + " This ranking applies to this metric and this held-out test split only — a different metric "
          "can legitimately crown a different model."
    )

    st.markdown("### Test-set results (held out — never used in training)")
    st.dataframe(evaluation.comparison_table(results, task), width="stretch", hide_index=True)
    for result in results.values():
        if result.status == "failed":
            st.error(f"**{result.name}** failed — {result.error}")

    chart_col, time_col = st.columns(2)
    names = [n for n, r in ok_results.items() if metric in r.metrics]
    fig = px.bar(
        x=names, y=[ok_results[n].metrics[metric] for n in names],
        labels={"x": "Model", "y": metric}, title=f"{metric} by model (test set)",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0A1120", plot_bgcolor="#0A1120")
    chart_col.plotly_chart(fig, width="stretch")
    fig_time = px.bar(
        x=list(results.keys()), y=[r.duration_s for r in results.values()],
        labels={"x": "Model", "y": "Seconds"}, title="Training duration",
    )
    fig_time.update_layout(template="plotly_dark", paper_bgcolor="#0A1120", plot_bgcolor="#0A1120")
    time_col.plotly_chart(fig_time, width="stretch")

    # --- cross-validation, kept visibly separate from test results --------
    if any(r.cv for r in ok_results.values()) or any(r.cv_error for r in ok_results.values()):
        st.markdown("### Cross-validation results (training data only)")
        st.caption(
            "Cross-validation estimates stability during model selection. It is computed on the "
            "training split only and is reported separately from the held-out test results above."
        )
        cv_rows = []
        for name, result in ok_results.items():
            if result.cv:
                row = {"Model": name}
                for m, (mean, std) in result.cv.items():
                    row[m] = f"{mean:.4f} ± {std:.4f}"
                cv_rows.append(row)
            elif result.cv_error:
                st.warning(f"Cross-validation for {name} could not run: {result.cv_error}")
        if cv_rows:
            st.dataframe(pd.DataFrame(cv_rows), width="stretch", hide_index=True)

    # --- per-model detail ---------------------------------------------------
    st.markdown("### Model detail")
    detail_name = st.selectbox("Inspect model", list(ok_results.keys()), key="detail_model")
    result = ok_results[detail_name]
    st.markdown(f"**About this result.** {explanations.explain_result(result, task, metric)}")

    if task == "classification":
        left, right = st.columns(2)
        left.plotly_chart(_confusion_matrix_fig(result), width="stretch")
        roc = _roc_fig(result)
        if roc is not None:
            right.plotly_chart(roc, width="stretch")
            right.caption(
                "Binary targets give one ROC curve. Multiclass targets use one-vs-rest curves per class "
                "plus a macro average — the two situations are not directly interchangeable."
            )
        else:
            right.info("ROC is unavailable for this model (it does not provide class probabilities).")
        report = result.metrics.get("_report", {})
        report_rows = {k: v for k, v in report.items() if k != "accuracy"}
        if report_rows:
            st.markdown("**Per-class classification report (test set)**")
            st.dataframe(pd.DataFrame(report_rows).T.round(3), width="stretch")
    else:
        y_true = np.asarray(result.y_test, dtype=float)
        y_pred = np.asarray(result.y_pred, dtype=float)
        left, right = st.columns(2)
        fig_avp = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                             title=f"Actual vs predicted — {detail_name}")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        fig_avp.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect prediction",
                                     line=dict(color="#5B6B82", dash="dot")))
        fig_avp.update_layout(template="plotly_dark", paper_bgcolor="#0A1120", plot_bgcolor="#0A1120")
        left.plotly_chart(fig_avp, width="stretch")
        residuals = y_true - y_pred
        fig_res = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residual (actual − predicted)"},
                             title=f"Residuals — {detail_name}")
        fig_res.add_hline(y=0, line_color="#5B6B82", line_dash="dot")
        fig_res.update_layout(template="plotly_dark", paper_bgcolor="#0A1120", plot_bgcolor="#0A1120")
        right.plotly_chart(fig_res, width="stretch")
        right.caption("Good residuals scatter randomly around zero. A curve or funnel shape means the model misses a pattern.")

    # --- feature importance ---------------------------------------------------
    st.markdown("### Feature importance")
    cache_key = (st.session_state.outcome_fingerprint, detail_name)
    if st.button(f"Compute feature importance for {detail_name}", key="fi_button"):
        with st.spinner("Computing feature importance…"):
            st.session_state.importance_cache[cache_key] = evaluation.feature_importance(result, task)
    if cache_key in st.session_state.importance_cache:
        imp_df, note = st.session_state.importance_cache[cache_key]
        if imp_df is not None:
            fig_imp = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                             title=f"Feature importance — {detail_name}")
            fig_imp.update_layout(template="plotly_dark", paper_bgcolor="#0A1120", plot_bgcolor="#0A1120")
            st.plotly_chart(fig_imp, width="stretch")
        st.caption(note)

    # --- export fitted preprocessing -----------------------------------------
    st.markdown("### Export")
    if st.button("Prepare preprocessed-features CSV (this model)", key="prep_button"):
        try:
            X_full = st.session_state.df.dropna(subset=[st.session_state.target]).drop(columns=[st.session_state.target])
            transformed = result.pipeline[:-1].transform(X_full)
            names_out = list(result.pipeline[:-1].get_feature_names_out())
            export_df = pd.DataFrame(transformed, columns=names_out)
            export_df[st.session_state.target] = st.session_state.df.dropna(subset=[st.session_state.target])[st.session_state.target].values
            st.download_button(
                "Download preprocessed features CSV",
                export_df.to_csv(index=False).encode("utf-8"),
                file_name="mlbench_preprocessed_features.csv",
                mime="text/csv",
            )
            st.caption("Features after the fitted pipeline (imputation, encoding, scaling as applicable) — the exact numbers the model consumed.")
        except Exception as exc:
            st.error(f"Could not export preprocessed features: {exc}")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def _align_input(df_new: pd.DataFrame, outcome) -> pd.DataFrame:
    """Reorder/extend a prediction frame to the exact training column layout.

    Missing non-feature columns are added as NaN (they were excluded from the
    pipeline anyway); missing *feature* columns are caught by validation
    before this is called.
    """
    frame = df_new.copy()
    for col in outcome.train_columns:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[outcome.train_columns]


def _predict_frame(result, outcome, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict with the stored fitted pipeline — preprocessing is NOT refitted."""
    aligned = _align_input(frame, outcome)
    predictions = result.pipeline.predict(aligned)
    out = frame.copy()
    if outcome.label_encoder is not None:
        out["Prediction"] = outcome.label_encoder.inverse_transform(np.asarray(predictions).astype(int))
        if hasattr(result.pipeline, "predict_proba"):
            try:
                proba = result.pipeline.predict_proba(aligned)
                for j, cls in enumerate(result.classes):
                    out[f"Probability_{cls}"] = np.round(proba[:, j], 4)
            except Exception:
                pass
    else:
        out["Prediction"] = np.round(np.asarray(predictions, dtype=float), 4)
    return out


def page_predictions() -> None:
    st.markdown('<div class="mlbench-title">Predictions</div>', unsafe_allow_html=True)
    outcome = valid_outcome()
    if outcome is None:
        st.info("Train at least one model first — predictions use the exact preprocessing pipeline fitted during training.")
        return
    ok_results = {n: r for n, r in outcome.results.items() if r.status == "ok"}
    if not ok_results:
        st.error("No model trained successfully in the last run.")
        return
    task = resolved_task()
    model_name = st.selectbox("Trained model", list(ok_results.keys()), key="pred_model")
    result = ok_results[model_name]
    st.caption(
        f"Inputs expected: {', '.join(outcome.input_columns)}. "
        "The fitted training-time pipeline (imputer, encoder, scaler) is reused exactly as-is — "
        "nothing is refitted on prediction data."
    )

    tab_single, tab_batch = st.tabs(["Single record", "Batch prediction (CSV/Excel)"])

    with tab_single:
        with st.form("single_prediction_form"):
            record: dict = {}
            cols = st.columns(2)
            for i, col in enumerate(outcome.input_columns):
                meta = outcome.feature_meta[col]
                with cols[i % 2]:
                    if meta["kind"] == "numeric":
                        record[col] = st.number_input(
                            col, min_value=float(meta["min"]), max_value=float(meta["max"]),
                            value=float(meta["default"]), key=f"pred_in_{col}",
                        )
                    elif meta["kind"] == "categorical":
                        record[col] = st.selectbox(col, meta["options"], key=f"pred_in_{col}")
                    else:  # datetime
                        picked = st.date_input(col, value=pd.to_datetime(meta["default"]).date(), key=f"pred_in_{col}")
                        record[col] = str(picked)
            submitted = st.form_submit_button("Predict", type="primary")
        if submitted:
            frame = pd.DataFrame([record])
            try:
                out = _predict_frame(result, outcome, frame)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                prediction = out["Prediction"].iloc[0]
                if task == "classification":
                    st.success(f"**Predicted {st.session_state.target}: {prediction}**")
                    proba_cols = [c for c in out.columns if c.startswith("Probability_")]
                    if proba_cols:
                        proba_row = out[proba_cols].iloc[0]
                        proba_df = pd.DataFrame(
                            {"Class": [c.replace("Probability_", "") for c in proba_cols],
                             "Probability": proba_row.values}
                        ).sort_values("Probability", ascending=False)
                        fig = px.bar(proba_df, x="Class", y="Probability", title="Class probabilities")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="#0A1120", plot_bgcolor="#0A1120",
                                          yaxis_range=[0, 1])
                        st.plotly_chart(fig, width="stretch")
                    st.caption(
                        f"The model classified this record as **{prediction}**. Probabilities are the model's "
                        "own confidence estimates from the training distribution — treat extreme confidence "
                        "on unusual inputs with suspicion."
                    )
                else:
                    st.success(f"**Predicted {st.session_state.target}: {float(prediction):,.2f}**")
                    st.caption(
                        "This is the model's point estimate from the fitted pipeline. Check the model's "
                        "MAE/RMSE on the Comparison page for the typical size of its error."
                    )

    with tab_batch:
        st.markdown(
            "Upload a CSV/Excel with the model's input columns (the target column is **not** required — "
            "if present it is ignored). Extra columns are ignored; missing input columns are reported."
        )
        batch_file = st.file_uploader("Batch prediction file", type=["csv", "xlsx", "xls"], key="batch_upload")
        if batch_file is not None:
            try:
                batch_df = _parse_prediction_file(batch_file.getvalue(), batch_file.name)
            except Exception as exc:
                st.error(f"Could not read that file: {exc}")
            else:
                batch_df.columns = [str(c).strip() for c in batch_df.columns]
                missing = [c for c in outcome.input_columns if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing required input column(s): {', '.join(missing)}")
                else:
                    notes = []
                    if st.session_state.target in batch_df.columns:
                        notes.append(f"Target column '{st.session_state.target}' was present and ignored.")
                    extra = [c for c in batch_df.columns if c not in outcome.train_columns and c != st.session_state.target]
                    if extra:
                        notes.append(f"Ignored extra column(s): {', '.join(extra)}.")
                    try:
                        out = _predict_frame(result, outcome, batch_df)
                    except Exception as exc:
                        st.error(f"Batch prediction failed: {exc}")
                    else:
                        for note in notes:
                            st.caption(note)
                        st.success(f"Predicted {len(out):,} record(s) with {model_name}.")
                        st.dataframe(out, width="stretch", hide_index=True)
                        st.download_button(
                            "Download predictions as CSV",
                            out.to_csv(index=False).encode("utf-8"),
                            file_name="mlbench_predictions.csv",
                            mime="text/csv",
                        )

    st.markdown("### Export trained model")
    buffer = io.BytesIO()
    joblib.dump(result.pipeline, buffer)
    st.download_button(
        f"Download {model_name} pipeline (.joblib)",
        buffer.getvalue(),
        file_name=f"mlbench_{model_name.replace(' ', '_').lower()}_pipeline.joblib",
        help="The full fitted sklearn pipeline (preprocessing + model). Loading it requires this "
             "project's mlbench package on the Python path.",
    )


# ---------------------------------------------------------------------------
# About / Help
# ---------------------------------------------------------------------------
def page_about() -> None:
    st.markdown('<div class="mlbench-title">About / Help</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **ML Bench — Machine Learning Experiment Studio** is a local, general-purpose ML workbench.
        It exists to answer one question honestly: *given this dataset and this target, which of these
        standard algorithms performs best on data the model has never seen?*

        ### How an experiment flows
        1. **Load data** — upload a CSV/Excel file, or use a built-in synthetic demo dataset.
        2. **Choose the target** — the column to predict. The app *suggests* classification vs
           regression but never forces it: a numeric target can still be a set of classes.
        3. **Train** — the data is split into train/test first. A scikit-learn `Pipeline`
           (imputation → one-hot encoding → scaling where needed → model) is fitted on the
           training split only, so nothing about the test split leaks into training.
        4. **Compare** — every metric comes from predictions on the held-out test split.
           Cross-validation, when enabled, runs separately on the training split.
        5. **Predict** — new records go through the *same fitted pipeline*. Encoders and scalers
           are never refitted on prediction data.
        """
    )
    st.markdown("### Algorithms")
    for name, text in explanations.MODEL_DESCRIPTIONS.items():
        st.markdown(f"- **{name}** — {text}")
    st.markdown("### Metric glossary")
    for name, text in explanations.METRIC_DESCRIPTIONS.items():
        st.markdown(f"- **{name}** — {text}")
    st.markdown(
        """
        ### Concepts worth knowing for a demo or viva
        - **Data leakage** — letting information from the test set (or the future) influence
          training. It inflates metrics and collapses on real data. Fitting preprocessing inside
          a Pipeline on training data only is the standard defence.
        - **Target leakage** — a feature that is itself derived from the target (for example a
          'result' column when predicting pass/fail). The Explorer warns when a feature tracks
          the target almost perfectly.
        - **Stratified split** — a train/test split that preserves each class's share, used for
          classification when every class has enough samples.
        - **Macro vs weighted averages** — macro treats every class equally; weighted accounts
          for class sizes. On imbalanced data they can tell very different stories.
        - **Binary vs multiclass ROC-AUC** — binary ROC uses one positive class; multiclass
          ROC-AUC here uses one-vs-rest averaging. They are related, not identical, quantities.

        ### Limitations
        - Very small datasets produce noisy metrics — the app warns below 100 usable rows.
        - Categorical columns with more than 50 distinct values are excluded (they usually encode
          identifiers or free text, not learnable categories).
        - Date/time columns are reduced to year/month/day/weekday features — no time-series
          modelling is attempted.
        - Demo datasets are synthetic: patterns in them were designed, so demo metrics demonstrate
          the pipeline, not real-world skill.

        ### Tech stack
        Python · Streamlit · pandas · NumPy · scikit-learn · XGBoost (optional) · Plotly ·
        OpenPyXL · Joblib. No paid APIs, no accounts, no GPU. See `VIVA_NOTES.md` in the project
        folder for a project explanation and viva preparation notes.
        """
    )
    st.warning(explanations.GENERALISATION_WARNING)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Apply any pending navigation BEFORE the sidebar radio widget is created —
# a widget's session-state key cannot be modified after it is instantiated.
_goto = st.session_state.pop("goto_page", None)
if _goto in PAGES:
    st.session_state.page = _goto

render_sidebar()
_PAGE_FUNCS = {
    "Dashboard": page_dashboard,
    "Dataset Explorer": page_explorer,
    "Model Training": page_training,
    "Model Comparison": page_comparison,
    "Predictions": page_predictions,
    "About / Help": page_about,
}

# Invalidate stale results the moment inputs change.
if st.session_state.outcome is not None and st.session_state.outcome_fingerprint != current_fingerprint():
    st.session_state.outcome = None
    st.session_state.train_config = None
    st.session_state.importance_cache = {}

_PAGE_FUNCS[st.session_state.page]()
