
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix
)

st.set_page_config(
    page_title="Smart Student ML Analyzer",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Smart Student Performance Analyzer")
st.caption(
    "An ML mini-project for academic performance classification "
    "and model evaluation."
)

# Generate demonstration data
@st.cache_data
def create_data():
    rng = np.random.default_rng(42)
    n = 1500

    df = pd.DataFrame({
        "Attendance": rng.uniform(35, 100, n),
        "Study Hours": rng.uniform(0, 8, n),
        "Internal Marks": rng.uniform(20, 100, n),
        "Assignment Score": rng.uniform(20, 100, n),
        "Previous Marks": rng.uniform(20, 100, n),
        "Sleep Hours": rng.uniform(4, 10, n)
    })

    score = (
        0.22 * df["Attendance"]
        + 0.23 * df["Internal Marks"]
        + 0.20 * df["Assignment Score"]
        + 0.18 * df["Previous Marks"]
        + 0.12 * df["Study Hours"] / 8 * 100
        + 0.05 * df["Sleep Hours"] / 10 * 100
        + rng.normal(0, 9, n)
    ).clip(0, 100)

    df["Performance"] = pd.cut(
        score,
        bins=[-1, 50, 65, 80, 101],
        labels=[
            "At Risk",
            "Needs Support",
            "Good",
            "Excellent"
        ]
    ).astype(str)

    return df


df = create_data()
features = [
    "Attendance", "Study Hours", "Internal Marks",
    "Assignment Score", "Previous Marks", "Sleep Hours"
]

X = df[features]
y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    trained[name] = model

    results.append({
        "Model": name,
        "Accuracy (%)": round(
            accuracy_score(y_test, pred) * 100, 2
        ),
        "Macro F1 (%)": round(
            f1_score(y_test, pred, average="macro") * 100, 2
        )
    })

results_df = pd.DataFrame(results)

# Overview
c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Input Features", len(features))
c3.metric("Models Trained", len(models))

st.divider()

# Dataset explorer
st.subheader("📊 Dataset Explorer")
left, right = st.columns([1, 1])

with left:
    st.write("Sample records")
    st.dataframe(df.head(10), use_container_width=True)

with right:
    st.write("Performance category distribution")
    counts = df["Performance"].value_counts().rename_axis(
        "Category"
    ).reset_index(name="Students")
    fig = px.bar(
        counts, x="Category", y="Students",
        color="Category",
        title="Distribution of performance categories"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Model comparison
st.subheader("🤖 Model Comparison")
st.dataframe(results_df, hide_index=True, use_container_width=True)

fig = px.bar(
    results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy (%)", "Macro F1 (%)"],
        var_name="Metric",
        value_name="Score"
    ),
    x="Model", y="Score", color="Metric",
    barmode="group", range_y=[0, 100],
    title="Model performance on held-out test data"
)
st.plotly_chart(fig, use_container_width=True)

best_name = results_df.sort_values(
    "Macro F1 (%)", ascending=False
).iloc[0]["Model"]
st.info(
    f"Highest macro F1 on this test split: **{best_name}**. "
    "This does not guarantee the best performance on real-world data."
)

st.divider()


# Prediction section
st.subheader("🔎 Live Student Performance Prediction")
st.write(
    "Change the student details in the sidebar. "
    "The prediction and estimated probabilities update automatically."
)

with st.sidebar:
    st.header("Student Details")
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study = st.slider("Study hours per day", 0.0, 8.0, 3.0, 0.5)
    internal = st.slider("Internal marks (%)", 0, 100, 65)
    assignment = st.slider("Assignment score (%)", 0, 100, 70)
    previous = st.slider("Previous marks (%)", 0, 100, 60)
    sleep = st.slider("Sleep hours per day", 4.0, 10.0, 7.0, 0.5)

    selected_model = st.selectbox(
        "Select model", list(models.keys())
    )

input_df = pd.DataFrame([{
    "Attendance": attendance,
    "Study Hours": study,
    "Internal Marks": internal,
    "Assignment Score": assignment,
    "Previous Marks": previous,
    "Sleep Hours": sleep
}])

model = trained[selected_model]
prediction = model.predict(input_df)[0]

# Display prediction prominently
st.metric("Predicted Performance Category", prediction)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_df)[0]

    prob_df = pd.DataFrame({
        "Category": model.classes_,
        "Estimated probability (%)": probabilities * 100
    }).sort_values(
        "Estimated probability (%)", ascending=False
    )

    st.write("### Estimated probabilities for each category")

    fig_prob = px.bar(
        prob_df,
        x="Category",
        y="Estimated probability (%)",
        color="Category",
        range_y=[0, 100],
        text=prob_df["Estimated probability (%)"].round(1),
        title=f"Prediction using {selected_model}"
    )

    st.plotly_chart(fig_prob, use_container_width=True)

    st.caption(
        "Probabilities are model estimates from synthetic data, "
        "not guaranteed real-world outcomes."
    )

st.write("### Current student inputs")
st.dataframe(input_df, hide_index=True, use_container_width=True)

# Confusion matrix
st.subheader("📈 Confusion Matrix")

test_predictions = model.predict(X_test)
labels = model.classes_

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

# Use Plotly Express instead of figure_factory
cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues",
    labels={
        "x": "Predicted category",
        "y": "Actual category",
        "color": "Number of students"
    },
    title=f"{selected_model} — Test Set"
)

fig_cm.update_layout(
    xaxis_title="Predicted category",
    yaxis_title="Actual category"
)

st.plotly_chart(fig_cm, use_container_width=True)
