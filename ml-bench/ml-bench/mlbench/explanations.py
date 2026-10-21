"""Plain-English explanations of models and metrics.

Used by the Comparison and Predictions pages so a student can explain —
in a viva or a demo — what each algorithm does and what each number means.
"""

from __future__ import annotations

MODEL_DESCRIPTIONS: dict[str, str] = {
    "Logistic Regression": (
        "A linear model that draws a straight-line (or flat-plane) decision boundary between "
        "classes and outputs probabilities through the logistic (sigmoid) function. Fast, "
        "interpretable, and a strong baseline when classes are roughly linearly separable."
    ),
    "Linear Regression": (
        "Fits a straight-line relationship between the features and the target by minimising "
        "squared errors (ordinary least squares). Simple and interpretable; it assumes the "
        "underlying relationship is approximately linear."
    ),
    "Decision Tree": (
        "A single CART tree that repeatedly splits the data on the most informative feature "
        "threshold, creating an if-else flowchart that ends in a prediction. Very interpretable, "
        "but a single deep tree can overfit."
    ),
    "Random Forest": (
        "An ensemble of many decision trees, each trained on a random subset of rows and "
        "features. Their votes (classification) or average (regression) cancel out individual "
        "errors, usually giving strong accuracy with little tuning."
    ),
    "Support Vector Machine": (
        "Finds the boundary (hyperplane) that separates the classes with the widest possible "
        "margin, using kernel functions to handle curved boundaries. Needs feature scaling; "
        "powerful on small-to-medium datasets."
    ),
    "Support Vector Regression": (
        "The regression form of SVM: it fits a tube of acceptable error (epsilon) around the "
        "data and penalises points outside it, using kernels for non-linear relationships. "
        "Needs feature scaling."
    ),
    "XGBoost": (
        "Gradient-boosted trees: trees are added one at a time, each new tree correcting the "
        "errors of the ones before it, with regularisation to control overfitting. Frequently "
        "among the most accurate models on tabular data."
    ),
}

METRIC_DESCRIPTIONS: dict[str, str] = {
    "Accuracy": "Share of all test predictions that were exactly correct. Misleading when classes are imbalanced.",
    "Precision (macro)": "Of the records the model assigned to a class, how many truly belonged to it — averaged equally over classes. High precision means few false alarms.",
    "Recall (macro)": "Of the records that truly belonged to a class, how many the model found — averaged equally over classes. High recall means few missed cases.",
    "F1 (macro)": "Harmonic mean of macro precision and recall. Balances false alarms and missed cases, treating every class equally — a good default when classes matter equally.",
    "F1 (weighted)": "F1-score averaged with weights proportional to class sizes. Reflects overall performance dominated by the frequent classes.",
    "ROC-AUC": "Area under the ROC curve: how well the model's scores rank a random positive above a random negative. 1.0 is perfect ranking, 0.5 is no better than chance. For multiclass targets this app uses one-vs-rest macro averaging.",
    "MAE": "Mean Absolute Error — the average size of the prediction error in the target's own units. Easy to explain: 'on average we are off by about this much'.",
    "MSE": "Mean Squared Error — average of squared errors. Penalises large mistakes much more than small ones; its units are the target squared, so it is less intuitive.",
    "RMSE": "Root Mean Squared Error — the square root of MSE, back in the target's units. Like MAE but more sensitive to occasional large errors.",
    "R2": "R² (coefficient of determination) — share of the target's variance the model explains. 1.0 is perfect, 0 means the model is no better than predicting the mean, and it can be negative.",
}


def explain_result(result, task: str, chosen_metric: str) -> str:
    """One short paragraph summarising a trained model's held-out result."""
    if result.status != "ok":
        return f"{result.name} could not be trained: {result.error}"
    description = MODEL_DESCRIPTIONS.get(result.name, "")
    if task == "classification":
        acc = result.metrics.get("Accuracy")
        f1 = result.metrics.get("F1 (macro)")
        auc = result.metrics.get("ROC-AUC")
        parts = [f"On the held-out test set, {result.name} reached accuracy {acc:.3f} and macro F1 {f1:.3f}."]
        if auc is not None:
            parts.append(f"Its ROC-AUC was {auc:.3f}.")
    else:
        rmse = result.metrics.get("RMSE")
        mae = result.metrics.get("MAE")
        r2 = result.metrics.get("R2")
        parts = [
            f"On the held-out test set, {result.name} had RMSE {rmse:,.2f} and MAE {mae:,.2f} "
            f"(target units), explaining {r2:.1%} of the target's variance (R² = {r2:.3f})."
        ]
    if chosen_metric in result.metrics:
        parts.append(f"Under the currently selected metric ({chosen_metric}) its score is {result.metrics[chosen_metric]:,.4g}.")
    if description:
        parts.append(description)
    return " ".join(parts)


GENERALISATION_WARNING = (
    "Model results depend on data quality and on how representative the dataset is. "
    "Scores measured here — especially on synthetic demo data — do not guarantee "
    "performance on real-world data."
)
