"""Built-in synthetic demo datasets for ML Bench.

Every dataset is generated with a fixed random seed, so the same data is
produced on every run and on every machine. The targets are derived from
plausible relationships between the features (plus noise) — they are NOT
random labels — so trained models learn real, explainable patterns.

All three datasets are SYNTHETIC. Results obtained on them demonstrate the
pipeline, not real-world performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Demo A — Student Performance (multiclass classification)
# ---------------------------------------------------------------------------
def generate_student_performance(seed: int = DEFAULT_SEED, n_samples: int = 600) -> pd.DataFrame:
    """Synthetic student records with a 4-class Performance target.

    Performance is a weighted score of attendance, study hours, internal and
    assignment marks, previous marks and sleep, plus noise, binned into
    Poor / Average / Good / Excellent. A few missing values are injected on
    purpose so the imputation step of the pipeline is exercised.
    """
    rng = np.random.default_rng(seed)

    attendance = rng.normal(78, 12, n_samples).clip(35, 100)
    study_hours = rng.gamma(shape=2.2, scale=1.6, size=n_samples).clip(0.2, 12)
    internal_marks = rng.normal(32, 8, n_samples).clip(0, 50)
    assignment = rng.normal(72, 14, n_samples).clip(0, 100)
    previous_marks = rng.normal(65, 13, n_samples).clip(20, 100)
    sleep_hours = rng.normal(7.0, 1.2, n_samples).clip(3.5, 11)
    extracurricular = rng.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55])

    # Latent ability/performance score built from plausible drivers.
    score = (
        0.22 * (attendance / 100.0)
        + 0.20 * (study_hours / 12.0)
        + 0.22 * (internal_marks / 50.0)
        + 0.14 * (assignment / 100.0)
        + 0.16 * (previous_marks / 100.0)
        + 0.06 * (sleep_hours / 11.0)
        + np.where(extracurricular == "Yes", 0.01, 0.0)
        + rng.normal(0, 0.045, n_samples)
    )
    performance = np.select(
        [score < 0.45, score < 0.58, score < 0.70],
        ["Poor", "Average", "Good"],
        default="Excellent",
    )

    df = pd.DataFrame(
        {
            "Attendance_Percentage": attendance.round(1),
            "Study_Hours_Per_Day": study_hours.round(2),
            "Internal_Marks": internal_marks.round(1),
            "Assignment_Score": assignment.round(1),
            "Previous_Marks": previous_marks.round(1),
            "Sleep_Hours": sleep_hours.round(1),
            "Extracurricular": extracurricular,
            "Performance": performance,
        }
    )

    # Inject a small number of missing values so imputation is demonstrable.
    for col, frac in [("Assignment_Score", 0.03), ("Sleep_Hours", 0.02), ("Internal_Marks", 0.015)]:
        mask = rng.random(n_samples) < frac
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Demo B — Customer Churn (binary classification)
# ---------------------------------------------------------------------------
def generate_customer_churn(seed: int = DEFAULT_SEED, n_samples: int = 800) -> pd.DataFrame:
    """Synthetic telecom customers with a binary Churn target.

    Churn probability follows a logistic model of plausible drivers:
    month-to-month contracts, fibre internet, high monthly charges, many
    support calls and short tenure raise churn risk; long contracts and
    online security lower it. Labels are sampled from that probability, so
    the signal is learnable but not perfectly separable.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n_samples)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20]
    )
    internet = rng.choice(["Fiber optic", "DSL", "No internet"], size=n_samples, p=[0.42, 0.38, 0.20])
    online_security = np.where(internet == "No internet", "No", rng.choice(["Yes", "No"], n_samples, p=[0.45, 0.55]))
    payment = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n_samples,
        p=[0.34, 0.26, 0.24, 0.16],
    )
    senior = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    support_calls = rng.poisson(1.4, n_samples).clip(0, 9)

    base_charge = np.select(
        [internet == "Fiber optic", internet == "DSL"],
        [78.0, 52.0],
        default=22.0,
    )
    monthly_charges = (base_charge + rng.normal(0, 11, n_samples) + np.where(online_security == "Yes", 6, 0)).clip(15, 135)
    total_charges = monthly_charges * tenure * rng.uniform(0.92, 1.0, n_samples)

    logit = (
        -1.55
        + np.where(contract == "Month-to-month", 1.15, 0.0)
        + np.where(contract == "Two year", -0.95, 0.0)
        + np.where(internet == "Fiber optic", 0.55, 0.0)
        + 0.022 * (monthly_charges - 65)
        - 0.028 * tenure
        + 0.34 * support_calls
        + np.where(online_security == "Yes", -0.45, 0.0)
        + np.where(payment == "Electronic check", 0.35, 0.0)
        + 0.18 * senior
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.random(n_samples) < churn_prob, "Yes", "No")

    return pd.DataFrame(
        {
            "Tenure_Months": tenure,
            "Monthly_Charges": monthly_charges.round(2),
            "Total_Charges": total_charges.round(2),
            "Contract_Type": contract,
            "Internet_Service": internet,
            "Online_Security": online_security,
            "Payment_Method": payment,
            "Senior_Citizen": senior,
            "Support_Calls": support_calls,
            "Churn": churn,
        }
    )


# ---------------------------------------------------------------------------
# Demo C — House Prices (regression)
# ---------------------------------------------------------------------------
def generate_house_prices(seed: int = DEFAULT_SEED, n_samples: int = 700) -> pd.DataFrame:
    """Synthetic property records with a numerical House_Price target (INR).

    Price is built from a per-square-foot rate that depends on location,
    plus contributions from bedrooms/bathrooms, minus depreciation for age
    and distance from the city centre, with multiplicative noise — a
    plausible, learnable relationship rather than a made-up number.
    """
    rng = np.random.default_rng(seed)

    area = rng.normal(1450, 520, n_samples).clip(350, 3600)
    bedrooms = np.clip(np.round(area / 480 + rng.normal(0, 0.7, n_samples)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n_samples), 1, 5).astype(int)
    age = rng.integers(0, 41, n_samples)
    location = rng.choice(["Premium", "Standard", "Budget"], size=n_samples, p=[0.3, 0.45, 0.25])
    distance = rng.gamma(shape=2.0, scale=6.0, size=n_samples).clip(0.5, 45)
    parking = rng.choice(["Yes", "No"], size=n_samples, p=[0.62, 0.38])

    rate_per_sqft = np.select(
        [location == "Premium", location == "Standard"],
        [12500.0, 8200.0],
        default=4800.0,
    )
    price = (
        area * rate_per_sqft
        + bedrooms * 145_000
        + bathrooms * 95_000
        - age * 38_000
        - distance * 42_000
        + np.where(parking == "Yes", 175_000, 0)
    )
    price = price * rng.normal(1.0, 0.09, n_samples)  # market noise
    price = np.clip(price, 900_000, None)

    return pd.DataFrame(
        {
            "Area_SqFt": area.round(0),
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Property_Age_Years": age,
            "Location_Category": location,
            "Distance_To_City_Km": distance.round(1),
            "Has_Parking": parking,
            "House_Price": price.round(0),
        }
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DEMO_DATASETS = {
    "student": {
        "title": "Demo A — Student Performance (Classification)",
        "generator": generate_student_performance,
        "target": "Performance",
        "task": "classification",
        "description": (
            "Synthetic records for 600 students. Numerical features (attendance, study hours, "
            "internal marks, assignment score, previous marks, sleep hours) and one categorical "
            "feature (Extracurricular) are used to predict the 4-class target **Performance** "
            "(Poor / Average / Good / Excellent). A few missing values are included on purpose "
            "so median imputation is exercised."
        ),
    },
    "churn": {
        "title": "Demo B — Customer Churn (Classification)",
        "generator": generate_customer_churn,
        "target": "Churn",
        "task": "classification",
        "description": (
            "Synthetic records for 800 telecom customers. Churn risk in the generator rises with "
            "month-to-month contracts, fibre internet, higher monthly charges, more support calls "
            "and shorter tenure — so models can learn a real pattern. Target **Churn** is Yes/No."
        ),
    },
    "house": {
        "title": "Demo C — House Prices (Regression)",
        "generator": generate_house_prices,
        "target": "House_Price",
        "task": "regression",
        "description": (
            "Synthetic records for 700 properties. **House_Price** (INR) is generated from a "
            "location-dependent rate per square foot plus bedroom/bathroom contributions, minus "
            "age and distance effects, with market noise — a plausible regression problem."
        ),
    },
}


def get_demo_dataset(key: str, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return a freshly generated copy of a demo dataset (reproducible)."""
    return DEMO_DATASETS[key]["generator"](seed=seed)
