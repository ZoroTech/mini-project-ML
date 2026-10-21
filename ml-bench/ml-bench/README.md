# ML Bench — Machine Learning Experiment Studio

A general-purpose machine learning web application. Upload a structured dataset, choose a
target column, train several standard ML algorithms through a leakage-safe pipeline, compare
them on a held-out test set, and make predictions on new records — for both **classification**
and **regression** problems.

Everything runs **locally**: no paid APIs, no API keys, no database accounts, no GPU, and
uploaded data never leaves your machine.

---

## Quick start (Windows)

### 1. Install Python
Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/) and tick
**"Add python.exe to PATH"** during setup.

### 2. Open a terminal in the project folder
In File Explorer, open the `ml-bench` folder, click the address bar, type `powershell`, and
press Enter. (Or open PowerShell and `cd` to the folder.)

### 3. Create a virtual environment (recommended)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run this once and retry:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> If the `xgboost` line fails on your machine, delete that line from `requirements.txt` and
> install again. ML Bench works fully without it — the XGBoost models are simply hidden.

### 5. Run the app
```powershell
python -m streamlit run app.py
```
Your browser opens at **http://localhost:8501**. To stop the app, press `Ctrl+C` in the terminal.

---

## What you can do

| Page | What it does |
| --- | --- |
| **Dashboard** | Current experiment at a glance: dataset, target, task, models trained, best model under the default metric. |
| **Dataset Explorer** | Upload CSV/Excel, preview data, column types, missing values, duplicates, statistics, categorical counts; select the target and task. |
| **Model Training** | Choose models, test-split size, optional K-fold cross-validation; trains with a leakage-safe sklearn Pipeline. |
| **Model Comparison** | Sortable metrics table, comparison charts, training time, confusion matrix, ROC curves, classification report / residual plots, feature importance, best-model explanation. |
| **Predictions** | Single-record prediction form generated from the training features, class probabilities, batch prediction from CSV/Excel, downloadable predictions, downloadable trained pipeline (.joblib). |
| **About / Help** | How the pipeline works, algorithm and metric glossary, leakage concepts, limitations. |

### Algorithms
- **Classification:** Logistic Regression, Decision Tree (CART), Random Forest, SVM (SVC), XGBoost (if installed)
- **Regression:** Linear Regression, Decision Tree (CART), Random Forest, SVR, XGBoost (if installed)

### Metrics (all computed from real held-out predictions — never hardcoded)
- **Classification:** accuracy, precision, recall, F1 (macro & weighted), confusion matrix,
  ROC curve & ROC-AUC (binary and multiclass one-vs-rest), per-class report
- **Regression:** MAE, MSE, RMSE, R², actual-vs-predicted plot, residual plot

---

## Demo mode (no dataset needed)

Use the sidebar:

- **Load Demo Dataset** — loads one of three built-in **synthetic** datasets, generated locally
  with a fixed seed (fully reproducible):
  - **Demo A — Student Performance**: 600 students → predict the 4-class `Performance` target.
  - **Demo B — Customer Churn**: 800 telecom customers → predict `Churn` (Yes/No).
  - **Demo C — House Prices**: 700 properties → predict `House_Price` (regression).
- **Run Complete Demo** — one click: loads Demo A, selects its target, auto-detects the task,
  trains all classification models, computes real test-set metrics and opens the comparison page.
- **Reset Demo** — clears the dataset, results and selections back to the initial state.

Demo datasets can also be downloaded as CSV from the Dataset Explorer. They are synthetic:
results on them demonstrate the pipeline, not real-world performance.

---

## How correctness is protected

- **No data leakage:** the train/test split happens before anything is fitted. Imputers,
  encoders and scalers live inside an sklearn `Pipeline` fitted on training data only.
- **No refitting at prediction time:** predictions reuse the exact fitted pipeline.
- **Honest failures:** if one algorithm fails, its error is shown and the others still run.
  No metric is ever fabricated for a failed model.
- **Stale-result invalidation:** changing the dataset, target or task discards old results.
- **Leakage warnings:** the Explorer flags identifier columns, high-cardinality categoricals
  and features that track the target suspiciously closely.

## Project structure

```
ml-bench/
├── app.py                  # Streamlit UI (all six pages)
├── mlbench/
│   ├── demo_data.py        # the three reproducible synthetic datasets
│   ├── data_utils.py       # loading, validation, profiling, task detection, leakage checks
│   ├── pipeline.py         # preprocessing, model registry, training engine
│   ├── evaluation.py       # metrics, ROC data, feature importance, best-model logic
│   └── explanations.py     # plain-English model/metric explanations
├── tests/
│   └── smoke_test.py       # end-to-end pipeline tests (no UI needed)
├── .streamlit/config.toml  # dark theme
├── requirements.txt
├── README.md
└── VIVA_NOTES.md           # project explanation + viva preparation
```

## Run the tests

```powershell
python tests/smoke_test.py
```
This trains every model on all three demo datasets, verifies metrics come from held-out
predictions, checks the CSV-upload path, batch prediction, target switching and
missing-value handling — without opening the browser UI.

## Notes & limitations

- Maximum upload size: 25 MB; minimum 10 rows.
- Categorical columns with more than 50 distinct values are excluded as likely identifiers/text.
- Date/time columns become year/month/day/weekday features (no time-series modelling).
- Results depend on data quality and representativeness — see the warning in the app.
