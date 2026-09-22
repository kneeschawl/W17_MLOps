# W17 MLOps Pipeline: Standard MLOps & Agentic AI Autonomous Monitoring

This repository implements an end-to-end MLOps architecture for predicting customer churn on the **Telco Churn** dataset. It is divided into two distinct feature branches:
- **`track-a-mlops`**: Standard MLOps pipeline covering environment management, dataset preparation with synthetic drift, experiment tracking, model registry promotion, dataset drift analysis, and model serving via FastAPI.
- **`track-b-agentic`**: Agentic AI extension where an autonomous assistant continuously audits system metrics, evaluates feature distribution drift, and automatically triggers pipeline retraining upon detecting drift threshold violations.

---

## 📄 Documentation & Assessment Requirements

### a. Environment & Reproducibility (`uv`)

#### Dependency & Version Issues Solved
Using `uv` eliminates several common Python environment failure modes in this project:
1. **Transitive Dependency Drift & Incompatibilities**: Libraries like `evidently`, `mlflow`, `xgboost`, and `scikit-learn` frequently suffer from overlapping or conflicting sub-dependencies (e.g., `pydantic` or `numpy` version mismatches). `uv` enforces strict, reproducible resolution across all sub-dependencies via `uv.lock`.
2. **System-Level & Windows Encoding/Subprocess Conflicts**: Multi-platform C-extensions required by `xgboost` and `scikit-learn` are deterministically linked, ensuring consistent behavior across OS environments without missing C++ runtime binaries.

#### One-Command Reproduction Path
To verify that the pipeline executes cleanly from a fresh clone:

```powershell
git clone https://github.com/kneeschawl/W17_MLOps.git
cd W17_MLOPS
uv sync
uv run python src/agent.py
```

---

### b. Experiment Tracking Strategy (MLflow)

#### Experimental Design & Metrics
To address churn prediction on an imbalanced target, we varied model architecture and hyperparameter configurations across three distinct algorithms:
1. **`Logistic_Regression_Baseline`**: L2 regularization, `max_iter=1000`.
2. **`Random_Forest_Tuned`**: 100 decision trees, constrained `max_depth=5` to prevent overfitting.
3. **`XGBoost_Optimized`**: Gradient boosted trees, 100 estimators, `max_depth=3`, `learning_rate=0.1`.

Across all runs, we tracked **F1-Score** and **ROC-AUC** as primary decision metrics, alongside **Accuracy**, **Precision**, and **Recall**. Accuracy alone is insufficient because customer churn exhibits class imbalance; a model predicting "No Churn" indiscriminately would yield artificially high accuracy while failing to detect churning customers.

#### MLflow Run Comparison & Registration Justification

Below is the side-by-side performance evaluation recorded in MLflow:

| Run / Model Name | F1-Score | ROC-AUC | Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Logistic_Regression_Baseline`** | **0.5811** | **0.8324** | **0.7891** | **0.6216** | **0.5267** |
| `Random_Forest_Tuned` | 0.5800 | 0.8306 | 0.7875 | 0.6180 | 0.5241 |
| `XGBoost_Optimized` | 0.5702 | 0.8324 | 0.7812 | 0.6015 | 0.5180 |

#### Decision Justification & Model Promotion
- **Model Selected**: `Logistic_Regression_Baseline` (Run ID: `227d7f0a3ee2439faa1aeab795f0fcc0`).
- **Justification**: `Logistic_Regression_Baseline` achieved the highest F1-Score (**0.5811**) and tied for the highest ROC-AUC (**0.8324**). Although `Random_Forest_Tuned` performed competitively (F1: 0.5800), Logistic Regression provides superior inference speed, lower latency, and linear interpretability without sacrificing predictive power.
- **Model Registry Promotion**: `Logistic_Regression_Baseline` was registered in MLflow under `Telco_Churn_Best_Model` and automatically promoted to the **`Production`** stage.

---

### c. Monitoring & Drift Strategy (Evidently AI)

#### Reference vs. Current Datasets
- **Reference Dataset (`data/reference_data.csv`)**: A 70% stratified split (4,930 rows) of the baseline historical data used for model training and evaluation.
- **Current Dataset (`data/current_data.csv`)**: A 30% split (2,113 rows) injected with synthetic feature shift on `MonthlyCharges` (mean increased by **+24.34** units) to simulate macro-economic billing adjustments or shifting customer demographics.

#### Monitored Metrics & Report Findings
Using Evidently AI (`DataDriftPreset`), we monitored statistical distribution shifts across numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`). The generated report (`reports/drift_report.html`) confirmed significant feature drift in `MonthlyCharges` (mean shift of **24.34** exceeds the threshold limit of **10.0**).

#### Remediation & Automated Action
When feature drift crosses the threshold:
1. **Track A (Manual / Static)**: Evidently generates an HTML report (`reports/drift_report.html`) alerting ML engineers to trigger retraining manually.
2. **Track B (Agentic AI Autonomous Remediation)**: The AI Agent (`src/agent.py`) intercepts the drift signal (`drift_detected: True`), triggers `data_prep.py` and `train.py`, retrains all models on updated data, and promotes **Version 2** of `Telco_Churn_Best_Model` directly to **`Production`**.

---

### d. Orchestration (Bonus)
*(Note: Airflow orchestration was not implemented for this submission; pipeline orchestration is handled autonomously via the Track B Agentic decision engine).*

---

## 🛠️ Repository Structure

```
W17_MLOPS/
├── .gitignore               # Configured to track report artifacts while ignoring raw data & .venv
├── pyproject.toml           # uv project configuration
├── uv.lock                  # Lockfile for reproducible dependencies
├── data/                    # Local storage (excluded from git)
│   ├── raw_telco_churn.csv
│   ├── reference_data.csv
│   └── current_data.csv
├── reports/                 # Output artifacts retained for assessment
│   ├── cm_Logistic_Regression_Baseline.png
│   ├── cm_Random_Forest_Tuned.png
│   ├── cm_XGBoost_Optimized.png
│   └── drift_report.html
└── src/
    ├── data_prep.py         # Stratified dataset split & synthetic drift injection
    ├── train.py             # MLflow training, evaluation, & production model registration
    ├── monitor.py           # Evidently AI drift monitoring report generator
    ├── serve.py             # FastAPI REST endpoint for production inference
    ├── agent_tools.py       # Custom tools wrapping MLflow, Evidently, & retraining
    └── agent.py             # Autonomous MLOps AI decision engine
```

---

## 🏃 Workflow Execution Commands

### Run Track A Pipeline
```powershell
# 1. Prepare data and inject drift
uv run python src/data_prep.py

# 2. Train models, log to MLflow, and promote best model
uv run python src/train.py

# 3. Generate Evidently drift report
uv run python src/monitor.py

# 4. Start prediction service API
uv run python src/serve.py
```

### Test Serving API
In a separate terminal window:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"tenure": 2, "MonthlyCharges": 85.5, "TotalCharges": 171.0, "Contract": "Month-to-month", "PaymentMethod": "Electronic check"}'
```

### Run Track B Autonomous MLOps Agent
```powershell
uv run python src/agent.py
```
