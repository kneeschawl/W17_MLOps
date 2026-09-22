import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

if not os.path.exists("data/reference_data.csv"):
    from data_prep import prepare_and_inject_drift
    prepare_and_inject_drift()

df = pd.read_csv("data/reference_data.csv")

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_cols = ["Contract", "PaymentMethod"]
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numeric_cols),
    ("cat", cat_transformer, categorical_cols)
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_experiment("Telco_Churn_Experiment_Track_A")

models_to_train = [
    {
        "name": "Logistic_Regression_Baseline",
        "model": LogisticRegression(C=0.1, max_iter=1000),
        "params": {"C": 0.1, "model_type": "LogisticRegression"}
    },
    {
        "name": "Random_Forest_Tuned",
        "model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "params": {"n_estimators": 100, "max_depth": 6, "model_type": "RandomForest"}
    },
    {
        "name": "XGBoost_Optimized",
        "model": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42),
        "params": {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "model_type": "XGBoost"}
    }
]

best_run_id = None
best_f1 = -1.0
best_model_name = ""

for model_info in models_to_train:
    with mlflow.start_run(run_name=model_info["name"]) as run:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model_info["model"])])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        roc_auc = roc_auc_score(y_val, probs)

        mlflow.log_params(model_info["params"])
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        cm = confusion_matrix(y_val, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_info['name']}")
        os.makedirs("reports", exist_ok=True)
        cm_path = f"reports/cm_{model_info['name']}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Force cloudpickle serialization format to bypass skops type-checking
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle"
        )
        print(f"Run: {model_info['name']} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_run_id = run.info.run_id
            best_model_name = model_info["name"]

print(f"\n[★] Best Performing Model: {best_model_name} (Run ID: {best_run_id}) with F1: {best_f1:.4f}")

model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri, "Telco_Churn_Best_Model")

client = MlflowClient()
client.transition_model_version_stage(name="Telco_Churn_Best_Model", version=registered_model.version, stage="Staging")
client.transition_model_version_stage(name="Telco_Churn_Best_Model", version=registered_model.version, stage="Production")
print(f"[✓] Model 'Telco_Churn_Best_Model' Version {registered_model.version} promoted to 'Production'.")