import os
import subprocess
import pandas as pd
from mlflow.tracking import MlflowClient

def check_latest_mlflow_metrics() -> dict:
    """Queries MLflow for the latest experiment runs and returns key metrics."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Telco_Churn_Experiment_Track_A")
    if not experiment:
        return {"error": "Experiment 'Telco_Churn_Experiment_Track_A' not found."}
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["attribute.start_time DESC"]
    )
    if not runs:
        return {"error": "No runs found in experiment."}
    
    latest_run = runs[0]
    return {
        "run_id": latest_run.info.run_id,
        "run_name": latest_run.data.tags.get("mlflow.runName", "Unnamed"),
        "metrics": latest_run.data.metrics
    }

def check_evidently_drift_status() -> dict:
    """Inspects reference and current datasets for distribution drift."""
    ref_path = "data/reference_data.csv"
    curr_path = "data/current_data.csv"
    
    if not os.path.exists(ref_path) or not os.path.exists(curr_path):
        return {"error": "Dataset files missing. Run data prep first."}
    
    ref_df = pd.read_csv(ref_path)
    curr_df = pd.read_csv(curr_path)
    
    mc_diff = abs(curr_df["MonthlyCharges"].mean() - ref_df["MonthlyCharges"].mean())
    drift_detected = mc_diff > 10.0
    
    return {
        "drift_detected": drift_detected,
        "monthly_charges_mean_shift": round(float(mc_diff), 2),
        "threshold": 10.0
    }

def trigger_pipeline_retrain() -> str:
    """Triggers data regeneration and model retraining."""
    print("[Agent Tool] Executing data preparation and retraining pipeline...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        subprocess.run(["uv", "run", "python", "src/data_prep.py"], env=env, check=True)
        train_res = subprocess.run(
            ["uv", "run", "python", "src/train.py"], 
            env=env,
            capture_output=True, 
            text=True, 
            encoding="utf-8",
            check=True
        )
        return f"[+] Retraining completed successfully.\n{train_res.stdout[-300:]}"
    except subprocess.CalledProcessError as e:
        return f"[!] Retraining failed: {e.stderr}"