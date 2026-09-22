import os
import pandas as pd
import mlflow

from evidently import Report
from evidently.presets import DataDriftPreset

def run_evidently_drift_analysis():
    ref_df = pd.read_csv("data/reference_data.csv")
    curr_df = pd.read_csv("data/current_data.csv")

    feature_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod"]
    
    # Ensure numerical columns are properly formatted
    for df in [ref_df, curr_df]:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    report_def = Report([
        DataDriftPreset()
    ])

    print("[+] Running Evidently AI Drift Detection...")
    # report.run(...) returns the evaluation snapshot object
    eval_result = report_def.run(
        reference_data=ref_df[feature_cols],
        current_data=curr_df[feature_cols]
    )

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    
    # Save HTML output from the evaluation result object
    eval_result.save_html(report_path)
    print(f"[✓] Evidently Drift Report saved to {report_path}")

    mlflow.set_experiment("Telco_Churn_Experiment_Track_A")
    with mlflow.start_run(run_name="Evidently_Monitoring_Run"):
        mlflow.log_artifact(report_path)
        print("[✓] Drift report artifact logged to MLflow.")

if __name__ == "__main__":
    run_evidently_drift_analysis()