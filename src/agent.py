from agent_tools import (
    check_latest_mlflow_metrics, 
    check_evidently_drift_status, 
    trigger_pipeline_retrain
)

class MLOpsAgent:
    def __init__(self):
        print("[★] Initialized Autonomous MLOps AI Agent")

    def run_health_audit(self):
        """Audits model performance, drift status, and triggers retraining if necessary."""
        print("\n=== Step 1: Querying MLflow Metrics ===")
        metrics_info = check_latest_mlflow_metrics()
        print(f"Latest Run ID: {metrics_info.get('run_id')}")
        print(f"Metrics: {metrics_info.get('metrics')}")

        print("\n=== Step 2: Inspecting Feature Drift ===")
        drift_info = check_evidently_drift_status()
        print(f"Drift Shift: {drift_info.get('monthly_charges_mean_shift')}")
        print(f"Drift Detected: {drift_info.get('drift_detected')}")

        print("\n=== Step 3: Decision Engine ===")
        if drift_info.get("drift_detected"):
            print("[!] Critical Drift Detected! Agent initializing automated retrain...")
            result = trigger_pipeline_retrain()
            print(result)
        else:
            print("[✓] Pipeline healthy. No remediation required.")

if __name__ == "__main__":
    agent = MLOpsAgent()
    agent.run_health_audit()