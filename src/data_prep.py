import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw_telco_churn.csv"
REF_PATH = "data/reference_data.csv"
CURR_PATH = "data/current_data.csv"

def generate_mock_telco_data(num_samples: int = 7000) -> pd.DataFrame:
    np.random.seed(42)
    tenure = np.random.randint(1, 72, size=num_samples)
    monthly_charges = np.random.uniform(18.25, 118.75, size=num_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 10, size=num_samples)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=num_samples)
    
    churn_prob = 1 / (1 + np.exp(-(-2.0 + 0.02 * monthly_charges - 0.03 * tenure + (contract == "Month-to-month") * 1.2)))
    churn = (np.random.rand(num_samples) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "Churn": churn
    })

def clean_and_format_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Map target Churn from ("Yes"/"No") to (1/0) if needed
    if df["Churn"].dtype == object or isinstance(df["Churn"].iloc[0], str):
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    
    # 2. Clean TotalCharges (Kaggle raw dataset has spaces " " for new customers)
    if "TotalCharges" in df.columns and df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    # Keep only relevant features for Track A
    features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "Churn"]
    return df[features]

def prepare_and_inject_drift():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"[+] Generating synthetic Telco Churn dataset at {DATA_PATH}...")
        df = generate_mock_telco_data()
        df.to_csv(DATA_PATH, index=False)
    else:
        print(f"[+] Loading raw dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)

    df = clean_and_format_data(df)

    ref_df, curr_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Churn"])
    curr_df = curr_df.copy()

    # Synthetic Drift Injection
    print("[+] Injecting synthetic drift into Current dataset...")
    curr_df["MonthlyCharges"] = curr_df["MonthlyCharges"] + np.random.uniform(20.0, 30.0, size=len(curr_df))
    curr_df["Contract"] = np.random.choice(["Month-to-month", "One year", "Two year"], size=len(curr_df), p=[0.85, 0.10, 0.05])

    # Flip 10% of binary labels (0 -> 1, 1 -> 0)
    flip_idx = curr_df.sample(frac=0.1, random_state=42).index
    curr_df.loc[flip_idx, "Churn"] = 1 - curr_df.loc[flip_idx, "Churn"]

    ref_df.to_csv(REF_PATH, index=False)
    curr_df.to_csv(CURR_PATH, index=False)
    print(f"[✓] Reference data saved ({len(ref_df)} rows), Current data saved ({len(curr_df)} rows)")

if __name__ == "__main__":
    prepare_and_inject_drift()