from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

app = FastAPI(title="Telco Churn Prediction API (Track A)")

MODEL_URI = "models:/Telco_Churn_Best_Model/Production"
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"[✓] Successfully loaded model from {MODEL_URI}")
except Exception as e:
    print(f"[!] Could not load Production model directly. Error: {e}")
    model = None

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str

@app.get("/")
def health_check():
    return {"status": "active", "model_uri": MODEL_URI}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if not model:
        raise HTTPException(status_code=500, detail="Production model not loaded.")
    
    input_df = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)