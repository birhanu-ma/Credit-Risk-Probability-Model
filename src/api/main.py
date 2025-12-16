from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerData, PredictionResponse
import pandas as pd
import mlflow.pyfunc

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Credit Risk Prediction API")

# -------------------------------
# Load the best registered model from MLflow (Production stage)
# -------------------------------
MODEL_NAME = "Credit_Risk_Best_Model"
MODEL_STAGE = "Production"

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Loaded MLflow model '{MODEL_NAME}' from stage '{MODEL_STAGE}'")
except Exception as e:
    model = None
    print(f"Error loading MLflow model '{MODEL_NAME}': {e}")

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_risk(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="MLflow model not loaded. Check the model registry.")

    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([customer.dict()])

    # Predict probability of high-risk
    try:
        # Assuming the model outputs probability of high risk
        risk_prob = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(CustomerId=customer.CustomerId, risk_probability=float(risk_prob))
