import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow.pyfunc

# Import your local Pydantic models
from src.api.pydantic_models import CustomerData, PredictionResponse

# 1. Set Path to the specific best model inside the root 'models' folder
# BASE_DIR points to 'Credit-Risk-Probability-Model'
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Points to 'Credit-Risk-Probability-Model/models/random_forest'
MODEL_PATH = BASE_DIR / "models" / "random_forest"

app = FastAPI(title="Credit Risk Prediction API")

# 2. Load Model on Startup
model = None
try:
    if MODEL_PATH.exists():
        # Loads the model artifacts from the clean root directory
        model = mlflow.pyfunc.load_model(str(MODEL_PATH))
        print(f"SUCCESS: System ready. Loaded model from: {MODEL_PATH}")
    else:
        print(f"WARNING: Model folder not found at {MODEL_PATH}. Prediction will fail.")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs for path issues.")

    # Convert Pydantic to DataFrame
    input_df = pd.DataFrame([customer.dict()])
    
    # Remove CustomerId if it exists in input (consistent with trainer logic)
    if 'CustomerId' in input_df.columns:
        input_df = input_df.drop(columns=['CustomerId'])

    try:
        # Note: If your model was saved as a Pipeline, scaling is handled.
        # If not, ensure input_df columns match the training format.
        prediction = model.predict(input_df)
        
        # Extract the probability or prediction value
        risk_val = float(prediction[0])
        
        return PredictionResponse(
            CustomerId=customer.CustomerId, 
            risk_probability=risk_val
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # This allows you to run the file directly with 'python main.py'
    uvicorn.run(app, host="0.0.0.0", port=8000)