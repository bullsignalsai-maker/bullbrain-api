from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# ===============================
# Root endpoint
# ===============================
@app.get("/")
def home():
    return {"message": "BullBrain AI API is running 🚀"}


# ===============================
# Pydantic model for input data
# ===============================
class StockData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float


# ===============================
# Prediction endpoint
# ===============================
@app.post("/predict")
def predict(data: StockData):
    model_path = "models/latest.pkl"

    if not os.path.exists(model_path):
        return {"error": "Model not found on server"}

    # Load model
    model = joblib.load(model_path)

    # Prepare input
    X = np.array([[data.open, data.high, data.low, data.close, data.volume]])

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()

    label = "BUY" if pred == 1 else "SELL"

    return {
        "prediction": label,
        "confidence": max(proba),
        "probabilities": {
            "SELL": proba[0],
            "BUY": proba[1]
        }
    }
