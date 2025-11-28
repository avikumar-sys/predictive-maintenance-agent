from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# ✅ Resolve paths safely for Render
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "X_train.csv")

# ✅ Load model and data
try:
    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Required file missing: {e.filename}")
except Exception as e:
    raise RuntimeError(f"Error loading model or data: {str(e)}")

# ✅ Input schema (your model expects 4 sensors)
class InputData(BaseModel):
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float

@app.get("/")
def read_root():
    return {"message": "✅ Predictive Maintenance API is running!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # ✅ Model prediction (0 or 1)
        pred = model.predict(input_df)[0]

        # ✅ Probability (if model supports predict_proba)
        try:
            prob = float(model.predict_proba(input_df)[0][1])
        except Exception:
            prob = 1.0 if pred == 1 else 0.0

        # ✅ Human-readable status
        status = "⚠️ Failure Predicted" if pred == 1 else "✅ Normal Operation"

        # ✅ Recommended action
        recommended_action = (
            "Schedule maintenance within 24 hours."
            if pred == 1
            else "Machine is operating normally."
        )

        # ✅ Final response (matches your Streamlit dashboard)
        return {
            "prediction": int(pred),
            "probability": round(prob, 3),
            "status": status,
            "recommended_action": recommended_action,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")