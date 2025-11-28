from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ✅ MUST BE AT THE TOP BEFORE ANY ROUTES
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

# ✅ Input schema (matches your Streamlit UI)
class InputData(BaseModel):
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float

@app.get("/")
def read_root():
    return {"message": "✅ Predictive Maintenance API is running!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # ✅ Convert API input into model's expected column names
        raw = data.dict()

        input_df = pd.DataFrame([{
            "Type": raw["Type"],
            "Air temperature [K]": raw["Air_temperature_K"],
            "Process temperature [K]": raw["Process_temperature_K"],
            "Rotational speed [rpm]": raw["Rotational_speed_rpm"],
            "Torque [Nm]": raw["Torque_Nm"],
            "Tool wear [min]": raw["Tool_wear_min"]
        }])

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

        return {
            "prediction": int(pred),
            "probability": round(prob, 3),
            "status": status,
            "recommended_action": recommended_action,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")