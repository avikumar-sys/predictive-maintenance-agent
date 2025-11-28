from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# ✅ Load model (adjust path if needed)
model = joblib.load("models/model.pkl")  # or use full path if required

# ✅ Input schema
class InputData(BaseModel):
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float

@app.post("/predict")
def predict(data: InputData):
    try:
        raw = data.dict()

        # ✅ Encode Type column
        type_map = {"L": 0, "M": 1, "H": 2}
        encoded_type = type_map.get(raw["Type"], 0)

        # ✅ Build DataFrame with correct model feature names
        input_df = pd.DataFrame([{
            "Type": encoded_type,
            "Air temperature [K]": raw["Air_temperature_K"],
            "Process temperature [K]": raw["Process_temperature_K"],
            "Rotational speed [rpm]": raw["Rotational_speed_rpm"],
            "Torque [Nm]": raw["Torque_Nm"],
            "Tool wear [min]": raw["Tool_wear_min"]
        }])

        # ✅ Model prediction
        pred = model.predict(input_df)[0]

        # ✅ Probability
        try:
            prob = float(model.predict_proba(input_df)[0][1])
        except Exception:
            prob = 1.0 if pred == 1 else 0.0

        # ✅ Output
        status = "⚠️ Failure Predicted" if pred == 1 else "✅ Normal Operation"
        recommended_action = (
            "Schedule maintenance within 24 hours." if pred == 1
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