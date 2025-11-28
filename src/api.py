from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import csv
from datetime import datetime

app = FastAPI(title="Predictive Maintenance API")

# ✅ Paths
MODEL_PATH = os.path.join("models", "model.pkl")
TRAIN_DATA_PATH = os.path.join("data", "processed", "X_train.csv")
LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "predictions.csv")

# ✅ Load trained model
model = joblib.load(MODEL_PATH)

# ✅ Load training data and fit scaler
train_df = pd.read_csv(TRAIN_DATA_PATH)
scaler = StandardScaler()
scaler.fit(train_df)

# ✅ Type encoding map
type_map = {"L": 0, "M": 1, "H": 2}

# ✅ Ensure logs directory and CSV exist
os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "type",
            "air_temperature_K",
            "process_temperature_K",
            "rotational_speed_rpm",
            "torque_Nm",
            "tool_wear_min",
            "prediction",
            "probability"
        ])


@app.get("/")
def home():
    return {"message": "✅ Predictive Maintenance API is running!"}


@app.post("/predict")
def predict(data: dict):
    """
    Expected keys in `data`:
    - "Type" (L/M/H)
    - "Air temperature [K]"
    - "Process temperature [K]"
    - "Rotational speed [rpm]"
    - "Torque [Nm]"
    - "Tool wear [min]"
    """

    # ✅ Convert Type to numeric
    if "Type" in data:
        data["Type"] = type_map.get(data["Type"], 0)

    # ✅ Build DataFrame from input
    df = pd.DataFrame([data])

    # ✅ Enforce same column order as training data
    expected_cols = train_df.columns.tolist()
    # This will raise a clear error if a column is missing
    try:
        df = df[expected_cols]
    except KeyError as e:
        return {
            "error": "Input features do not match training features.",
            "details": str(e),
            "expected_columns": expected_cols,
            "received_columns": df.columns.tolist(),
        }

    # ✅ Scale input
    df_scaled = scaler.transform(df)

    # ✅ Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    # ✅ Log the result
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            data.get("Type"),
            data.get("Air temperature [K]"),
            data.get("Process temperature [K]"),
            data.get("Rotational speed [rpm]"),
            data.get("Torque [Nm]"),
            data.get("Tool wear [min]"),
            int(prediction),
            float(probability),
        ])

    # ✅ Response
    return {
        "prediction": int(prediction),
        "probability": float(round(probability, 3)),
        "status": "⚠️ Failure Risk" if prediction == 1 else "✅ Normal Operation",
        "recommended_action": (
            "Schedule immediate inspection." if prediction == 1 else "Continue monitoring."
        )
    }