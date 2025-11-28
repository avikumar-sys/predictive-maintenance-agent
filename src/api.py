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