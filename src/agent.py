import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# ✅ Load the trained model
model_path = os.path.join("models", "model.pkl")
model = joblib.load(model_path)

# ✅ Load training data for scaler
train_data_path = os.path.join("data", "processed", "X_train.csv")
train_df = pd.read_csv(train_data_path)

# ✅ Fit scaler on training data
scaler = StandardScaler()
scaler.fit(train_df)

def predict_failure(input_data: dict):
    # Convert categorical 'Type' to numeric
    type_map = {"L": 0, "M": 1, "H": 2}
    input_data["Type"] = type_map.get(input_data["Type"], 0)

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Scale input
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    # Generate recommendation
    if prediction == 1:
        status = "⚠️ High risk of machine failure"
        action = "Schedule immediate inspection or maintenance."
    else:
        status = "✅ Machine operating normally"
        action = "Continue monitoring."

    return {
        "prediction": int(prediction),
        "probability": round(probability, 3),
        "status": status,
        "recommended_action": action
    }

# ✅ MAIN BLOCK (This prints output)
if __name__ == "__main__":
    sample_input = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0
    }

    result = predict_failure(sample_input)
    print("\n✅ Prediction Result:")
    print(result)