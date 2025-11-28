import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ✅ Load processed data
data_path = os.path.join("data", "processed")

X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))

print("✅ Processed data loaded successfully!")

# ✅ Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train.values.ravel())

print("✅ Model training complete!")

# ✅ Evaluate model
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# ✅ Save model
model_path = os.path.join("models", "model.pkl")
os.makedirs("models", exist_ok=True)
joblib.dump(model, model_path)

print(f"\n✅ Model saved successfully at: {model_path}")