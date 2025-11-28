import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ✅ Load dataset from correct path
data_path = os.path.join("data", "ai4i2020.csv")
df = pd.read_csv(data_path)

print("✅ Dataset loaded successfully!")
print(df.head())

# ✅ Drop unnecessary columns
drop_cols = ["UDI", "Product ID"]
df = df.drop(columns=drop_cols, errors="ignore")

# ✅ Remove failure-type columns (not available in real-time)
failure_cols = ["HDF", "OSF", "PWF", "RNF", "TWF"]
df = df.drop(columns=failure_cols, errors="ignore")

# ✅ Encode categorical column
if "Type" in df.columns:
    df["Type"] = df["Type"].astype("category").cat.codes

# ✅ Separate features and labels
if "Machine failure" not in df.columns:
    raise ValueError("❌ 'Machine failure' column not found in dataset.")

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# ✅ Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ✅ Save processed data
processed_path = os.path.join("data", "processed")
os.makedirs(processed_path, exist_ok=True)

X_train.to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

print("✅ Preprocessing complete!")
print("✅ Processed files saved in /data/processed/")