"""
Train the insurance charge prediction model and save it for the Streamlit app.
Recreates the pipeline from Insurance_1.ipynb, with scaling fit ONLY on the
training set (the notebook fit it on the full dataset, which leaks test info).
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Load data
df = pd.read_csv("insurance.csv")

# 2. Clean / encode (same mapping as the notebook)
df["is_female"] = df["sex"].map({"male": 0, "female": 1})
df["is_smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df = pd.get_dummies(df, columns=["region"], drop_first=True)

# 3. Feature engineering: BMI category
df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 24.9, 29.9, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
)
df = pd.get_dummies(df, columns=["bmi_category"], drop_first=True)

# 4. Final feature set (same as notebook's final_df)
feature_cols = [
    "age", "is_female", "bmi", "children", "is_smoker",
    "region_southeast", "bmi_category_Obese",
]
X = df[feature_cols].astype(int if False else float)  # keep numeric dtype
y = df["charges"]

# 5. Train/test split BEFORE scaling (avoids leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 6. Scale continuous columns, fit on train only
scale_cols = ["age", "bmi", "children"]
scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# 7. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
n, p = X_test.shape[0], X_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"R2: {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")

# 9. Save model + scaler + feature order for the app to reuse
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print("Saved model.pkl, scaler.pkl, feature_cols.pkl")
