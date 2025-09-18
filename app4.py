import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv(
    r"/Users/hussseinsabbagh/Desktop/PROGRAMING/Machine learning/mobile_prices_2023 2.csv"
)

# ----------------------------
# Data Cleaning
# ----------------------------
df.drop(columns=['Number of Ratings', 'Date of Scraping'],
        inplace=True, errors='ignore')

df.drop(columns=['ROM/Storage', 'Front Camera', 'Battery', 'Processor'],
        inplace=True, errors='ignore')

df['Price in INR'] = df['Price in INR'].str.replace("₹", "").str.replace(",", "").astype(float)

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop("Price in INR", axis=1)
y = df["Price in INR"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Feature Engineering
# ----------------------------
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ----------------------------
# Train Model
# ----------------------------
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)

# ----------------------------
# Evaluation
# ----------------------------
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation")

print(f"R²: {r2:.4f}")

# ----------------------------
# Prediction Example
# ----------------------------
# Example input (replace with actual feature values from your dataset)
example_input = X.iloc[[0]]  # taking the first row as a test case

# Convert INR to USD
EXCHANGE_RATE = 83  # 1 USD ≈ 83 INR
prediction_inr = rf_pipeline.predict(example_input)[0]
prediction_usd = prediction_inr / EXCHANGE_RATE

print(f"💰 Predicted Price: ₹{prediction_inr:,.2f}  (≈ ${prediction_usd:,.2f})")
