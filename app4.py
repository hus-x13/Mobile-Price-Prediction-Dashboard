import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Mobile Price Prediction", layout="wide")

# ----------------------------
# Sidebar Navigation
# ----------------------------
page = st.sidebar.selectbox("Navigate", ["Prediction", "About Me"])

# ----------------------------
# About Me Page
# ----------------------------
if page == "About Me":
    st.title("👤 About Me")
    st.write("""
    Hello! I'm **Hussein Sabbagh**, a Machine Learning enthusiast.
    
    This dashboard allows users to:
    _ This is an website that predict the price of mobile phones.
    - Upload their own CSV files containing mobile phone features.
    - Train a Random Forest model to predict mobile prices USD.
    - Evaluate the model performance with R² score.
    - See predictions directly on the web without using the terminal.
    """)
# ----------------------------
# Prediction Page
# ----------------------------
else:
    st.title("📱 Mobile Price Prediction Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data")
        st.dataframe(df.head(10))

        # ----------------------------
        # Data Cleaning
        # ----------------------------
        drop_cols = ['ROM/Storage', 'Front Camera', 'Battery',
                     'Processor', 'Number of Ratings', 'Date of Scraping']
        df.drop(columns=[c for c in drop_cols if c in df.columns],
                inplace=True, errors='ignore')

        if 'Price in INR' not in df.columns:
            st.error("❌ The CSV must contain a 'Price in INR' column.")
        else:
            df['Price in INR'] = df['Price in INR'].astype(
                str).str.replace("₹", "").str.replace(",", "").astype(float)

            # ----------------------------
            # Detect important columns dynamically
            # ----------------------------
            phone_col = [
                col for col in df.columns if "Phone" in col or "phone" in col][0]
            rating_col = [
                col for col in df.columns if "Rating" in col or "rating" in col][0]
            ram_col = [
                col for col in df.columns if "RAM" in col or "ram" in col][0]
            camera_col = [
                col for col in df.columns if "Camera" in col or "camera" in col][0]

            # Features & Target
            X = df.drop("Price in INR", axis=1)
            y = df["Price in INR"]

            # ----------------------------
            # Train Model on log(price)
            # ----------------------------
            y_log = np.log1p(y)  # log(1 + price)

            categorical_cols = X.select_dtypes(
                include=["object", "category"]).columns
            numeric_cols = X.select_dtypes(
                include=["int64", "float64"]).columns

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
                ]
            )

            rf_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42))
            ])
            rf_pipeline.fit(X, y_log)

            # ----------------------------
            # Model Evaluation
            # ----------------------------
            y_pred_log = rf_pipeline.predict(X)
            y_pred = np.expm1(y_pred_log)
            r2 = r2_score(y, y_pred)
            st.subheader("📊 Model Evaluation")
            st.metric("R² Score", f"{r2:.4f}")

            # ----------------------------
            # User Inputs
            # ----------------------------
            st.subheader("💡 Enter Phone Features for Prediction")

            selected_phone = st.selectbox(
                "Select Phone", df[phone_col].unique())
            rating = st.slider(
                f"Rate this phone ({rating_col})", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
            ram = st.selectbox("Select RAM", df[ram_col].unique())
            camera = st.selectbox(
                "Select Back & Rear Camera", df[camera_col].unique())

            # Build input dataframe
            input_data = pd.DataFrame({
                phone_col: [selected_phone],
                rating_col: [rating],
                ram_col: [ram],
                camera_col: [camera]
            })

            # ----------------------------
            # Make Prediction
            # ----------------------------
            predicted_log_price = rf_pipeline.predict(input_data)[0]
            predicted_price_inr = np.expm1(predicted_log_price)
            EXCHANGE_RATE = 83  # 1 USD ≈ 83 INR
            predicted_price_usd = predicted_price_inr / EXCHANGE_RATE

            st.success(f"💰 Predicted Price: ${predicted_price_usd:,.2f}")
    else:
        st.info("📂 Upload a CSV file to get started.")
