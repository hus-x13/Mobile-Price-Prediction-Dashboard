import streamlit as st
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

# --- Page Configuration ---
st.set_page_config(
    page_title="Mobile Price Prediction Dashboard",
    layout="wide"
)

# --- Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "About Us"])

# ========================
#  Dashboard Page
# ========================
if page == "Dashboard":
    st.title("📱 Mobile Price Prediction Dashboard")
    st.markdown("Predict mobile prices using Random Forest regression")

    # --- File Upload ---
    uploaded_file = st.file_uploader(
        "📂 Upload your mobile prices CSV", type=["csv"])

    if uploaded_file is None:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

    # --- Load Data ---
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()  # Clean column names
        return df

    df = load_data(uploaded_file)

    # --- Detect Price Column ---
    price_col_candidates = [
        col for col in df.columns if 'price' in col.lower()]
    if not price_col_candidates:
        st.error("No column containing 'price' found in CSV!")
        st.stop()
    price_col = price_col_candidates[0]

    # --- Data Cleaning ---
    drop_cols = ['Number of Ratings', 'Date of Scraping',
                 'ROM/Storage', 'Front Camera', 'Battery', 'Processor']
    df.drop(columns=[col for col in drop_cols if col in df.columns],
            inplace=True, errors='ignore')

    # Convert price to numeric
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace("₹", "$")
        .str.replace(",", "")
        .astype(float)
    )

    # --- Features & Target ---
    X = df.drop(price_col, axis=1)
    y = df[price_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

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

    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    # --- Prediction Section ---
    st.subheader("💡 Predict a Mobile Price")
    st.markdown(
        "Enter the features of the mobile below to see the predicted price:")

    input_cols = st.columns(len(X.columns) // 2 + 1)
    input_data = {}
    for i, col in enumerate(X.columns):
        container = input_cols[i % len(input_cols)]
        if col in categorical_cols:
            input_data[col] = container.selectbox(f"{col}", df[col].unique())
        else:
            input_data[col] = container.number_input(
                f"{col}", value=float(df[col].median()))

    if st.button("Predict Price"):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = rf_pipeline.predict(input_df)[0]
            st.success(f"💰 Predicted Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")

    # --- Raw Data Preview ---
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head())

    # --- Evaluation & Heatmap Side by Side ---
    st.subheader("📊 Model Evaluation & Correlation Heatmap")
    eval_col, heatmap_col = st.columns(2)

    with eval_col:

        st.metric("R² Score", f"{r2:.4f}")

    with heatmap_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True),
                    annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("Developed with ❤️ using Python & Streamlit")

# ========================
#  About Us Page
# ========================
elif page == "About Us":
    st.title("ℹ️ About Us")
    st.markdown("""
    ### Mobile Price Prediction Dashboard  
    This app helps you **predict mobile prices** using machine learning 
    (Random Forest Regression).
    
    #### 👨‍💻 Developer
    - **Name:** Hussein Sabbagh  
    - **Passion:** Data Science & Machine Learning  
    - **Tools Used:** Python, Streamlit, Scikit-learn, Pandas, Matplotlib , Seaborn , NumPy  
    
    #### 📌 Features
    - Upload your own mobile prices dataset (CSV)
    - Train a Random Forest model automatically
    - Predict mobile prices with custom inputs
    - View evaluation metrics (R²)
    - Explore data with correlation heatmaps
    
    #### ❤️ Motivation
    Making machine learning **simple & interactive** for everyone.
    """)
