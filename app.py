import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved models
@st.cache_data
def load_model(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

# Load Classification & Regression Models
rf_classifier = load_model("random_forest.pkl")
xgb_classifier = load_model("xgboost.pkl")
rf_regressor = load_model("random_forest_regressor.pkl")
xgb_regressor = load_model("xgboost_regressor.pkl")

# Streamlit UI
st.title("🔮 Model Prediction & Data Visualization App")
st.write("This app supports **Classification & Regression** along with Data Analysis & Visualization.")

# Sidebar options
option = st.sidebar.radio("Choose an option:", ["Upload Data", "Model Prediction"])

### 📂 DATA UPLOAD & EXPLORATION ###
if option == "Upload Data":
    st.header("📊 Upload & Explore Data")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display Data
        st.subheader("🔍 Preview of Data")
        st.write(df.head())

        # Data Summary
        st.subheader("📊 Data Summary")
        st.write(df.describe())

        # Check for missing values
        st.subheader("❗ Missing Values")
        st.write(df.isnull().sum())

        # Graphs Section
        st.subheader("📈 Data Visualizations")

        # Numeric Feature Selection
        numeric_columns = df.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            feature = st.selectbox("Select a feature to visualize:", numeric_columns)

            # Histogram
            st.subheader("📊 Histogram")
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, bins=30, ax=ax)
            st.pyplot(fig)

            # Box Plot
            st.subheader("📦 Box Plot")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[feature], ax=ax)
            st.pyplot(fig)

            # Scatter Plot
            if len(numeric_columns) > 1:
                st.subheader("📌 Scatter Plot")
                second_feature = st.selectbox("Choose a second feature for scatter plot:", numeric_columns)
                
                if feature != second_feature:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=df[feature], y=df[second_feature], ax=ax)
                    st.pyplot(fig)

        # Correlation Heatmap
        if len(numeric_columns) > 1:
            st.subheader("📊 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical features for correlation heatmap.")

### 🔮 MODEL PREDICTION ###
elif option == "Model Prediction":
    st.header("🔮 Model Prediction")

    # Sidebar selection
    model_type = st.sidebar.selectbox("Choose Model Type:", ["Classification", "Regression"])
    model_choice = st.sidebar.selectbox("Choose Model:", ["Random Forest", "XGBoost"])
    input_features = st.sidebar.text_input("Enter input values (comma-separated):")

    # Predict when button is clicked
    if st.sidebar.button("Predict"):
        try:
            # Convert input to array
            input_array = np.array([list(map(float, input_features.split(",")))]).reshape(1, -1)

            # Choose the right model
            if model_type == "Classification":
                model = rf_classifier if model_choice == "Random Forest" else xgb_classifier
            else:
                model = rf_regressor if model_choice == "Random Forest" else xgb_regressor

            # Make prediction
            prediction = model.predict(input_array)

            # Display result
            st.subheader("🔹 Prediction:")
            st.write(f"**{prediction[0]}**")

        except Exception as e:
            st.error(f"Error: {e}")
