
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# Load model and scaler
model = pickle.load(open("final_model_v3.pkl", "rb"))
scaler = pickle.load(open("scaler_v3.pkl", "rb"))

# Feature columns after preprocessing
feature_cols = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_Yes", "OnlineBackup_Yes",
    "DeviceProtection_Yes", "TechSupport_Yes",
    "StreamingTV_Yes", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaperlessBilling", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "PaymentMethod_Credit card (automatic)",
    "MonthlyCharges"
]

# Remove noise features
noise_features = [
    "InternetService_No",
    "MultipleLines_No phone service",
    "StreamingMovies_No internet service",
    "OnlineSecurity_No internet service",
    "StreamingTV_No internet service",
    "OnlineBackup_No internet service",
    "DeviceProtection_No internet service",
    "TechSupport_No internet service"
]

final_cols = [f for f in feature_cols if f not in noise_features]

st.set_page_config(page_title="Churn Predictor", 
                    page_icon="📊", layout="wide")

st.title("📊 Customer Churn Predictor")
st.markdown("Predict whether a telecom customer will churn "
            "using CatBoost + SHAP explainability")

st.sidebar.header("Customer Information")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", 
                                       ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", 
                                 ["Fiber optic", "DSL", "No"])
online_sec = st.sidebar.selectbox("Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
device_prot = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", 
                                         ["Yes", "No"])
contract = st.sidebar.selectbox("Contract", 
                                 ["Month-to-month", 
                                  "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", 
                                ["Electronic check",
                                 "Mailed check",
                                 "Bank transfer (automatic)",
                                 "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 
                                     18.0, 120.0, 65.0)

if st.sidebar.button("Predict Churn", type="primary"):
    
    # Build input dict
    input_dict = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone == "Yes" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "OnlineSecurity_Yes": 1 if online_sec == "Yes" else 0,
        "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
        "DeviceProtection_Yes": 1 if device_prot == "Yes" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "MonthlyCharges": monthly_charges
    }
    
    # Create dataframe
    input_df = pd.DataFrame([input_dict])
    
    # Scale numerical
    input_df[["tenure", "MonthlyCharges"]] = scaler.transform(
        input_df[["tenure", "MonthlyCharges"]])
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    prediction = "WILL CHURN" if prob > 0.5 else "WILL NOT CHURN"
    
    # Display result
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "red" if prob > 0.5 else "green"
        st.markdown(f"### Prediction")
        st.markdown(f"<h2 style=color:{color}>{prediction}</h2>",
                    unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Churn Probability")
        st.markdown(f"<h2 style=color:{color}>{prob:.1%}</h2>",
                    unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Risk Level")
        if prob > 0.7:
            st.markdown("<h2 style=color:red>🔴 HIGH</h2>",
                        unsafe_allow_html=True)
        elif prob > 0.4:
            st.markdown("<h2 style=color:orange>🟡 MEDIUM</h2>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<h2 style=color:green>🟢 LOW</h2>",
                        unsafe_allow_html=True)
    
    st.divider()
    
    # SHAP explanation
    st.markdown("### Why this prediction? (SHAP Explanation)")
    
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    
    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns.tolist()
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Top reasons in plain English
    st.markdown("### Top Reasons")
    shap_series = pd.Series(shap_vals[0], 
                             index=input_df.columns)
    top_churn = shap_series.nlargest(3)
    top_retain = shap_series.nsmallest(3)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pushing toward churn:**")
        for feat, val in top_churn.items():
            st.markdown(f"- {feat}: +{val:.3f}")
    with col2:
        st.markdown("**Pushing away from churn:**")
        for feat, val in top_retain.items():
            st.markdown(f"- {feat}: {val:.3f}")

st.divider()
st.markdown("**Model:** CatBoost | "
            "**AUC-ROC:** 0.8421 | "
            "**Recall:** 0.8075 | "
            "**Dataset:** Telco Customer Churn (Kaggle)")
