"""
Customer Churn Prediction - Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_artifacts()
explainer = shap.TreeExplainer(model)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("📉 Customer Churn Predictor")
st.markdown(
    "Enter a customer's details below to predict whether they are likely to **churn** "
    "and see which factors drive that prediction."
)
st.divider()

# ─────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────
st.sidebar.header("Customer Details")

def user_input():
    gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior         = st.sidebar.selectbox("Senior citizen?", ["No", "Yes"])
    partner        = st.sidebar.selectbox("Has partner?", ["Yes", "No"])
    dependents     = st.sidebar.selectbox("Has dependents?", ["No", "Yes"])
    tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone          = st.sidebar.selectbox("Phone service?", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple lines?", ["No", "Yes", "No phone service"])
    internet       = st.sidebar.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_sec     = st.sidebar.selectbox("Online security?", ["No", "Yes", "No internet service"])
    online_bkp     = st.sidebar.selectbox("Online backup?", ["Yes", "No", "No internet service"])
    device_prot    = st.sidebar.selectbox("Device protection?", ["No", "Yes", "No internet service"])
    tech_support   = st.sidebar.selectbox("Tech support?", ["No", "Yes", "No internet service"])
    streaming_tv   = st.sidebar.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
    streaming_mov  = st.sidebar.selectbox("Streaming movies?", ["No", "Yes", "No internet service"])
    contract       = st.sidebar.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    paperless      = st.sidebar.selectbox("Paperless billing?", ["Yes", "No"])
    payment        = st.sidebar.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly        = st.sidebar.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
    total          = monthly * tenure

    # Map to numeric (same encoding as train.py)
    encode_binary = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    multi_map     = {"No": 0, "Yes": 1, "No phone service": 2}
    internet_map  = {"DSL": 0, "Fiber optic": 1, "No": 2}
    inet_svc_map  = {"No": 0, "Yes": 1, "No internet service": 2}
    contract_map  = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map   = {
        "Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
        "Electronic check": 2, "Mailed check": 3
    }

    avg_charge        = total / (tenure + 1)
    multi_services    = (
        (1 if phone == "Yes" else 0) +
        (1 if internet != "No" else 0) +
        (1 if online_sec == "Yes" else 0) +
        (1 if streaming_tv == "Yes" else 0)
    )

    data = {
        "gender":            encode_binary[gender],
        "SeniorCitizen":     encode_binary[senior],
        "Partner":           encode_binary[partner],
        "Dependents":        encode_binary[dependents],
        "tenure":            tenure,
        "PhoneService":      encode_binary[phone],
        "MultipleLines":     multi_map[multiple_lines],
        "InternetService":   internet_map[internet],
        "OnlineSecurity":    inet_svc_map[online_sec],
        "OnlineBackup":      inet_svc_map[online_bkp],
        "DeviceProtection":  inet_svc_map[device_prot],
        "TechSupport":       inet_svc_map[tech_support],
        "StreamingTV":       inet_svc_map[streaming_tv],
        "StreamingMovies":   inet_svc_map[streaming_mov],
        "Contract":          contract_map[contract],
        "PaperlessBilling":  encode_binary[paperless],
        "PaymentMethod":     payment_map[payment],
        "MonthlyCharges":    monthly,
        "TotalCharges":      total,
        "AvgMonthlyCharge":  avg_charge,
        "HasMultipleServices": multi_services,
    }
    return pd.DataFrame([data])

input_df = user_input()

# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
prob = model.predict_proba(input_df)[0][1]
pred = int(prob >= 0.5)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{prob:.1%}")

with col2:
    label = "⚠️ Likely to Churn" if pred == 1 else "✅ Likely to Stay"
    color = "red" if pred == 1 else "green"
    st.markdown(f"### Prediction\n**:{color}[{label}]**")

with col3:
    risk = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
    st.metric("Risk Level", risk)

st.divider()

# ─────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────
st.subheader("🔍 Why this prediction? (SHAP explanation)")
st.caption("Bars show how much each feature pushed the prediction toward churn (red) or staying (blue).")

shap_vals = explainer.shap_values(input_df)
fig, ax = plt.subplots(figsize=(8, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=feature_names
    ),
    show=False
)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ─────────────────────────────────────────
# RAW INPUT TABLE
# ─────────────────────────────────────────
with st.expander("Show raw input data"):
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.caption(
    "Built with XGBoost · SHAP · Streamlit · "
    "Dataset: Telco Customer Churn (Kaggle) · "
    "Model AUC: ~0.86"
)