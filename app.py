import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import os
import tensorflow as tf

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮", layout="wide")

st.title("🏦 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer is likely to churn using an Artificial Neural Network (ANN).")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    model_path = os.path.join("artifacts", "churn_ann_model.keras")
    scaler_path = os.path.join("artifacts", "scaler.pkl")
    features_path = os.path.join("artifacts", "feature_columns.json")
    
    # Check if necessary files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        raise FileNotFoundError("Artifact files missing. Please generate the data and train the model first.")
        
    model = tf.keras.models.load_model(model_path)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    with open(features_path, "r") as f:
        feature_columns = json.load(f)
        
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("🎯 Customer Demographics")

geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 35)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)

st.sidebar.header("📊 Banking Patterns")
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, value=80000.0, step=1000.0)

# --- Feature Engineering ---
# Map inputs to what model expects
input_data = {
    'CreditScore': credit_score,
    'Gender': 1 if gender == 'Male' else 0,  # Based on standard LabelEncoder (Female: 0, Male: 1)
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_cr_card == "Yes" else 0,
    'IsActiveMember': 1 if is_active_member == "Yes" else 0,
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if geography == "France" else 0,
    'Geography_Germany': 1 if geography == "Germany" else 0,
    'Geography_Spain': 1 if geography == "Spain" else 0
}

input_df = pd.DataFrame([input_data])

# Ensure columns strictly align with training feature structure
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# Scale
input_scaled = scaler.transform(input_df)

# Predict
prediction_output = model.predict(input_scaled, verbose=0)
churn_probability = float(prediction_output[0][0])
retention_probability = 1.0 - churn_probability

st.markdown("---")

# --- Main Display Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("💡 Prediction Summary")
    risk_level = "High Risk" if churn_probability > 0.50 else "Low Risk"
    color = "red" if risk_level == "High Risk" else "green"
    
    st.markdown(f"### Target Status: <strong style='color:{color}'>{risk_level}</strong>", unsafe_allow_html=True)
    st.metric("Churn Risk", f"{churn_probability * 100:.2f}%")
    st.metric("Retention Probability", f"{retention_probability * 100:.2f}%")

with col2:
    st.subheader("📈 Risk Assessment")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability * 100,
        title={'text': "Churn Probability %", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 60], 'color': 'gold'},
                {'range': [60, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.info("ℹ️ **Tip:** Adjust the slider panels on the left to see real-time updates and understand how variables like Age and Balance impact the customer churn probability.")
