import streamlit as st
import numpy as np
import joblib

# Load Model
model = joblib.load("startup_profit_model.pkl")

st.set_page_config(page_title="Startup Profit Predictor")

st.title(" Startup Profit Prediction")

st.write("Enter the startup details below.")

rd = st.number_input("Enter R&D Spend", min_value=0.0)

admin = st.number_input("Enter Administration Cost", min_value=0.0)

marketing = st.number_input("Enter Marketing Spend", min_value=0.0)

if st.button("Predict Profit"):

    features = np.array([[rd, admin, marketing]])

    prediction = model.predict(features)

    st.success(f"💰 Predicted Profit: ${prediction[0]:,.2f}")