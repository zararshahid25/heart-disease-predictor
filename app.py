# app.py - Streamlit Web App for Heart Disease Prediction

import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("best_rf_model.pkl")

# Set up the page
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Title and Description
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
Welcome to the Heart Disease Predictor App. This tool uses a trained machine learning model to assess the likelihood of heart disease based on clinical parameters. 
Please enter your details below:
""")

# Sidebar image or logo (optional)
# st.sidebar.image("logo.png", use_column_width=True)

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.radio("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
        chol = st.number_input("Serum Cholestoral in mg/dl (chol)", 100, 600, 240)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
        exang = st.radio("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict")

if submitted:
    sex_value = 1 if sex == "Male" else 0
    user_input = np.array([[age, sex_value, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(user_input)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ The model predicts: **High risk of heart disease**")
    else:
        st.success("✅ The model predicts: **Low risk of heart disease**")

    st.markdown("_Note: This tool is for educational purposes only. Please consult a doctor for medical advice._")

# Footer
st.markdown("""
---
Made with ❤️ by Zarar Shahid  
Powered by Streamlit and Scikit-learn
""")
