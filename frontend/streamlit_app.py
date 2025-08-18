import os
import requests
import streamlit as st

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://127.0.0.1:8000"))

st.title("🩺 Diabetes Prediction")
st.caption("Pima Indians Diabetes — FastAPI inference")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
        Glucose = st.number_input("Glucose", min_value=0.0, step=1.0, value=120.0)
        BloodPressure = st.number_input("Blood Pressure", min_value=0.0, step=1.0, value=70.0)
        SkinThickness = st.number_input("Skin Thickness", min_value=0.0, step=1.0, value=20.0)
    with col2:
        Insulin = st.number_input("Insulin", min_value=0.0, step=1.0, value=85.0)
        BMI = st.number_input("BMI", min_value=0.0, step=0.1, value=33.6)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, value=0.35)
        Age = st.number_input("Age", min_value=0, step=1, value=29)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Pregnancies": int(Pregnancies),
        "Glucose": float(Glucose),
        "BloodPressure": float(BloodPressure),
        "SkinThickness": float(SkinThickness),
        "Insulin": float(Insulin),
        "BMI": float(BMI),
        "DiabetesPedigreeFunction": float(DiabetesPedigreeFunction),
        "Age": int(Age)
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        st.success(f"Result: **{data['result']}**")
        st.metric("Prediction", data["prediction"])
        st.metric("Confidence", f"{data['confidence']*100:.2f}%")
    except Exception as e:
        st.error(f"Request failed: {e}")

with st.expander("Metrics (from test set)"):
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=15).json()
        st.json(m)
    except Exception as e:
        st.error(f"Could not load metrics: {e}")

st.caption(f"API: {API_URL}")
