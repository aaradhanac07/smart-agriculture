import streamlit as st
import joblib
import numpy as np

from src.predict import load_predictor, predict_crop
from src.fertilizer_recommendation import fertilizer_advice
from src.soil_health import soil_health_score
from src.config import SCALER_PATH

# ---------------- PAGE CONFIG ----------------
st.write("App is running")
st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>
    🌾 Smart Agriculture Recommendation System
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("### Enter Soil Parameters Below")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    n = st.slider("Nitrogen (N)", 0, 150, 90)
    p = st.slider("Phosphorus (P)", 0, 150, 40)
    k = st.slider("Potassium (K)", 0, 150, 40)

with col2:
    ph = st.slider("pH Level", 0.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)
    temperature = st.slider("Temperature (°C)", 0, 50, 25)

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict Best Crop"):

    predictor = load_predictor()
    scaler = joblib.load("models/scaler.joblib")

    input_data = np.array([[n, p, k, temperature, rainfall, ph]])
    input_scaled = scaler.transform(input_data)

    predicted_crop, top3 = predict_crop(predictor, input_data)

    st.success(f"🌱 Recommended Crop: **{predicted_crop}**")

    st.subheader("📊 Top 3 Predictions")
    for crop, prob in top3:
        st.write(f"- {crop}: {prob*100:.2f}%")

    # Fertilizer Advice
    st.subheader("🧪 Fertilizer Advice")
    advice = fertilizer_advice(n, p, k)
    for line in advice:
        st.write(f"- {line}")

    # Soil Health
    score, level = soil_health_score(n, p, k, ph)

    st.subheader("🌿 Soil Health Score")
    st.write(f"**{score}/100 ({level})**")
