import joblib
import numpy as np

from src.config import (
    OPTIMIZED_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
)

def load_predictor():
    model = joblib.load(OPTIMIZED_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler, le

def predict_crop(model, scaler, label_encoder, features_dict: dict, top_k=3):
    """
    features_dict must contain:
    n, p, k, temperature, humidity, ph, rainfall
    """
    order = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

    x = np.array([[features_dict[c] for c in order]], dtype=float)
    x_scaled = scaler.transform(x)

    proba = model.predict_proba(x_scaled)[0]
    top_indices = np.argsort(proba)[::-1][:top_k]

    results = []
    for idx in top_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        results.append((crop_name, float(proba[idx])))

    best_crop = results[0][0]
    return best_crop, results
