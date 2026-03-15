from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

app = Flask(__name__)
CORS(app)  # Allow HTML UI to talk to this server

# ── Load models once at startup ───────────────────────────────────────────────
try:
    from src.config import OPTIMIZED_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH
    model   = joblib.load(OPTIMIZED_MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    le      = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = scaler = le = None

# ── Import your logic ─────────────────────────────────────────────────────────
try:
    from src.fertilizer_recommendation import fertilizer_advice
    from src.soil_health import soil_health_score
except Exception:
    def fertilizer_advice(n, p, k):
        advice = []
        if n < 50:    advice.append("Nitrogen is LOW → add Urea / Nitrogen fertilizer.")
        elif n > 120: advice.append("Nitrogen is HIGH → reduce nitrogen fertilizer.")
        else:         advice.append("Nitrogen is OPTIMAL.")
        if p < 40:    advice.append("Phosphorus is LOW → add DAP / SSP fertilizer.")
        elif p > 100: advice.append("Phosphorus is HIGH → reduce phosphorus fertilizer.")
        else:         advice.append("Phosphorus is OPTIMAL.")
        if k < 40:    advice.append("Potassium is LOW → add MOP / Potash fertilizer.")
        elif k > 100: advice.append("Potassium is HIGH → reduce potash fertilizer.")
        else:         advice.append("Potassium is OPTIMAL.")
        return advice

    def soil_health_score(n, p, k, ph):
        score = 100
        if n < 50 or n > 120: score -= 15
        if p < 40 or p > 100: score -= 15
        if k < 40 or k > 100: score -= 15
        if ph < 5.5 or ph > 7.5: score -= 25
        elif 6.0 <= ph <= 7.0:   score += 5
        score = max(0, min(100, score))
        level = "Excellent" if score >= 85 else "Good" if score >= 70 else "Average" if score >= 50 else "Poor"
        return score, level


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Smart Agriculture API is running!", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract inputs
        n        = float(data["n"])
        p        = float(data["p"])
        k        = float(data["k"])
        ph       = float(data["ph"])
        temp     = float(data["temperature"])
        humidity = float(data["humidity"])
        rainfall = float(data["rainfall"])

        result = {}

        # ── Crop prediction ───────────────────────────────────────────────────
        if model and scaler and le:
            features  = np.array([[n, p, k, temp, humidity, ph, rainfall]])
            scaled    = scaler.transform(features)
            probs     = model.predict_proba(scaled)[0]
            top3_idx  = np.argsort(probs)[::-1][:3]
            top3 = [
                {"crop": str(le.inverse_transform([i])[0]), "probability": round(float(probs[i]) * 100, 2)}
                for i in top3_idx
            ]
            result["crop_prediction"] = {
                "best_crop":   top3[0]["crop"],
                "confidence":  top3[0]["probability"],
                "top3":        top3
            }
        else:
            result["crop_prediction"] = {"error": "Model not loaded. Run main.py first."}

        # ── Soil health ───────────────────────────────────────────────────────
        score, level = soil_health_score(n, p, k, ph)
        result["soil_health"] = {"score": score, "level": level}

        # ── Fertilizer advice ─────────────────────────────────────────────────
        result["fertilizer_advice"] = fertilizer_advice(n, p, k)

        return jsonify({"success": True, "data": result})

    except KeyError as e:
        return jsonify({"success": False, "error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/weather", methods=["GET"])
def weather():
    """Proxy weather data so the HTML UI doesn't expose the API key."""
    try:
        import requests
        from src.config import OPENWEATHER_API_KEY, OPENWEATHER_BASE_URL

        city = request.args.get("city", "Mumbai")
        if not OPENWEATHER_API_KEY:
            return jsonify({"success": False, "error": "No API key set in config.py"}), 400

        resp = requests.get(OPENWEATHER_BASE_URL, params={
            "q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"
        })
        weather_data = resp.json()

        if weather_data.get("cod") != 200:
            return jsonify({"success": False, "error": weather_data.get("message", "City not found")}), 404

        return jsonify({
            "success": True,
            "city":        weather_data["name"],
            "temperature": round(weather_data["main"]["temp"], 1),
            "humidity":    weather_data["main"]["humidity"],
            "description": weather_data["weather"][0]["description"],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n🌱 Starting Smart Agriculture API...")
    print("📡 Open your HTML UI — it will now use real ML predictions!\n")
    app.run(debug=True, port=5000)