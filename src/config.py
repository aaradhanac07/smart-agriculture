import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "crop_recommendation.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.joblib")
OPTIMIZED_MODEL_PATH = os.path.join(MODELS_DIR, "optimized_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params.json")

RANDOM_STATE = 42

# Weather API
OPENWEATHER_API_KEY = os.getenv("11cd33b2b7ac237357874fcf5e9096f9", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
