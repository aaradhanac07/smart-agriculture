import joblib

from src.config import DATA_PATH, SCALER_PATH, LABEL_ENCODER_PATH
from src.preprocess import load_dataset, preprocess_dataset
from src.train_baseline import train_baseline_model
from src.pso_optimizer import PSOOptimizer
from src.train_optimized import train_optimized_model

from src.predict import load_predictor, predict_crop
from src.fertilizer_recommendation import fertilizer_advice
from src.soil_health import soil_health_score
# from src.weather_api import get_weather_by_city   # optional


def run_pipeline():
    print("📌 Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("📌 Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, le, feature_cols = preprocess_dataset(df)

    # Save scaler + label encoder
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print("\n==============================")
    print("1) Training Baseline Model")
    print("==============================")
    train_baseline_model(X_train, y_train, X_test, y_test)

    print("\n==============================")
    print("2) PSO Hyperparameter Optimization (Parallel)")
    print("==============================")
    pso = PSOOptimizer(
        n_particles=10,
        n_iterations=10,
        n_estimators_bounds=(50, 400),
        max_depth_bounds=(3, 30),
    )
    pso.optimize(X_train, y_train, parallel=True)

    print("\n==============================")
    print("3) Training Optimized Model")
    print("==============================")
    train_optimized_model(X_train, y_train, X_test, y_test)

    print("\n✅ Pipeline complete!")


def demo_prediction():
    print("\n==============================")
    print("4) Demo Prediction (No UI)")
    print("==============================")

    model, scaler, le = load_predictor()

    # Example input (you can change)
    user_input = {
        "n": 90,
        "p": 42,
        "k": 43,
        "temperature": 25.5,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 120
    }

    best_crop, top3 = predict_crop(model, scaler, le, user_input, top_k=3)

    print("\n🌾 Recommended Crop:", best_crop)
    print("\n📌 Top 3 Predictions:")
    for crop, prob in top3:
        print(f"  - {crop}: {prob*100:.2f}%")

    print("\n🧪 Fertilizer Advice:")
    for line in fertilizer_advice(user_input["n"], user_input["p"], user_input["k"]):
        print("  -", line)

    score, level = soil_health_score(
        user_input["n"], user_input["p"], user_input["k"], user_input["ph"]
    )
    print(f"\n🌱 Soil Health Score: {score}/100 ({level})")


if __name__ == "__main__":
    run_pipeline()
    demo_prediction()
