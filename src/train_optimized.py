import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import OPTIMIZED_MODEL_PATH, BEST_PARAMS_PATH, RANDOM_STATE

def train_optimized_model(X_train, y_train, X_test, y_test):
    with open(BEST_PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    model = RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n✅ PSO Optimized Random Forest Accuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    joblib.dump(model, OPTIMIZED_MODEL_PATH)
    print(f"\n💾 Optimized model saved at: {OPTIMIZED_MODEL_PATH}")

    return model, acc
