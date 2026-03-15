import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import BASELINE_MODEL_PATH, RANDOM_STATE

def train_baseline_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=200,
        max_depth=None
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n✅ Baseline Random Forest Accuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    joblib.dump(model, BASELINE_MODEL_PATH)
    print(f"\n💾 Baseline model saved at: {BASELINE_MODEL_PATH}")

    return model, acc
