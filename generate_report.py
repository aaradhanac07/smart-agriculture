"""
generate_report.py
Run this once to generate your Model Accuracy Report.
It prints stats AND saves a confusion matrix image.

Usage:
    python generate_report.py
"""

import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Imports ───────────────────────────────────────────────────────────────────
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.config import (
    DATA_PATH, OPTIMIZED_MODEL_PATH,
    SCALER_PATH, LABEL_ENCODER_PATH
)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
X  = df.drop("label", axis=1).values
y  = df["label"].values

# ── Load model artifacts ──────────────────────────────────────────────────────
print("🔧 Loading saved model, scaler, encoder...")
model  = joblib.load(OPTIMIZED_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le     = joblib.load(LABEL_ENCODER_PATH)

# ── Encode labels ─────────────────────────────────────────────────────────────
y_encoded = le.transform(y)

# ── Scale features ────────────────────────────────────────────────────────────
X_scaled = scaler.transform(X)

# ── Predictions ───────────────────────────────────────────────────────────────
print("🤖 Running predictions on full dataset...")
y_pred_encoded = model.predict(X_scaled)
y_pred         = le.inverse_transform(y_pred_encoded)

# ── Metrics ───────────────────────────────────────────────────────────────────
acc  = accuracy_score(y, y_pred) * 100
prec = precision_score(y, y_pred, average="weighted", zero_division=0) * 100
rec  = recall_score(y, y_pred, average="weighted", zero_division=0) * 100
f1   = f1_score(y, y_pred, average="weighted", zero_division=0) * 100

print("\n" + "="*50)
print("       MODEL ACCURACY REPORT")
print("="*50)
print(f"  Accuracy  : {acc:.2f}%")
print(f"  Precision : {prec:.2f}%")
print(f"  Recall    : {rec:.2f}%")
print(f"  F1 Score  : {f1:.2f}%")
print("="*50)

# ── Per-class report ──────────────────────────────────────────────────────────
print("\n📋 Per-Class Classification Report:\n")
print(classification_report(y, y_pred, zero_division=0))

# ── Save metrics to JSON (for the report doc) ─────────────────────────────────
metrics = {
    "accuracy":  round(acc, 2),
    "precision": round(prec, 2),
    "recall":    round(rec, 2),
    "f1_score":  round(f1, 2),
}
os.makedirs("reports", exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\n✅ Metrics saved to reports/metrics.json")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
print("\n🎨 Generating confusion matrix image...")
classes = le.classes_
cm      = confusion_matrix(y, y_pred, labels=classes)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlGn",
    xticklabels=classes, yticklabels=classes,
    linewidths=0.5, linecolor="#e0e0e0",
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Confusion Matrix — Smart Agriculture Crop Prediction", fontsize=16, fontweight="bold", pad=20, color="#1a3d2b")
ax.set_xlabel("Predicted Crop", fontsize=13, labelpad=12)
ax.set_ylabel("Actual Crop",    fontsize=13, labelpad=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0,  fontsize=10)
plt.tight_layout()

cm_path = "reports/confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Confusion matrix saved to {cm_path}")

# ── Bar chart of per-class F1 ─────────────────────────────────────────────────
print("\n📊 Generating per-class F1 score chart...")
from sklearn.metrics import f1_score as f1_per
f1_per_class = f1_score(y, y_pred, average=None, labels=classes, zero_division=0) * 100

fig2, ax2 = plt.subplots(figsize=(14, 6))
colors = ["#2d6a4f" if v >= 95 else "#52b788" if v >= 85 else "#f4a261" if v >= 70 else "#e63946" for v in f1_per_class]
bars = ax2.barh(classes, f1_per_class, color=colors, height=0.65, edgecolor="white")
ax2.set_xlabel("F1 Score (%)", fontsize=12)
ax2.set_title("Per-Crop F1 Score", fontsize=14, fontweight="bold", color="#1a3d2b", pad=15)
ax2.set_xlim(0, 105)
ax2.axvline(x=90, color="#888888", linestyle="--", linewidth=1, alpha=0.5, label="90% threshold")
for bar, val in zip(bars, f1_per_class):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=9, color="#333")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=10)
plt.tight_layout()

f1_path = "reports/f1_per_crop.png"
plt.savefig(f1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ F1 chart saved to {f1_path}")

print("\n🎉 Report generation complete!")
print("   Check the reports/ folder for all output files.")
print(f"\n   Accuracy  : {acc:.2f}%")
print(f"   Precision : {prec:.2f}%")
print(f"   Recall    : {rec:.2f}%")
print(f"   F1 Score  : {f1:.2f}%\n")