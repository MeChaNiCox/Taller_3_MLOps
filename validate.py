# validate.py
import json
import sys
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Umbrales mínimos (ajústalos si tu profe indicó otros)
THRESHOLDS = {
    "accuracy": 0.90,
    "f1": 0.90,
    "roc_auc": 0.95,
}

# Cargar el mismo dataset y split que en train.py
X, y = load_breast_cancer(return_X_y=True, as_frame=False)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cargar modelo
model = joblib.load("model.pkl")

# Evaluar
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
}

print("🔎 Métricas de validación:", metrics)

# Validación por umbrales
fails = []
for k, thr in THRESHOLDS.items():
    if metrics[k] < thr:
        fails.append(f"{k}={metrics[k]:.4f} < {thr}")

if fails:
    print("❌ Validación fallida. Motivos:")
    for m in fails:
        print("  -", m)
    sys.exit(1)

print("✅ Validación superada.")
sys.exit(0)
