# train.py
import os
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1) Datos integrados (no requiere archivos externos)
X, y = load_breast_cancer(return_X_y=True, as_frame=False)

# 2) Split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Pipeline sencillo
pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ]
)

# 4) MLflow local en carpeta del repo
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlflow-ci")

with mlflow.start_run(run_name="train"):
    # Params
    params = {"clf__max_iter": 200, "clf__solver": "lbfgs"}
    for k, v in params.items():
        mlflow.log_param(k, v)

    # Entrenar
    pipe.fit(X_train, y_train)

    # Métricas
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Guardados locales para CI
    joblib.dump(pipe, "model.pkl")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Log de artefactos en MLflow
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("metrics.json")

print("✅ Entrenamiento OK. Métricas:", metrics)
