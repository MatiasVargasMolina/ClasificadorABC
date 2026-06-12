from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, List

import autosklearn.classification
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.ml.explainability.shap_formatter import (
    build_global_importance,
    build_local_contributions,
    normalize_shap_values,
)


FEATURE_COLUMNS = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
    "en_promocion",
]

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/artifacts"))

MODEL_PATH = ARTIFACT_DIR / "autosklearn_model.joblib"
BACKGROUND_PATH = ARTIFACT_DIR / "autosklearn_background.joblib"
META_PATH = ARTIFACT_DIR / "autosklearn_meta.joblib"


class TrainRow(BaseModel):
    publication_id: str
    ventas_30d: int = Field(..., ge=0)
    visitas_30d: int = Field(..., ge=0)
    precio_actual: float = Field(..., gt=0)
    stock_actual: int = Field(..., ge=0)
    en_promocion: int = Field(..., ge=0, le=1)
    categoria: str


class PredictRow(BaseModel):
    publication_id: str
    ventas_30d: int = Field(..., ge=0)
    visitas_30d: int = Field(..., ge=0)
    precio_actual: float = Field(..., gt=0)
    stock_actual: int = Field(..., ge=0)
    en_promocion: int = Field(..., ge=0, le=1)


class TrainRequest(BaseModel):
    rows: List[TrainRow] = Field(..., min_items=3)
    time_left_for_this_task: int = Field(default=120, ge=30, le=1800)
    per_run_time_limit: int = Field(default=30, ge=10, le=300)


class ExplainRequest(BaseModel):
    rows: List[PredictRow] = Field(..., min_items=1)
    top_n: int = Field(default=5, ge=1, le=10)


app = FastAPI(title="AutoSklearn Worker")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "autosklearn-worker",
        "artifact_dir": str(ARTIFACT_DIR),
        "model_exists": MODEL_PATH.exists(),
        "background_exists": BACKGROUND_PATH.exists(),
        "meta_exists": META_PATH.exists(),
    }


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def rows_to_dataframe(rows: List[Any]) -> pd.DataFrame:
    df = pd.DataFrame([row.dict() for row in rows])
    df["en_promocion"] = df["en_promocion"].astype(int)
    return df


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No existe modelo entrenado. Ejecuta /train primero.")

    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError("No existe background SHAP. Ejecuta /train primero.")

    if not META_PATH.exists():
        raise FileNotFoundError("No existe metadata. Ejecuta /train primero.")

    model = joblib.load(MODEL_PATH)
    background = joblib.load(BACKGROUND_PATH)
    meta = joblib.load(META_PATH)

    return model, background, meta


def build_predict_proba_fn(model):
    """
    SHAP KernelExplainer normalmente entrega numpy arrays al modelo.
    AutoSklearn fue entrenado usando pandas DataFrame y su pipeline interno
    puede requerir nombres de columnas. Esta función asegura que cualquier
    entrada se transforme a DataFrame con FEATURE_COLUMNS antes de predecir.
    """

    def predict_proba_fn(data):
        if isinstance(data, pd.DataFrame):
            X_input = data[FEATURE_COLUMNS].copy()
        else:
            X_input = pd.DataFrame(data, columns=FEATURE_COLUMNS)

        return model.predict_proba(X_input)

    return predict_proba_fn


def build_predict_fn(model):
    """
    Igual que build_predict_proba_fn, pero para predict().
    Mantiene entrada como DataFrame con columnas.
    """

    def predict_fn(data):
        if isinstance(data, pd.DataFrame):
            X_input = data[FEATURE_COLUMNS].copy()
        else:
            X_input = pd.DataFrame(data, columns=FEATURE_COLUMNS)

        return model.predict(X_input)

    return predict_fn


def get_autosklearn_model_info(model) -> dict:
    """
    Extrae información del modelo/pipeline encontrado por AutoSklearn.

    AutoSklearn suele entrenar varios pipelines y construir un ensamble.
    Por eso exponemos:
    - sprint_statistics: resumen general del entrenamiento
    - leaderboard: tabla de modelos evaluados
    - models_with_weights: pipelines seleccionados y pesos del ensamble
    """

    info = {
        "automl_class": model.__class__.__name__,
        "sprint_statistics": None,
        "leaderboard": None,
        "models_with_weights": None,
    }

    try:
        info["sprint_statistics"] = str(model.sprint_statistics())
    except Exception as exc:
        info["sprint_statistics"] = f"No disponible: {exc}"

    try:
        leaderboard = model.leaderboard()
        info["leaderboard"] = leaderboard.to_dict(orient="records")
    except Exception as exc:
        info["leaderboard"] = f"No disponible: {exc}"

    try:
        info["models_with_weights"] = str(model.show_models())
    except Exception as exc:
        info["models_with_weights"] = f"No disponible: {exc}"

    return info


@app.post("/train")
def train(request: TrainRequest):
    ensure_artifact_dir()

    df = rows_to_dataframe(request.rows)

    X = df[FEATURE_COLUMNS].copy()
    y = df["categoria"].astype(str).copy()

    if y.nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 2 clases distintas para entrenar AutoSklearn.",
        )

    stratify = y if y.value_counts().min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=None,
        )

    start = time.time()

    model = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=request.time_left_for_this_task,
        per_run_time_limit=request.per_run_time_limit,
        memory_limit=4096,
        seed=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predict_fn = build_predict_fn(model)
    y_pred = predict_fn(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": sorted(y.unique().tolist()),
        "training_seconds": round(time.time() - start, 3),
    }

    model_info = get_autosklearn_model_info(model)

    background = X_train.sample(
        n=min(20, len(X_train)),
        random_state=42,
    ).copy()

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "classes": sorted(y.unique().tolist()),
        "model_info": model_info,
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(background, BACKGROUND_PATH)
    joblib.dump(meta, META_PATH)

    return {
        "mensaje": "AutoSklearn entrenado correctamente",
        "metrics": metrics,
        "model_info": model_info,
        "artifacts": {
            "model_path": str(MODEL_PATH),
            "background_path": str(BACKGROUND_PATH),
            "meta_path": str(META_PATH),
        },
    }


@app.post("/explain")
def explain(request: ExplainRequest):
    try:
        model, background, meta = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    df = rows_to_dataframe(request.rows)
    X = df[FEATURE_COLUMNS].copy()

    predict_fn = build_predict_fn(model)
    predict_proba_fn = build_predict_proba_fn(model)

    predictions = predict_fn(X)
    probabilities = predict_proba_fn(X)

    classes = [str(c) for c in model.classes_]

    explainer = shap.KernelExplainer(
        predict_proba_fn,
        background,
    )

    shap_raw = explainer.shap_values(
        X,
        nsamples=100,
    )

    shap_values = normalize_shap_values(shap_raw)

    predicciones = []

    for i in range(len(X)):
        prediction = str(predictions[i])
        class_index = classes.index(prediction) if prediction in classes else 0

        if shap_values.shape[-1] == 1:
            local_values = shap_values[i, :, 0]
        else:
            local_values = shap_values[i, :, class_index]

        contribuciones = build_local_contributions(
            feature_columns=FEATURE_COLUMNS,
            feature_values=X.iloc[i].tolist(),
            shap_values=local_values.tolist(),
        )

        proba_row = probabilities[i]
        probabilidades = {
            classes[j]: float(proba_row[j])
            for j in range(len(classes))
        }

        predicciones.append(
            {
                "publication_id": df.loc[i, "publication_id"],
                "prediccion": prediction,
                "probabilidades": probabilidades,
                "top_contribuciones": contribuciones[: request.top_n],
                "contribuciones": contribuciones,
            }
        )

    importancia_global = build_global_importance(
        feature_columns=FEATURE_COLUMNS,
        shap_values=shap_values,
    )

    return {
        "mensaje": "Explicaciones generadas correctamente",
        "modelo": "AutoSklearnClassifier",
        "metrics_entrenamiento": meta.get("metrics"),
        "model_info": meta.get("model_info"),
        "importancia_global": importancia_global,
        "predicciones": predicciones,
    }