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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from app.ml.explainability.shap_formatter import (
    build_additivity_diagnostics,
    build_global_importance,
    build_global_importance_by_class,
    build_local_contributions,
    normalize_expected_values,
    normalize_shap_values,
)


FEATURE_COLUMNS = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
]

ARTIFACT_DIR = Path(
    os.getenv(
        "ARTIFACT_DIR",
        "/artifacts",
    )
)

MODEL_PATH = (
    ARTIFACT_DIR / "autosklearn_model.joblib"
)
BACKGROUND_PATH = (
    ARTIFACT_DIR / "autosklearn_background.joblib"
)
META_PATH = (
    ARTIFACT_DIR / "autosklearn_meta.joblib"
)

SHAP_BACKGROUND_SIZE = int(
    os.getenv(
        "SHAP_BACKGROUND_SIZE",
        "50",
    )
)

SHAP_BACKGROUND_RANDOM_STATE = int(
    os.getenv(
        "SHAP_BACKGROUND_RANDOM_STATE",
        "42",
    )
)

SHAP_MAX_EXPLAIN_ROWS = int(
    os.getenv(
        "SHAP_MAX_EXPLAIN_ROWS",
        "60",
    )
)

SHAP_NSAMPLES = int(
    os.getenv(
        "SHAP_NSAMPLES",
        "6",
    )
)


class TrainRow(BaseModel):
    publication_id: str
    ventas_30d: int = Field(..., ge=0)
    visitas_30d: int = Field(..., ge=0)
    precio_actual: float = Field(..., gt=0)
    categoria: str


class PredictRow(BaseModel):
    publication_id: str
    ventas_30d: int = Field(..., ge=0)
    visitas_30d: int = Field(..., ge=0)
    precio_actual: float = Field(..., gt=0)


class TrainRequest(BaseModel):
    rows: List[TrainRow] = Field(
        ...,
        min_items=3,
    )
    time_left_for_this_task: int = Field(
        default=600,
        ge=30,
        le=1800,
    )
    per_run_time_limit: int = Field(
        default=60,
        ge=10,
        le=300,
    )


class ExplainRequest(BaseModel):
    rows: List[PredictRow] = Field(
        ...,
        min_items=1,
    )
    top_n: int = Field(
        default=3,
        ge=1,
        le=3,
    )


app = FastAPI(
    title="AutoSklearn Worker",
)


@app.get("/health")
def health():
    artifact_features = None
    artifacts_compatible = False

    if META_PATH.exists():
        try:
            meta = joblib.load(META_PATH)
            artifact_features = meta.get(
                "feature_columns"
            )
            artifacts_compatible = (
                artifact_features
                == FEATURE_COLUMNS
            )
        except Exception:
            artifacts_compatible = False

    return {
        "status": "ok",
        "service": "autosklearn-worker",
        "artifact_dir": str(ARTIFACT_DIR),
        "model_exists": MODEL_PATH.exists(),
        "background_exists": (
            BACKGROUND_PATH.exists()
        ),
        "meta_exists": META_PATH.exists(),
        "feature_columns": FEATURE_COLUMNS,
        "artifact_feature_columns": (
            artifact_features
        ),
        "artifacts_compatible": (
            artifacts_compatible
        ),
        "shap_config": {
            "background_size": (
                SHAP_BACKGROUND_SIZE
            ),
            "background_random_state": (
                SHAP_BACKGROUND_RANDOM_STATE
            ),
            "max_explain_rows": (
                SHAP_MAX_EXPLAIN_ROWS
            ),
            "nsamples": SHAP_NSAMPLES,
        },
    }


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def rows_to_dataframe(
    rows: List[Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [row.dict() for row in rows]
    )


def to_feature_dataframe(
    data: Any,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data[FEATURE_COLUMNS].copy()

    return pd.DataFrame(
        data,
        columns=FEATURE_COLUMNS,
    )


def build_predict_proba_fn(model):
    def predict_proba_fn(data):
        X_input = to_feature_dataframe(data)
        return model.predict_proba(X_input)

    return predict_proba_fn


def build_predict_fn(model):
    def predict_fn(data):
        X_input = to_feature_dataframe(data)
        return model.predict(X_input)

    return predict_fn


def load_artifacts():
    missing_paths = [
        str(path)
        for path in (
            MODEL_PATH,
            BACKGROUND_PATH,
            META_PATH,
        )
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Faltan artefactos del modelo: "
            f"{missing_paths}. Ejecuta /train."
        )

    model = joblib.load(MODEL_PATH)
    background = joblib.load(BACKGROUND_PATH)
    meta = joblib.load(META_PATH)

    artifact_features = meta.get(
        "feature_columns"
    )

    if artifact_features != FEATURE_COLUMNS:
        raise RuntimeError(
            "Los artefactos fueron entrenados con variables "
            f"{artifact_features}, pero el worker actual utiliza "
            f"{FEATURE_COLUMNS}. Ejecuta /train nuevamente."
        )

    if list(background.columns) != FEATURE_COLUMNS:
        raise RuntimeError(
            "El background SHAP no coincide con las variables "
            "del worker. Ejecuta /train nuevamente."
        )

    return model, background, meta


def get_autosklearn_model_info(
    model,
) -> dict:
    info = {
        "automl_class": (
            model.__class__.__name__
        ),
        "sprint_statistics": None,
        "leaderboard": None,
        "models_with_weights": None,
    }

    try:
        info["sprint_statistics"] = str(
            model.sprint_statistics()
        )
    except Exception as exc:
        info["sprint_statistics"] = (
            f"No disponible: {exc}"
        )

    try:
        leaderboard = model.leaderboard()
        info["leaderboard"] = (
            leaderboard.to_dict(
                orient="records"
            )
        )
    except Exception as exc:
        info["leaderboard"] = (
            f"No disponible: {exc}"
        )

    try:
        info["models_with_weights"] = str(
            model.show_models()
        )
    except Exception as exc:
        info["models_with_weights"] = (
            f"No disponible: {exc}"
        )

    return info


def build_background(
    X_train: pd.DataFrame,
) -> pd.DataFrame:
    background_size = min(
        SHAP_BACKGROUND_SIZE,
        len(X_train),
    )

    return (
        X_train.sample(
            n=background_size,
            random_state=(
                SHAP_BACKGROUND_RANDOM_STATE
            ),
        )
        .reset_index(drop=True)
        .copy()
    )


@app.post("/train")
def train(
    request: TrainRequest,
):
    if (
        request.per_run_time_limit
        > request.time_left_for_this_task
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "per_run_time_limit no puede ser mayor que "
                "time_left_for_this_task."
            ),
        )

    ensure_artifact_dir()

    df = rows_to_dataframe(request.rows)
    X = df[FEATURE_COLUMNS].copy()
    y = df["categoria"].astype(str).copy()

    if y.nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Se necesitan al menos dos clases distintas "
                "para entrenar AutoSklearn."
            ),
        )

    class_counts = y.value_counts()
    stratify = (
        y
        if class_counts.min() >= 2
        else None
    )

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=None,
        )

    start = time.time()

    model = (
        autosklearn.classification
        .AutoSklearnClassifier(
            time_left_for_this_task=(
                request.time_left_for_this_task
            ),
            per_run_time_limit=(
                request.per_run_time_limit
            ),
            memory_limit=4096,
            seed=42,
            n_jobs=-1,
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    predict_fn = build_predict_fn(model)
    y_pred = predict_fn(X_test)

    classes = sorted(
        y.unique().tolist()
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_test,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                y_pred,
                average="macro",
            )
        ),
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": classes,
        "confusion_matrix": (
            confusion_matrix(
                y_test,
                y_pred,
                labels=classes,
            ).tolist()
        ),
        "classification_report": report,
        "training_seconds": round(
            time.time() - start,
            3,
        ),
    }

    model_info = get_autosklearn_model_info(
        model
    )

    background = build_background(
        X_train
    )

    meta = {
        "artifact_version": 2,
        "feature_columns": (
            FEATURE_COLUMNS.copy()
        ),
        "metrics": metrics,
        "classes": classes,
        "model_info": model_info,
        "shap_config": {
            "background_size": int(
                len(background)
            ),
            "background_random_state": (
                SHAP_BACKGROUND_RANDOM_STATE
            ),
            "max_explain_rows": (
                SHAP_MAX_EXPLAIN_ROWS
            ),
            "nsamples": SHAP_NSAMPLES,
        },
    }

    joblib.dump(
        model,
        MODEL_PATH,
    )
    joblib.dump(
        background,
        BACKGROUND_PATH,
    )
    joblib.dump(
        meta,
        META_PATH,
    )

    return {
        "mensaje": (
            "AutoSklearn entrenado correctamente"
        ),
        "metrics": metrics,
        "model_info": model_info,
        "shap_config": meta["shap_config"],
        "artifacts": {
            "model_path": str(MODEL_PATH),
            "background_path": str(
                BACKGROUND_PATH
            ),
            "meta_path": str(META_PATH),
        },
    }


@app.post("/explain")
def explain(
    request: ExplainRequest,
):
    try:
        model, background, meta = (
            load_artifacts()
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    df_original = rows_to_dataframe(
        request.rows
    )
    total_rows_received = len(df_original)

    df = (
        df_original
        .head(SHAP_MAX_EXPLAIN_ROWS)
        .copy()
    )

    X = df[FEATURE_COLUMNS].copy()

    predict_fn = build_predict_fn(model)
    predict_proba_fn = (
        build_predict_proba_fn(model)
    )

    predictions = predict_fn(X)
    probabilities = np.asarray(
        predict_proba_fn(X),
        dtype=float,
    )

    classes = [
        str(class_name)
        for class_name in model.classes_
    ]

    start = time.time()

    explainer = shap.KernelExplainer(
        predict_proba_fn,
        background,
        link="identity",
    )

    shap_raw = explainer.shap_values(
        X,
        nsamples=SHAP_NSAMPLES,
        silent=True,
    )

    shap_seconds = round(
        time.time() - start,
        3,
    )

    shap_values = normalize_shap_values(
        shap_values=shap_raw,
        n_samples=len(X),
        n_features=len(FEATURE_COLUMNS),
        n_classes=len(classes),
    )

    expected_values = (
        normalize_expected_values(
            expected_value=(
                explainer.expected_value
            ),
            n_classes=len(classes),
        )
    )

    importancia_global = (
        build_global_importance(
            feature_columns=FEATURE_COLUMNS,
            shap_values=shap_values,
        )
    )

    importancia_por_clase = (
        build_global_importance_by_class(
            feature_columns=FEATURE_COLUMNS,
            shap_values=shap_values,
            classes=classes,
        )
    )

    validacion_aditividad = (
        build_additivity_diagnostics(
            expected_values=expected_values,
            shap_values=shap_values,
            model_outputs=probabilities,
            classes=classes,
        )
    )

    predicciones = []

    for row_index in range(len(X)):
        prediction = str(
            predictions[row_index]
        )

        if prediction not in classes:
            raise HTTPException(
                status_code=500,
                detail=(
                    "El modelo devolvió una clase "
                    f"desconocida: {prediction}."
                ),
            )

        class_index = classes.index(
            prediction
        )

        local_values = shap_values[
            row_index,
            :,
            class_index,
        ]

        contributions = (
            build_local_contributions(
                feature_columns=(
                    FEATURE_COLUMNS
                ),
                feature_values=(
                    X.iloc[row_index].tolist()
                ),
                shap_values=(
                    local_values.tolist()
                ),
            )
        )

        proba_row = probabilities[
            row_index
        ]

        probability_by_class = {
            classes[class_position]: float(
                proba_row[class_position]
            )
            for class_position in range(
                len(classes)
            )
        }

        reconstructed_probability = float(
            expected_values[class_index]
            + np.sum(local_values)
        )

        predicciones.append(
            {
                "publication_id": str(
                    df.iloc[row_index][
                        "publication_id"
                    ]
                ),
                "prediccion": prediction,
                "probabilidades": (
                    probability_by_class
                ),
                "explicacion_clase": prediction,
                "valor_base": float(
                    expected_values[class_index]
                ),
                "probabilidad_reconstruida": (
                    reconstructed_probability
                ),
                "error_aditividad": float(
                    abs(
                        reconstructed_probability
                        - proba_row[class_index]
                    )
                ),
                "top_contribuciones": (
                    contributions[
                        : request.top_n
                    ]
                ),
                "contribuciones": (
                    contributions
                ),
            }
        )

    return {
        "mensaje": (
            "Explicaciones generadas correctamente"
        ),
        "modelo": "AutoSklearnClassifier",
        "variables_explicadas": FEATURE_COLUMNS,
        "metrics_entrenamiento": (
            meta.get("metrics")
        ),
        "model_info": (
            meta.get("model_info")
        ),
        "shap_config": {
            "explainer": "KernelExplainer",
            "link": "identity",
            "background_rows": int(
                len(background)
            ),
            "background_random_state": (
                meta.get(
                    "shap_config",
                    {},
                ).get(
                    "background_random_state"
                )
            ),
            "nsamples": SHAP_NSAMPLES,
            "productos_recibidos": int(
                total_rows_received
            ),
            "productos_explicados": int(
                len(X)
            ),
            "limite_productos_explicados": (
                SHAP_MAX_EXPLAIN_ROWS
            ),
            "shap_seconds": shap_seconds,
            "truncated": (
                total_rows_received
                > len(X)
            ),
        },
        "valores_base_por_clase": {
            classes[class_index]: float(
                expected_values[class_index]
            )
            for class_index in range(
                len(classes)
            )
        },
        "validacion_aditividad": (
            validacion_aditividad
        ),
        "importancia_global": (
            importancia_global
        ),
        "importancia_por_clase": (
            importancia_por_clase
        ),
        "predicciones": predicciones,
    }