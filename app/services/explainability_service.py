from __future__ import annotations

import os
from typing import Any, Dict

import requests

from app.ml.explainability.surrogate_payload import (
    build_predict_rows_from_request,
    build_train_rows_from_classification,
)
from app.services.clasificacion_service import ejecutar_clasificacion


AUTOSKLEARN_WORKER_URL = os.getenv(
    "AUTOSKLEARN_WORKER_URL",
    "http://127.0.0.1:8010",
)


def _post_to_worker(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{AUTOSKLEARN_WORKER_URL}{path}"

    try:
        response = requests.post(url, json=payload, timeout=900)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"No se pudo conectar con el worker AutoSklearn en {url}. "
            f"Verifica que Docker esté corriendo. Detalle: {exc}"
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Error desde worker AutoSklearn ({response.status_code}): "
            f"{response.text}"
        )

    return response.json()


def _get_from_worker(path: str) -> Dict[str, Any]:
    url = f"{AUTOSKLEARN_WORKER_URL}{path}"

    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"No se pudo conectar con el worker AutoSklearn en {url}. "
            f"Verifica que Docker esté corriendo. Detalle: {exc}"
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Error desde worker AutoSklearn ({response.status_code}): "
            f"{response.text}"
        )

    return response.json()


def health_explainability_worker() -> Dict[str, Any]:
    return _get_from_worker("/health")


def entrenar_surrogate_autosklearn(
    data: Any,
    time_left_for_this_task: int = 120,
    per_run_time_limit: int = 30,
) -> Dict[str, Any]:
    """
    Flujo:
    1. Ejecuta clasificación SS-E-KMeans.
    2. Usa categoria A/B/C como target supervisado.
    3. Envía dataset al worker Docker con AutoSklearn.
    """

    clasificacion_result = ejecutar_clasificacion(data)
    resultados = clasificacion_result.get("resultados", [])

    if not resultados:
        return {
            "mensaje": "No hay resultados válidos para entrenar el modelo sustituto",
            "clasificacion": clasificacion_result,
        }

    train_rows = build_train_rows_from_classification(clasificacion_result)

    payload = {
        "rows": train_rows,
        "time_left_for_this_task": time_left_for_this_task,
        "per_run_time_limit": per_run_time_limit,
    }

    worker_result = _post_to_worker("/train", payload)

    return {
        "mensaje": "Surrogate AutoSklearn entrenado desde etiquetas SS-E-KMeans",
        "diagnostico_clasificacion": clasificacion_result.get("diagnostico"),
        "productos_invalidos": clasificacion_result.get("productos_invalidos", []),
        "autosklearn": worker_result,
    }


def explicar_con_surrogate_autosklearn(
    data: Any,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Usa el surrogate AutoSklearn ya entrenado para predecir y explicar con SHAP.
    """

    predict_rows = build_predict_rows_from_request(data)

    payload = {
        "rows": predict_rows,
        "top_n": top_n,
    }

    return _post_to_worker("/explain", payload)