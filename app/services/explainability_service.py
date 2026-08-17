from __future__ import annotations

import os
from typing import Any, Dict

import requests

from app.ml.assignment.constrained_assignment import MetodoAsignacion
from app.ml.explainability.surrogate_payload import (
    build_predict_rows_from_request,
    build_train_rows_from_classification,
)
from app.services.clasificacion_service import (
    ejecutar_clasificacion,
)


AUTOSKLEARN_WORKER_URL = os.getenv(
    "AUTOSKLEARN_WORKER_URL",
    "http://127.0.0.1:8010",
)

TRAIN_TIMEOUT_SECONDS = int(
    os.getenv(
        "AUTOSKLEARN_TRAIN_TIMEOUT_SECONDS",
        "1200",
    )
)

EXPLAIN_TIMEOUT_SECONDS = int(
    os.getenv(
        "AUTOSKLEARN_EXPLAIN_TIMEOUT_SECONDS",
        "3600",
    )
)


def _post_to_worker(
    path: str,
    payload: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    url = f"{AUTOSKLEARN_WORKER_URL}{path}"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=None,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"El worker AutoSklearn superó el tiempo máximo "
            f"de {timeout_seconds} segundos en {url}."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"No se pudo conectar con el worker AutoSklearn en {url}. "
            f"Verifica que Docker esté corriendo. Detalle: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Error desde worker AutoSklearn "
            f"({response.status_code}): {response.text}"
        )

    return response.json()


def _get_from_worker(
    path: str,
) -> Dict[str, Any]:
    url = f"{AUTOSKLEARN_WORKER_URL}{path}"

    try:
        response = requests.get(
            url,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"No se pudo conectar con el worker AutoSklearn en {url}. "
            f"Verifica que Docker esté corriendo. Detalle: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Error desde worker AutoSklearn "
            f"({response.status_code}): {response.text}"
        )

    return response.json()


def health_explainability_worker() -> Dict[str, Any]:
    return _get_from_worker(
        "/health"
    )


def entrenar_surrogate_autosklearn(
    data: Any,
    time_left_for_this_task: int = 600,
    per_run_time_limit: int = 60,
    metodo_asignacion: MetodoAsignacion = "global",
) -> Dict[str, Any]:
    """
    Ejecuta SS-E-KMeans con el método de asignación seleccionado,
    utiliza sus categorías como target y entrena el modelo sustituto
    AutoSklearn.
    """
    clasificacion_result = ejecutar_clasificacion(
        data,
        metodo_asignacion=metodo_asignacion,
    )

    resultados = clasificacion_result.get(
        "resultados",
        [],
    )

    if not resultados:
        return {
            "mensaje": (
                "No hay resultados válidos para entrenar "
                "el modelo sustituto"
            ),
            "clasificacion": clasificacion_result,
        }

    train_rows = build_train_rows_from_classification(
        clasificacion_result
    )

    payload = {
        "rows": train_rows,
        "time_left_for_this_task": (
            time_left_for_this_task
        ),
        "per_run_time_limit": (
            per_run_time_limit
        ),
    }

    worker_result = _post_to_worker(
        path="/train",
        payload=payload,
        timeout_seconds=max(
            TRAIN_TIMEOUT_SECONDS,
            time_left_for_this_task + 300,
        ),
    )

    return {
        "mensaje": (
            "Surrogate AutoSklearn entrenado desde "
            "etiquetas SS-E-KMeans"
        ),
        "metodo_asignacion_utilizado": (
            clasificacion_result[
                "metodo_asignacion_utilizado"
            ]
        ),
        "diagnostico_clasificacion": (
            clasificacion_result.get(
                "diagnostico"
            )
        ),
        "productos_invalidos": (
            clasificacion_result.get(
                "productos_invalidos",
                [],
            )
        ),
        "autosklearn": worker_result,
    }


def explicar_con_surrogate_autosklearn(
    data: Any,
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    Usa el modelo sustituto entrenado para predecir y explicar.
    """
    predict_rows = build_predict_rows_from_request(
        data
    )

    payload = {
        "rows": predict_rows,
        "top_n": min(
            top_n,
            3,
        ),
    }

    return _post_to_worker(
        path="/explain",
        payload=payload,
        timeout_seconds=EXPLAIN_TIMEOUT_SECONDS,
    )