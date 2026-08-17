from fastapi import APIRouter, HTTPException, Query

from app.ml.assignment.constrained_assignment import MetodoAsignacion
from app.schemas.input_schema import RequestInput
from app.services.explainability_service import (
    entrenar_surrogate_autosklearn,
    explicar_con_surrogate_autosklearn,
    health_explainability_worker,
)


router = APIRouter()


@router.get("/explainability/health")
def explainability_health():
    try:
        return health_explainability_worker()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.post("/explainability/autosklearn/train")
def train_surrogate_autosklearn(
    data: RequestInput,
    time_left_for_this_task: int = Query(
        default=600,
        ge=30,
        le=1800,
    ),
    per_run_time_limit: int = Query(
        default=60,
        ge=10,
        le=300,
    ),
    metodo_asignacion: MetodoAsignacion = Query(
        default="global",
        description=(
            "Método de asignación utilizado por SS-EKMeans "
            "para generar las etiquetas de entrenamiento. "
            "Valores permitidos: global o secuencial."
        ),
    ),
):
    try:
        return entrenar_surrogate_autosklearn(
            data=data,
            time_left_for_this_task=(
                time_left_for_this_task
            ),
            per_run_time_limit=(
                per_run_time_limit
            ),
            metodo_asignacion=metodo_asignacion,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.post("/explainability/autosklearn/explain")
def explain_surrogate_autosklearn(
    data: RequestInput,
    top_n: int = Query(
        default=3,
        ge=1,
        le=3,
    ),
):
    try:
        return explicar_con_surrogate_autosklearn(
            data=data,
            top_n=top_n,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc