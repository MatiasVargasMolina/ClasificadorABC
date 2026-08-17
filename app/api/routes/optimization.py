from fastapi import APIRouter, Query

from app.ml.assignment.constrained_assignment import MetodoAsignacion
from app.schemas.input_schema import RequestInput
from app.services.optimization_service import OptimizationService


router = APIRouter()


@router.post("/optuna")
def optimize_with_optuna(
    request: RequestInput,
    metodo_asignacion: MetodoAsignacion = Query(
        default="global",
        description=(
            "Método de asignación utilizado por SS-EKMeans "
            "durante la optimización. "
            "Valores permitidos: global o secuencial."
        ),
    ),
):
    service = OptimizationService()

    return service.optimize(
        request,
        metodo_asignacion=metodo_asignacion,
    )