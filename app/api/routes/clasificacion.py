from fastapi import APIRouter, Query

from app.ml.assignment.constrained_assignment import MetodoAsignacion
from app.schemas.input_schema import RequestInput
from app.services.clasificacion_service import ejecutar_clasificacion


router = APIRouter()


@router.post("/clasificar")
def clasificar(
    data: RequestInput,
    metodo_asignacion: MetodoAsignacion = Query(
        default="global",
        description=(
            "Método de asignación de capacidad utilizado por SS-EKMeans. "
            "Valores permitidos: global o secuencial."
        ),
    ),
):
    return ejecutar_clasificacion(
        data,
        metodo_asignacion=metodo_asignacion,
    )