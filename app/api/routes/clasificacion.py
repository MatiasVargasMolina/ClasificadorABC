from fastapi import APIRouter
from app.schemas.input_schema import RequestInput
from app.services.clasificacion_service import ejecutar_clasificacion

router = APIRouter()


@router.post("/clasificar")
def clasificar(data: RequestInput):
    return ejecutar_clasificacion(data)