# app/api/routes/optimization.py

from fastapi import APIRouter
from app.schemas.input_schema import RequestInput
from app.services.optimization_service import OptimizationService

router = APIRouter()

@router.post("/optuna")
def optimize_with_optuna(request: RequestInput):
    service = OptimizationService()
    return service.optimize(request)