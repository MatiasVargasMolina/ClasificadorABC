from fastapi import FastAPI
from app.api.routes.clasificacion import router as clasificacion_router
from app.api.routes.optimization import router as optimization_router
from app.api.routes.explainability import router as explainability_router

app = FastAPI(title="ABC Microservice")

app.include_router(clasificacion_router, prefix="/api")
app.include_router(optimization_router,prefix="/optimization",tags=["Optimization"])
app.include_router(
    explainability_router,
    prefix="/api",
    tags=["Explainability"],
)
