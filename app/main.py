from fastapi import FastAPI
from app.api.routes.clasificacion import router as clasificacion_router

app = FastAPI(title="ABC Microservice")

app.include_router(clasificacion_router, prefix="/api")