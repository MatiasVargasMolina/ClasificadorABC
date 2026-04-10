from fastapi import FastAPI
from app.schemas import RequestInput

app = FastAPI()

@app.post("/clasificar")
def clasificar(data: RequestInput):
    return {
        "status": "ok",
        "cantidad_productos": len(data.productos)
    }