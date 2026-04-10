from pydantic import BaseModel, Field, field_validator
from typing import Dict, List


class ResultadoClasificacion(BaseModel):
    """
    Resultado individual de clasificación para una publicación.
    """

    publication_id: str = Field(..., min_length=1)
    categoria: str = Field(..., description="Categoría ABC asignada a la publicación")
    contribuciones: Dict[str, float] = Field(
        ...,
        description="Contribución de cada variable en la clasificación final"
    )

    @field_validator("publication_id")
    @classmethod
    def validar_publication_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("publication_id no puede estar vacío")
        return v

    @field_validator("categoria")
    @classmethod
    def validar_categoria(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in {"A", "B", "C"}:
            raise ValueError("categoria debe ser A, B o C")
        return v

    @field_validator("contribuciones")
    @classmethod
    def validar_contribuciones(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("contribuciones no puede estar vacío")
        return v


class ResponseOutput(BaseModel):
    """
    Respuesta completa del endpoint de clasificación.
    """

    resultados: List[ResultadoClasificacion] = Field(..., min_length=1)