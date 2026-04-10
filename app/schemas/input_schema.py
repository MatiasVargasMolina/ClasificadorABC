from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional


class ProductoInput(BaseModel):
    """
    Representa una publicación individual dentro del sistema.
    """

    publication_id: str = Field(..., min_length=1)
    ventas_30d: int = Field(..., ge=0)
    visitas_30d: int = Field(..., ge=0)
    precio_actual: float = Field(..., gt=0)
    stock_actual: int = Field(..., ge=0)

    en_promocion: Optional[bool] = None
    etiqueta_abc_opcional: Optional[str] = None

    # -------------------------
    # VALIDADORES DE CAMPO
    # -------------------------

    @field_validator("publication_id")
    @classmethod
    def limpiar_publication_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("publication_id no puede estar vacío")
        return v

    @field_validator("etiqueta_abc_opcional")
    @classmethod
    def validar_etiqueta(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v

        v = v.strip().upper()

        if v == "":
            return None

        if v not in {"A", "B", "C"}:
            raise ValueError("etiqueta_abc_opcional debe ser A, B o C")

        return v

    @field_validator("en_promocion")
    @classmethod
    def default_en_promocion(cls, v: Optional[bool]) -> bool:
        # define comportamiento por defecto
        return False if v is None else v


class RequestInput(BaseModel):
    """
    Representa la solicitud completa que contiene múltiples publicaciones.
    """

    productos: List[ProductoInput] = Field(..., min_length=1)

    # -------------------------
    # VALIDADORES DE CONJUNTO
    # -------------------------

    @model_validator(mode="after")
    def validar_unicidad_ids(self):
        ids = [p.publication_id for p in self.productos]
        duplicados = set([x for x in ids if ids.count(x) > 1])

        if duplicados:
            raise ValueError(f"publication_id duplicados: {duplicados}")

        return self