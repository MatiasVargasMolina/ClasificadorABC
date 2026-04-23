from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


ABC_VALIDAS = {"A", "B", "C"}


class ProductoInput(BaseModel):
    """
    Representa una publicación individual de entrada.
    """

    publication_id: str = Field(..., min_length=1)
    ventas_30d: int = Field(..., ge=0)
    visitas_30d: int = Field(..., ge=0)
    precio_actual: float = Field(..., gt=0)
    stock_actual: int = Field(..., ge=0)

    en_promocion: bool = False
    etiqueta_abc_opcional: Optional[str] = None

    @field_validator("publication_id")
    @classmethod
    def validar_publication_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("publication_id no puede estar vacío")
        return v

    @field_validator("etiqueta_abc_opcional")
    @classmethod
    def validar_etiqueta_abc_opcional(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None

        v = v.strip().upper()

        if not v:
            return None

        if v not in ABC_VALIDAS:
            raise ValueError("etiqueta_abc_opcional debe ser A, B o C")

        return v


class RequestInput(BaseModel):
    """
    Representa la solicitud completa de clasificación.
    """

    productos: List[ProductoInput] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validar_unicidad_ids(self):
        ids = [producto.publication_id for producto in self.productos]
        ids_vistos = set()
        duplicados = set()

        for publication_id in ids:
            if publication_id in ids_vistos:
                duplicados.add(publication_id)
            ids_vistos.add(publication_id)

        if duplicados:
            duplicados_str = ", ".join(sorted(duplicados))
            raise ValueError(f"publication_id duplicados: {duplicados_str}")

        return self