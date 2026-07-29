from __future__ import annotations

from typing import Any, Dict, List


FEATURE_COLUMNS = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
]


def _build_feature_values(
    item: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ventas_30d": int(item["ventas_30d"]),
        "visitas_30d": int(item["visitas_30d"]),
        "precio_actual": float(item["precio_actual"]),
    }


def build_train_rows_from_classification(
    clasificacion_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Convierte la salida de SS-E-KMeans en filas supervisadas.

    AutoSklearn aprende a imitar la categoría A, B o C utilizando
    exactamente las mismas variables consideradas por el clasificador.
    """
    rows: List[Dict[str, Any]] = []

    for item in clasificacion_result.get("resultados", []):
        row = {
            "publication_id": str(item["publication_id"]),
            **_build_feature_values(item),
            "categoria": str(item["categoria"]),
        }
        rows.append(row)

    return rows


def build_predict_rows_from_request(
    data: Any,
) -> List[Dict[str, Any]]:
    """
    Convierte RequestInput en filas para predecir y explicar.

    stock_actual y en_promocion pueden permanecer en el contrato de la
    API principal, pero no se envían al modelo sustituto porque no forman
    parte del clasificador ABC definitivo.
    """
    rows: List[Dict[str, Any]] = []

    for producto in data.productos:
        rows.append(
            {
                "publication_id": str(producto.publication_id),
                "ventas_30d": int(producto.ventas_30d),
                "visitas_30d": int(producto.visitas_30d),
                "precio_actual": float(producto.precio_actual),
            }
        )

    return rows