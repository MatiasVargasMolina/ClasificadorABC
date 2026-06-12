from __future__ import annotations

from typing import Any, Dict, List


FEATURE_COLUMNS = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
    "en_promocion",
]


def build_train_rows_from_classification(
    clasificacion_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Convierte la salida de ejecutar_clasificacion() en filas supervisadas
    para entrenar el modelo sustituto AutoSklearn.

    El target supervisado será la categoria generada por SS-E-KMeans:
    A, B o C.
    """

    rows: List[Dict[str, Any]] = []

    for item in clasificacion_result.get("resultados", []):
        rows.append(
            {
                "publication_id": item["publication_id"],
                "ventas_30d": int(item["ventas_30d"]),
                "visitas_30d": int(item["visitas_30d"]),
                "precio_actual": float(item["precio_actual"]),
                "stock_actual": int(item["stock_actual"]),
                "en_promocion": int(item["en_promocion"]),
                "categoria": str(item["categoria"]),
            }
        )

    return rows


def build_predict_rows_from_request(data: Any) -> List[Dict[str, Any]]:
    """
    Convierte el RequestInput original en filas para que el worker
    AutoSklearn pueda predecir y explicar con SHAP.
    """

    rows: List[Dict[str, Any]] = []

    for producto in data.productos:
        rows.append(
            {
                "publication_id": producto.publication_id,
                "ventas_30d": int(producto.ventas_30d),
                "visitas_30d": int(producto.visitas_30d),
                "precio_actual": float(producto.precio_actual),
                "stock_actual": int(producto.stock_actual),
                "en_promocion": int(producto.en_promocion),
            }
        )

    return rows