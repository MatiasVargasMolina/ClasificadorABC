import pandas as pd
from typing import Any, List, Tuple


COLUMNAS_BASE = [
    "publication_id",
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
    "en_promocion",
    "etiqueta_abc_opcional",
]

COLUMNAS_FEATURES = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
    "en_promocion",
]


def producto_a_dict(producto: Any) -> dict:
    return {
        "publication_id": producto.publication_id,
        "ventas_30d": producto.ventas_30d,
        "visitas_30d": producto.visitas_30d,
        "precio_actual": producto.precio_actual,
        "stock_actual": producto.stock_actual,
        "en_promocion": producto.en_promocion,
        "etiqueta_abc_opcional": producto.etiqueta_abc_opcional,
    }


def transformar_productos(productos: List[Any]) -> pd.DataFrame:
    """
    Convierte una lista de productos validados a DataFrame.
    """
    registros = [producto_a_dict(producto) for producto in productos]
    df = pd.DataFrame(registros)

    if df.empty:
        return pd.DataFrame(columns=COLUMNAS_BASE)

    return df[COLUMNAS_BASE].copy()


def transformar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza transformaciones necesarias para el modelo.
    """
    df_transformado = df.copy()

    # convertir bool a int
    df_transformado["en_promocion"] = df_transformado["en_promocion"].astype(int)

    return df_transformado


def obtener_features_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna solo las variables que entran al modelo.
    """
    return df[COLUMNAS_FEATURES].copy()


def preparar_datos_modelo(productos: List[Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flujo completo de transformación para el modelo.

    Retorna:
    - df_transformado: DataFrame completo transformado
    - X: features que entran al modelo
    """
    df = transformar_productos(productos)
    df_transformado = transformar_variables(df)
    X = obtener_features_modelo(df_transformado)

    return df_transformado, X