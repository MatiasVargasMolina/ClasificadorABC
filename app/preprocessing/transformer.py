import pandas as pd
from typing import Any, List, Tuple


# Se conservan para mostrarlas en la respuesta de la API.
COLUMNAS_BASE = [
    "publication_id",
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
    "en_promocion",
]


# Solo estas variables ingresan al clustering.
COLUMNAS_FEATURES = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
]


def producto_a_dict(producto: Any) -> dict:
    return {
        "publication_id": producto.publication_id,
        "ventas_30d": producto.ventas_30d,
        "visitas_30d": producto.visitas_30d,
        "precio_actual": producto.precio_actual,
        "stock_actual": producto.stock_actual,
        "en_promocion": getattr(
            producto,
            "en_promocion",
            False,
        ),
    }


def transformar_productos(
    productos: List[Any],
) -> pd.DataFrame:
    """
    Convierte los productos validados a un DataFrame completo.
    """
    registros = [
        producto_a_dict(producto)
        for producto in productos
    ]

    df = pd.DataFrame(registros)

    if df.empty:
        return pd.DataFrame(columns=COLUMNAS_BASE)

    return df[COLUMNAS_BASE].copy()


def transformar_variables(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Realiza transformaciones para la respuesta.

    en_promocion se conserva en la salida, pero no entra al modelo.
    """
    df_transformado = df.copy()

    if "en_promocion" in df_transformado.columns:
        df_transformado["en_promocion"] = (
            df_transformado["en_promocion"]
            .fillna(False)
            .astype(int)
        )

    return df_transformado


def obtener_features_modelo(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selecciona exclusivamente las variables utilizadas por el modelo.
    """
    faltantes = [
        columna
        for columna in COLUMNAS_FEATURES
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan variables para el modelo: {faltantes}"
        )

    return df[COLUMNAS_FEATURES].copy()


def preparar_datos_modelo(
    productos: List[Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:

    - df_transformado: datos completos para la respuesta.
    - X: ventas, visitas y precio para el clustering.
    """
    df = transformar_productos(productos)
    df_transformado = transformar_variables(df)
    X = obtener_features_modelo(df_transformado)

    return df_transformado, X