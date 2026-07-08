from typing import Dict, Tuple

import pandas as pd


COLUMNAS_NORMALIZABLES = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
]


def validar_columnas(df_features: pd.DataFrame) -> None:
    faltantes = [
        col for col in COLUMNAS_NORMALIZABLES
        if col not in df_features.columns
    ]

    if faltantes:
        raise ValueError(f"Faltan columnas para normalizar: {faltantes}")


def normalizar_minmax(
    df_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normaliza variables numéricas usando Min-Max.

    Fórmula:
        x_norm = (x - min) / (max - min)

    Si una columna tiene rango 0, se deja en 0.0 para evitar división por cero.
    """

    validar_columnas(df_features)

    df_normalizado = df_features.copy()
    stats: Dict[str, Dict[str, float]] = {}

    for col in COLUMNAS_NORMALIZABLES:
        col_min = float(df_normalizado[col].min())
        col_max = float(df_normalizado[col].max())
        col_range = col_max - col_min

        if col_range == 0:
            df_normalizado[col] = 0.0
        else:
            df_normalizado[col] = (
                df_normalizado[col].astype(float) - col_min
            ) / col_range

        stats[col] = {
            "min": col_min,
            "max": col_max,
            "range": col_range,
        }

    return df_normalizado, stats