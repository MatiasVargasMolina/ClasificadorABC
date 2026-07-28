from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


COLUMNAS_LOGARITMICAS = [
    "ventas_30d",
    "visitas_30d",
]

COLUMNAS_ESCALABLES = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
]


def validar_columnas(df_features: pd.DataFrame) -> None:
    faltantes = [
        columna
        for columna in COLUMNAS_ESCALABLES
        if columna not in df_features.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas para escalar: {faltantes}"
        )


def preparar_variables(
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aplica log1p a las variables de conteo antes de estandarizar.
    """
    validar_columnas(df_features)

    df_preparado = df_features.copy()

    df_preparado[COLUMNAS_LOGARITMICAS] = np.log1p(
        df_preparado[COLUMNAS_LOGARITMICAS].astype(float)
    )

    return df_preparado


def ajustar_scaler(
    df_features: pd.DataFrame,
) -> StandardScaler:
    """
    Ajusta StandardScaler después de aplicar log1p a ventas y visitas.
    """
    df_preparado = preparar_variables(df_features)

    scaler = StandardScaler()
    scaler.fit(
        df_preparado[COLUMNAS_ESCALABLES].astype(float)
    )

    return scaler


def transformar_con_scaler(
    df_features: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Aplica log1p y luego la estandarización.
    """
    df_preparado = preparar_variables(df_features)
    df_escalado = df_preparado.copy()

    df_escalado[COLUMNAS_ESCALABLES] = scaler.transform(
        df_preparado[COLUMNAS_ESCALABLES].astype(float)
    )

    return df_escalado


def ajustar_y_transformar(
    df_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Ajusta StandardScaler y transforma las variables del modelo.
    """
    scaler = ajustar_scaler(df_features)

    df_escalado = transformar_con_scaler(
        df_features,
        scaler,
    )

    return df_escalado, scaler