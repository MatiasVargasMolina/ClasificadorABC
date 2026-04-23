import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


COLUMNAS_ESCALABLES = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
]


def validar_columnas(df_features: pd.DataFrame) -> None:
    faltantes = [col for col in COLUMNAS_ESCALABLES if col not in df_features.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas para escalar: {faltantes}")


def ajustar_scaler(df_features: pd.DataFrame) -> StandardScaler:
    """
    Ajusta un StandardScaler usando las variables numéricas escalables.
    """
    validar_columnas(df_features)

    scaler = StandardScaler()
    scaler.fit(df_features[COLUMNAS_ESCALABLES])

    return scaler


def transformar_con_scaler(
    df_features: pd.DataFrame,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Aplica el scaler a las variables numéricas.
    """
    validar_columnas(df_features)

    df_escalado = df_features.copy()

    df_escalado[COLUMNAS_ESCALABLES] = scaler.transform(
        df_escalado[COLUMNAS_ESCALABLES].astype(float)
    )

    return df_escalado


def ajustar_y_transformar(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Ajusta el scaler y transforma el DataFrame en un solo paso.
    """
    scaler = ajustar_scaler(df_features)
    df_escalado = transformar_con_scaler(df_features, scaler)

    return df_escalado, scaler