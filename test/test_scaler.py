import pytest
import pandas as pd
from app.preprocessing.scaler import ajustar_scaler, transformar_con_scaler, ajustar_y_transformar


def crear_df_features():
    return pd.DataFrame([
        {
            "ventas_30d": 10,
            "visitas_30d": 100,
            "precio_actual": 15990,
            "stock_actual": 5,
            "en_promocion": 1
        },
        {
            "ventas_30d": 2,
            "visitas_30d": 40,
            "precio_actual": 8990,
            "stock_actual": 10,
            "en_promocion": 0
        }
    ])


def test_ajustar_scaler_retorna_scaler():
    df = crear_df_features()
    scaler = ajustar_scaler(df)

    assert scaler is not None


def test_transformar_con_scaler_mantiene_en_promocion():
    df = crear_df_features()
    scaler = ajustar_scaler(df)
    df_escalado = transformar_con_scaler(df, scaler)

    assert df_escalado["en_promocion"].tolist() == [1, 0]


def test_transformar_con_scaler_modifica_variables_numericas():
    df = crear_df_features()
    scaler = ajustar_scaler(df)
    df_escalado = transformar_con_scaler(df, scaler)

    assert df_escalado["ventas_30d"].tolist() != df["ventas_30d"].tolist()


def test_ajustar_y_transformar_retorna_dataframe_y_scaler():
    df = crear_df_features()
    df_escalado, scaler = ajustar_y_transformar(df)

    assert df_escalado.shape == df.shape
    assert scaler is not None


def test_falla_si_faltan_columnas():
    df = pd.DataFrame([
        {
            "ventas_30d": 10,
            "visitas_30d": 100,
            "stock_actual": 5,
            "en_promocion": 1
        }
    ])

    with pytest.raises(ValueError, match="Faltan columnas para escalar"):
        ajustar_scaler(df)