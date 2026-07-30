import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from app.preprocessing.scaler import (
    COLUMNAS_ESCALABLES,
    ajustar_scaler,
    ajustar_y_transformar,
    preparar_variables,
    transformar_con_scaler,
)


def feature_frame():
    return pd.DataFrame(
        {
            "ventas_30d": [0, 1, 9],
            "visitas_30d": [0, 9, 99],
            "precio_actual": [1000.0, 2000.0, 4000.0],
        }
    )


def test_preparar_variables_applies_log1p_only_to_counts():
    prepared = preparar_variables(feature_frame())
    assert prepared["ventas_30d"].tolist() == pytest.approx(
        np.log1p([0, 1, 9])
    )
    assert prepared["visitas_30d"].tolist() == pytest.approx(
        np.log1p([0, 9, 99])
    )
    assert prepared["precio_actual"].tolist() == [1000.0, 2000.0, 4000.0]


def test_ajustar_y_transformar_standardizes_three_model_features():
    scaled, scaler = ajustar_y_transformar(feature_frame())
    assert scaled.columns.tolist() == COLUMNAS_ESCALABLES
    assert scaled.mean().to_numpy() == pytest.approx([0.0, 0.0, 0.0])
    assert scaler.n_features_in_ == 3


def test_transformar_con_existing_scaler_is_reproducible():
    source = feature_frame()
    scaler = ajustar_scaler(source)
    first = transformar_con_scaler(source, scaler)
    second = transformar_con_scaler(source.copy(), scaler)
    pdt.assert_frame_equal(first, second)


def test_scaler_preserves_extra_columns_but_does_not_scale_them():
    source = feature_frame().assign(stock_actual=[1, 2, 3])
    scaled, _ = ajustar_y_transformar(source)
    assert scaled["stock_actual"].tolist() == [1, 2, 3]


def test_scaler_rejects_missing_model_feature():
    with pytest.raises(ValueError, match="Faltan columnas"):
        ajustar_y_transformar(feature_frame().drop(columns=["precio_actual"]))


def test_scaler_handles_constant_feature_without_nan_or_infinity():
    source = feature_frame()
    source["precio_actual"] = 1000.0
    scaled, _ = ajustar_y_transformar(source)
    assert np.isfinite(scaled.to_numpy()).all()
