from types import SimpleNamespace

import pandas as pd
import pytest

from app.preprocessing.transformer import (
    COLUMNAS_BASE,
    COLUMNAS_FEATURES,
    obtener_features_modelo,
    preparar_datos_modelo,
    producto_a_dict,
    transformar_productos,
    transformar_variables,
)
from app.schemas.input_schema import ProductoInput


def make_product(publication_id="MLC-1", promotion=True):
    return ProductoInput(
        publication_id=publication_id,
        ventas_30d=2,
        visitas_30d=10,
        precio_actual=9990.0,
        stock_actual=5,
        en_promocion=promotion,
    )


def test_producto_a_dict_keeps_operational_fields_without_old_label():
    result = producto_a_dict(make_product())
    assert result == {
        "publication_id": "MLC-1",
        "ventas_30d": 2,
        "visitas_30d": 10,
        "precio_actual": 9990.0,
        "stock_actual": 5,
        "en_promocion": True,
    }


def test_producto_a_dict_defaults_missing_promotion_to_false():
    product = SimpleNamespace(
        publication_id="MLC-1",
        ventas_30d=0,
        visitas_30d=0,
        precio_actual=1000.0,
        stock_actual=1,
    )
    assert producto_a_dict(product)["en_promocion"] is False


def test_transformar_productos_has_stable_column_order_and_empty_contract():
    frame = transformar_productos([make_product()])
    assert frame.columns.tolist() == COLUMNAS_BASE
    empty = transformar_productos([])
    assert empty.empty
    assert empty.columns.tolist() == COLUMNAS_BASE


def test_transformar_variables_converts_promotion_to_integer():
    transformed = transformar_variables(
        pd.DataFrame(
            {
                "en_promocion": [True, False, None],
                "publication_id": ["1", "2", "3"],
            }
        )
    )
    assert transformed["en_promocion"].tolist() == [1, 0, 0]


def test_obtener_features_modelo_excludes_stock_and_promotion():
    frame = transformar_variables(transformar_productos([make_product()]))
    features = obtener_features_modelo(frame)
    assert features.columns.tolist() == COLUMNAS_FEATURES
    assert COLUMNAS_FEATURES == [
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    ]


def test_obtener_features_modelo_rejects_missing_feature():
    with pytest.raises(ValueError, match="Faltan variables"):
        obtener_features_modelo(
            pd.DataFrame(
                {
                    "ventas_30d": [1],
                    "visitas_30d": [2],
                }
            )
        )


def test_preparar_datos_modelo_returns_full_frame_and_three_features():
    transformed, features = preparar_datos_modelo(
        [make_product("MLC-1"), make_product("MLC-2", False)]
    )
    assert transformed.columns.tolist() == COLUMNAS_BASE
    assert features.columns.tolist() == COLUMNAS_FEATURES
    assert len(transformed) == len(features) == 2
