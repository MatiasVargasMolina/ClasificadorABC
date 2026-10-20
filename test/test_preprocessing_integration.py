import numpy as np

from app.schemas.input_schema import RequestInput
from app.services.preprocessing_service import ejecutar_preprocesamiento


def product(publication_id, sales, visits, price):
    return {
        "publication_id": publication_id,
        "ventas_30d": sales,
        "visitas_30d": visits,
        "precio_actual": price,
        "stock_actual": 10,
        "en_promocion": False,
    }


def test_preprocessing_service_integrates_validation_transform_and_scaler():
    request = RequestInput(
        productos=[
            product("A", 10, 100, 15990),
            product("B", 2, 40, 8990),
            product("C", 0, 5, 4990),
            product("MULTI", 4, 2, 7990),
        ]
    )
    result = ejecutar_preprocesamiento(request)
    assert result["hay_validos"] is True
    assert len(result["productos_validos"]) == 4
    assert result["productos_invalidos"] == []
    assert result["df_transformado"].shape == (4, 6)
    assert result["X_modelo"].columns.tolist() == [
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    ]
    assert np.allclose(result["X_modelo"].mean(), 0.0)
    assert result["normalizacion"].n_features_in_ == 3


def test_preprocessing_retains_units_above_visits_without_altering_counts():
    request = RequestInput(productos=[
        product("MULTI", 2, 1, 7990),
        product("ZERO_VISITS", 2, 0, 5990),
    ])
    result = ejecutar_preprocesamiento(request)
    assert result['hay_validos'] is True
    assert result['productos_invalidos'] == []
    assert result['X']['ventas_30d'].tolist() == [2, 2]
    assert result['X']['visitas_30d'].tolist() == [1, 0]
    assert np.isfinite(result['X_modelo'].to_numpy()).all()
