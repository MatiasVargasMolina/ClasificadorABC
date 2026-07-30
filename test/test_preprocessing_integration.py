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
            product("INVALID", 4, 2, 7990),
        ]
    )
    result = ejecutar_preprocesamiento(request)
    assert result["hay_validos"] is True
    assert len(result["productos_validos"]) == 3
    assert len(result["productos_invalidos"]) == 1
    assert result["productos_invalidos"][0]["publication_id"] == "INVALID"
    assert result["df_transformado"].shape == (3, 6)
    assert result["X_modelo"].columns.tolist() == [
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    ]
    assert np.allclose(result["X_modelo"].mean(), 0.0)
    assert result["normalizacion"].n_features_in_ == 3


def test_preprocessing_service_returns_explicit_contract_without_valid_rows():
    request = RequestInput(
        productos=[
            product("INVALID", 5, 2, 7990),
        ]
    )
    result = ejecutar_preprocesamiento(request)
    assert result == {
        "hay_validos": False,
        "mensaje": "No hay productos válidos para clasificar",
        "productos_validos": [],
        "productos_invalidos": [
            {
                "publication_id": "INVALID",
                "errores": [
                    "inconsistencia: visitas_30d menores que ventas_30d"
                ],
            }
        ],
        "df_transformado": None,
        "X": None,
        "X_modelo": None,
        "normalizacion": None,
    }
