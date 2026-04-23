import pytest
from pydantic import ValidationError
from app.schemas.input_schema import RequestInput


def test_request_input_valido():
    data = {
        "productos": [
            {
                "publication_id": " MLC123 ",
                "ventas_30d": 10,
                "visitas_30d": 100,
                "precio_actual": 15990,
                "stock_actual": 5,
                "etiqueta_abc_opcional": "a"
            }
        ]
    }

    request = RequestInput(**data)

    assert request.productos[0].publication_id == "MLC123"
    assert request.productos[0].en_promocion is False
    assert request.productos[0].etiqueta_abc_opcional == "A"


def test_rechaza_precio_invalido():
    data = {
        "productos": [
            {
                "publication_id": "MLC123",
                "ventas_30d": 10,
                "visitas_30d": 100,
                "precio_actual": 0,
                "stock_actual": 5
            }
        ]
    }

    with pytest.raises(ValidationError):
        RequestInput(**data)


def test_rechaza_ids_duplicados():
    data = {
        "productos": [
            {
                "publication_id": "MLC123",
                "ventas_30d": 10,
                "visitas_30d": 100,
                "precio_actual": 15990,
                "stock_actual": 5
            },
            {
                "publication_id": "MLC123",
                "ventas_30d": 2,
                "visitas_30d": 40,
                "precio_actual": 8990,
                "stock_actual": 10
            }
        ]
    }

    with pytest.raises(ValidationError):
        RequestInput(**data)


def test_rechaza_etiqueta_invalida():
    data = {
        "productos": [
            {
                "publication_id": "MLC123",
                "ventas_30d": 10,
                "visitas_30d": 100,
                "precio_actual": 15990,
                "stock_actual": 5,
                "etiqueta_abc_opcional": "D"
            }
        ]
    }

    with pytest.raises(ValidationError):
        RequestInput(**data)