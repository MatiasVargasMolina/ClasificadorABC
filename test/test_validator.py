from app.schemas.input_schema import RequestInput
from app.preprocessing.validator import validar_producto, validar_productos


def crear_producto(**overrides):
    data = {
        "productos": [
            {
                "publication_id": "MLC123",
                "ventas_30d": 10,
                "visitas_30d": 100,
                "precio_actual": 15990,
                "stock_actual": 5,
                "en_promocion": True
            }
        ]
    }

    data["productos"][0].update(overrides)
    request = RequestInput(**data)
    return request.productos[0]


def test_validar_producto_sin_errores():
    producto = crear_producto()
    errores = validar_producto(producto)

    assert errores == []


def test_detecta_ventas_sin_visitas():
    producto = crear_producto(ventas_30d=5, visitas_30d=0)
    errores = validar_producto(producto)

    assert "inconsistencia: ventas_30d > 0 pero visitas_30d = 0" in errores


def test_detecta_visitas_menores_que_ventas():
    producto = crear_producto(ventas_30d=10, visitas_30d=5)
    errores = validar_producto(producto)

    assert "inconsistencia: visitas_30d menores que ventas_30d" in errores


def test_validar_productos_separa_validos_e_invalidos():
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
                "publication_id": "MLC124",
                "ventas_30d": 5,
                "visitas_30d": 0,
                "precio_actual": 12990,
                "stock_actual": 8
            }
        ]
    }

    request = RequestInput(**data)
    resultado = validar_productos(request.productos)

    assert len(resultado["validos"]) == 1
    assert len(resultado["invalidos"]) == 1
    assert resultado["invalidos"][0]["publication_id"] == "MLC124"