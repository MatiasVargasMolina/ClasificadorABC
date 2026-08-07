from app.preprocessing.validator import validar_producto, validar_productos
from app.schemas.input_schema import ProductoInput


def make_product(publication_id, sales, visits):
    return ProductoInput(
        publication_id=publication_id,
        ventas_30d=sales,
        visitas_30d=visits,
        precio_actual=1000.0,
        stock_actual=1,
    )


def test_validar_producto_accepts_consistent_commercial_counts():
    assert validar_producto(make_product("OK", 2, 10)) == []


def test_validar_producto_rejects_sales_without_visits():
    errors = validar_producto(make_product("BAD", 2, 0))
    assert "ventas_30d > 0" in errors[0]
    assert any("menores que ventas_30d" in error for error in errors)


def test_validar_producto_rejects_visits_lower_than_sales():
    errors = validar_producto(make_product("BAD", 5, 3))
    assert errors == [
        "inconsistencia: visitas_30d menores que ventas_30d"
    ]


def test_validar_productos_separates_valid_and_invalid_records():
    valid = make_product("OK", 1, 3)
    invalid = make_product("BAD", 4, 2)
    result = validar_productos([valid, invalid])
    assert result["validos"] == [valid]
    assert result["invalidos"][0]["publication_id"] == "BAD"
    assert result["invalidos"][0]["errores"]
