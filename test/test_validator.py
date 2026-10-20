import pytest
from pydantic import ValidationError

from app.preprocessing.validator import validar_producto, validar_productos
from app.schemas.input_schema import ProductoInput, RequestInput


def make_product(publication_id, sales, visits):
    return ProductoInput(
        publication_id=publication_id, ventas_30d=sales, visitas_30d=visits,
        precio_actual=1000.0, stock_actual=1,
    )


@pytest.mark.parametrize('sales,visits', [(2, 10), (2, 1), (5, 3), (2, 0), (0, 0)])
def test_counts_are_validated_independently(sales, visits):
    assert validar_producto(make_product('OK', sales, visits)) == []


def test_batch_retains_multiple_units_per_visit():
    products = [make_product('A', 1, 3), make_product('B', 4, 2)]
    assert validar_productos(products) == {'validos': products, 'invalidos': []}


@pytest.mark.parametrize('field,value', [
    ('publication_id', ' '), ('ventas_30d', -1),
    ('visitas_30d', -1), ('precio_actual', 0),
])
def test_schema_still_rejects_invalid_fields(field, value):
    record = make_product('A', 2, 1).model_dump()
    record[field] = value
    with pytest.raises(ValidationError):
        ProductoInput(**record)


def test_request_still_rejects_duplicate_ids():
    product = make_product('A', 2, 1)
    with pytest.raises(ValidationError):
        RequestInput(productos=[product, product])
