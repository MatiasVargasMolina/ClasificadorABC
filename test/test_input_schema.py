import pytest
from pydantic import ValidationError

from app.schemas.input_schema import ProductoInput, RequestInput


def valid_product(**overrides):
    values = {
        "publication_id": "MLC-1",
        "ventas_30d": 2,
        "visitas_30d": 10,
        "precio_actual": 9990.0,
        "stock_actual": 5,
    }
    values.update(overrides)
    return values


def test_producto_input_normalizes_id_and_defaults_promotion():
    product = ProductoInput(**valid_product(publication_id="  MLC-1  "))
    assert product.publication_id == "MLC-1"
    assert product.en_promocion is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ventas_30d", -1),
        ("visitas_30d", -1),
        ("precio_actual", 0),
        ("stock_actual", -1),
    ],
)
def test_producto_input_rejects_values_outside_schema(field, value):
    with pytest.raises(ValidationError):
        ProductoInput(**valid_product(**{field: value}))


def test_producto_input_rejects_blank_id():
    with pytest.raises(ValidationError, match="publication_id"):
        ProductoInput(**valid_product(publication_id="   "))


def test_removed_optional_label_is_ignored_by_input_schema():
    product = ProductoInput(
        **valid_product(),
        etiqueta_abc_opcional="A",
    )
    assert not hasattr(product, "etiqueta_abc_opcional")


def test_request_input_rejects_empty_products():
    with pytest.raises(ValidationError):
        RequestInput(productos=[])


def test_request_input_rejects_duplicate_ids_after_normalization():
    with pytest.raises(ValidationError, match="duplicados"):
        RequestInput(
            productos=[
                valid_product(publication_id="MLC-1"),
                valid_product(publication_id=" MLC-1 "),
            ]
        )


def test_request_input_accepts_unique_products():
    request = RequestInput(
        productos=[
            valid_product(publication_id="MLC-1"),
            valid_product(publication_id="MLC-2"),
        ]
    )
    assert [item.publication_id for item in request.productos] == [
        "MLC-1",
        "MLC-2",
    ]
