import pytest
from pydantic import ValidationError

from app.schemas.output_schema import (
    ResponseOutput,
    ResultadoClasificacion,
)


def valid_result(**overrides):
    values = {
        "publication_id": "MLC-1",
        "categoria": "A",
        "contribuciones": {"ventas_30d": 0.25},
    }
    values.update(overrides)
    return values


def test_resultado_clasificacion_normalizes_id_and_category():
    result = ResultadoClasificacion(
        **valid_result(publication_id="  MLC-1 ", categoria=" b ")
    )
    assert result.publication_id == "MLC-1"
    assert result.categoria == "B"


@pytest.mark.parametrize(
    "overrides",
    [
        {"publication_id": "   "},
        {"categoria": "D"},
        {"contribuciones": {}},
    ],
)
def test_resultado_clasificacion_rejects_invalid_output(overrides):
    with pytest.raises(ValidationError):
        ResultadoClasificacion(**valid_result(**overrides))


def test_response_output_requires_at_least_one_result():
    with pytest.raises(ValidationError):
        ResponseOutput(resultados=[])
    response = ResponseOutput(resultados=[valid_result()])
    assert response.resultados[0].categoria == "A"
