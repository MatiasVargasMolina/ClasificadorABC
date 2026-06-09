import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_clasificar_abc_desde_json():
    with open("data/input_request.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    response = client.post(
        "/api/clasificar",  # cambia si tu ruta real es otra
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)

    assert "mensaje" in data
    assert data["mensaje"] == "Clasificación ejecutada correctamente"

    assert "productos_validos" in data
    assert isinstance(data["productos_validos"], int)
    assert data["productos_validos"] > 0

    assert "productos_invalidos" in data
    assert isinstance(data["productos_invalidos"], list)

    assert "diagnostico" in data
    assert isinstance(data["diagnostico"], dict)

    diagnostico = data["diagnostico"]

    assert "conteos_finales" in diagnostico
    assert isinstance(diagnostico["conteos_finales"], dict)

    assert "capacidades_objetivo" in diagnostico
    assert isinstance(diagnostico["capacidades_objetivo"], dict)

    assert "iteraciones" in diagnostico
    assert isinstance(diagnostico["iteraciones"], int)

    assert "inertia" in diagnostico
    assert isinstance(diagnostico["inertia"], (int, float))

    conteos = diagnostico["conteos_finales"]
    capacidades = diagnostico["capacidades_objetivo"]

    for categoria in ["A", "B", "C"]:
        assert categoria in conteos
        assert categoria in capacidades
        assert isinstance(conteos[categoria], int)
        assert isinstance(capacidades[categoria], int)
        assert conteos[categoria] >= 0

    total_clasificado = sum(conteos.values())

    assert total_clasificado == data["productos_validos"]

    total = data["productos_validos"]

    assert abs(conteos["A"] / total - 0.20) < 0.02
    assert abs(conteos["B"] / total - 0.30) < 0.02
    assert abs(conteos["C"] / total - 0.50) < 0.02

    assert conteos["A"] == capacidades["A"]
    assert conteos["B"] == capacidades["B"]
    assert conteos["C"] == capacidades["C"]