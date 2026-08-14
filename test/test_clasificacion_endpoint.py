from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "productos": [
        {
            "publication_id": "MLC-1",
            "ventas_30d": 2,
            "visitas_30d": 10,
            "precio_actual": 9990,
            "stock_actual": 5,
            "en_promocion": False,
        }
    ]
}


def test_clasificar_endpoint_serializes_service_response(monkeypatch):
    expected = {
        "mensaje": "Clasificación ejecutada correctamente",
        "productos_validos": 1,
        "productos_invalidos": [],
        "diagnostico": {
            "capacidades_objetivo": {"A": 1, "B": 0, "C": 0},
            "conteos_finales": {"A": 1, "B": 0, "C": 0},
            "iteraciones": 2,
            "inertia": 0.0,
            "metricas": {},
        },
        "resultados": [
            {
                "publication_id": "MLC-1",
                "ventas_30d": 2,
                "visitas_30d": 10,
                "precio_actual": 9990,
                "stock_actual": 5,
                "en_promocion": 0,
                "categoria": "A",
                "score_inicial": 1.0,
            }
        ],
    }
    monkeypatch.setattr(
        "app.api.routes.clasificacion.ejecutar_clasificacion",
        lambda data: expected,
    )
    response = client.post("/api/clasificar", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == expected


def test_clasificar_endpoint_rejects_invalid_request_before_service_call():
    response = client.post(
        "/api/clasificar",
        json={
            "productos": [
                {
                    **VALID_PAYLOAD["productos"][0],
                    "precio_actual": 0,
                }
            ]
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_clasificar_endpoint_rejects_duplicate_publication_ids():
    duplicate = {
        "productos": [
            VALID_PAYLOAD["productos"][0],
            VALID_PAYLOAD["productos"][0],
        ]
    }
    response = client.post("/api/clasificar", json=duplicate)
    assert response.status_code == 422
    assert "duplicados" in str(response.json())

def test_clasificar_endpoint_requires_twenty_valid_products():
    response = client.post("/api/clasificar", json=VALID_PAYLOAD)
    body = response.json()

    assert response.status_code == 200
    assert body["productos_validos"] == 1
    assert body["minimo_operacional"] == 20
    assert body["resultados"] == []
    assert "después de excluir" in body["mensaje"]