from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "productos": [
        {
            "publication_id": "MLC-1",
            "ventas_30d": 1,
            "visitas_30d": 5,
            "precio_actual": 9990,
            "stock_actual": 2,
            "en_promocion": False,
        }
    ]
}


def test_openapi_contains_all_public_routes_and_methods():
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/api/clasificar"]
    assert "post" in paths["/optimization/optuna"]
    assert "get" in paths["/api/explainability/health"]
    assert "post" in paths["/api/explainability/autosklearn/train"]
    assert "post" in paths["/api/explainability/autosklearn/explain"]


def test_health_route_returns_worker_payload(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.explainability.health_explainability_worker",
        lambda: {"status": "ok"},
    )
    response = client.get("/api/explainability/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_route_converts_worker_error_to_bad_gateway(monkeypatch):
    def fail():
        raise RuntimeError("worker no disponible")

    monkeypatch.setattr(
        "app.api.routes.explainability.health_explainability_worker",
        fail,
    )
    response = client.get("/api/explainability/health")
    assert response.status_code == 502
    assert response.json()["detail"] == "worker no disponible"


def test_explainability_query_parameters_are_validated():
    too_many = client.post(
        "/api/explainability/autosklearn/explain?top_n=4",
        json=VALID_PAYLOAD,
    )
    short_training = client.post(
        "/api/explainability/autosklearn/train"
        "?time_left_for_this_task=29&per_run_time_limit=10",
        json=VALID_PAYLOAD,
    )
    assert too_many.status_code == 422
    assert short_training.status_code == 422


def test_optimization_route_delegates_to_service(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.optimization.OptimizationService.optimize",
        lambda self, request: {
            "best_params": {"n_init": 10},
            "productos": len(request.productos),
        },
    )
    response = client.post("/optimization/optuna", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {
        "best_params": {"n_init": 10},
        "productos": 1,
    }
