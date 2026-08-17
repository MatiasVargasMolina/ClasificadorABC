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
        lambda self, request, metodo_asignacion="global": {
            "best_params": {
                "n_init": 10,
            },
            "productos": len(request.productos),
            "metodo_asignacion_utilizado": metodo_asignacion,
        },
    )

    response = client.post(
        "/optimization/optuna",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    assert response.json() == {
        "best_params": {
            "n_init": 10,
        },
        "productos": 1,
        "metodo_asignacion_utilizado": "global",
    }
def test_optimization_route_uses_global_by_default(monkeypatch):
    captured = {}

    def fake_optimize(
        self,
        request,
        metodo_asignacion="global",
    ):
        captured["metodo"] = metodo_asignacion

        return {
            "metodo_asignacion_utilizado": metodo_asignacion,
        }

    monkeypatch.setattr(
        "app.api.routes.optimization.OptimizationService.optimize",
        fake_optimize,
    )

    response = client.post(
        "/optimization/optuna",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200
    assert captured["metodo"] == "global"

    assert (
        response.json()["metodo_asignacion_utilizado"]
        == "global"
    )


def test_optimization_route_accepts_sequential_assignment(
    monkeypatch,
):
    captured = {}

    def fake_optimize(
        self,
        request,
        metodo_asignacion="global",
    ):
        captured["metodo"] = metodo_asignacion

        return {
            "metodo_asignacion_utilizado": metodo_asignacion,
        }

    monkeypatch.setattr(
        "app.api.routes.optimization.OptimizationService.optimize",
        fake_optimize,
    )

    response = client.post(
        "/optimization/optuna?metodo_asignacion=secuencial",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200
    assert captured["metodo"] == "secuencial"

    assert (
        response.json()["metodo_asignacion_utilizado"]
        == "secuencial"
    )


def test_optimization_route_rejects_invalid_assignment_method(
    monkeypatch,
):
    called = False

    def fake_optimize(
        self,
        request,
        metodo_asignacion="global",
    ):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "app.api.routes.optimization.OptimizationService.optimize",
        fake_optimize,
    )

    response = client.post(
        "/optimization/optuna?metodo_asignacion=invalido",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 422
    assert called is False
def test_surrogate_train_uses_global_by_default(monkeypatch):
    captured = {}

    def fake_train(
        data,
        time_left_for_this_task=600,
        per_run_time_limit=60,
        metodo_asignacion="global",
    ):
        captured["metodo"] = metodo_asignacion

        return {
            "metodo_asignacion_utilizado": metodo_asignacion,
        }

    monkeypatch.setattr(
        "app.api.routes.explainability.entrenar_surrogate_autosklearn",
        fake_train,
    )

    response = client.post(
        "/api/explainability/autosklearn/train",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200
    assert captured["metodo"] == "global"

    assert (
        response.json()["metodo_asignacion_utilizado"]
        == "global"
    )


def test_surrogate_train_accepts_sequential_assignment(
    monkeypatch,
):
    captured = {}

    def fake_train(
        data,
        time_left_for_this_task=600,
        per_run_time_limit=60,
        metodo_asignacion="global",
    ):
        captured["metodo"] = metodo_asignacion

        return {
            "metodo_asignacion_utilizado": metodo_asignacion,
        }

    monkeypatch.setattr(
        "app.api.routes.explainability.entrenar_surrogate_autosklearn",
        fake_train,
    )

    response = client.post(
        (
            "/api/explainability/autosklearn/train"
            "?metodo_asignacion=secuencial"
        ),
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200
    assert captured["metodo"] == "secuencial"

    assert (
        response.json()["metodo_asignacion_utilizado"]
        == "secuencial"
    )


def test_surrogate_train_rejects_invalid_assignment_method(
    monkeypatch,
):
    called = False

    def fake_train(
        data,
        time_left_for_this_task=600,
        per_run_time_limit=60,
        metodo_asignacion="global",
    ):
        nonlocal called
        called = True

        return {}

    monkeypatch.setattr(
        "app.api.routes.explainability.entrenar_surrogate_autosklearn",
        fake_train,
    )

    response = client.post(
        (
            "/api/explainability/autosklearn/train"
            "?metodo_asignacion=invalido"
        ),
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 422
    assert called is False