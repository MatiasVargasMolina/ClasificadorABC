from types import SimpleNamespace

import pandas as pd
import pytest
import requests

import app.services.clasificacion_service as classification_service
import app.services.explainability_service as explainability_service
import app.services.kmeans_service as kmeans_service
from app.schemas.input_schema import RequestInput


class FakeKMeans:
    received_config = None

    def __init__(self, config):
        type(self).received_config = config
        self.proportions = config.proportions
        self.max_iter = config.max_iter
        self.tol = config.tol
        self.n_init = config.n_init
        self.random_state = config.random_state
        self.shuffle_unlabeled = config.shuffle_unlabeled
        self.capacities_ = {"A": 1, "B": 1, "C": 1}
        self.counts_ = {"A": 1, "B": 1, "C": 1}
        self.n_iter_ = 2
        self.inertia_ = 1.5
        self.converged_ = True
        self.stop_reason_ = "labels_stable"
        self.converged_runs_ = config.n_init
        self.discarded_runs_ = 0

    def fit_predict(self, X):
        return pd.DataFrame(
            {
                "categoria": ["A", "B", "C"],
                "cluster": [0, 1, 2],
                "score_inicial": [3.0, 2.0, 1.0],
            },
            index=X.index,
        )


def request_with_one_product():
    return RequestInput(
        productos=[
            {
                "publication_id": "MLC-1",
                "ventas_30d": 1,
                "visitas_30d": 5,
                "precio_actual": 9990,
                "stock_actual": 2,
            }
        ]
    )


def test_kmeans_service_uses_central_production_configuration(monkeypatch):
    monkeypatch.setattr(kmeans_service, "SSEKMeans", FakeKMeans)
    monkeypatch.setattr(
        kmeans_service,
        "evaluate_internal_metrics",
        lambda X, labels: {"silhouette": 0.5},
    )
    X = pd.DataFrame(
        {
            "ventas_30d": [1.0, 0.0, -1.0],
            "visitas_30d": [1.0, 0.0, -1.0],
            "precio_actual": [-1.0, 1.0, 0.0],
        }
    )
    results, diagnostics = kmeans_service.ejecutar_ss_kmeans(X)
    assert results["categoria"].tolist() == ["A", "B", "C"]
    assert FakeKMeans.received_config.n_init == 10
    assert FakeKMeans.received_config.shuffle_unlabeled is True
    assert diagnostics["convergio"] is True
    assert diagnostics["metricas"] == {"silhouette": 0.5}


def test_classification_service_returns_invalid_contract_without_clustering():
    request = RequestInput(
        productos=[
            {
                "publication_id": "BAD",
                "ventas_30d": 5,
                "visitas_30d": 2,
                "precio_actual": 1000,
                "stock_actual": 1,
            }
        ]
    )
    result = classification_service.ejecutar_clasificacion(request)
    assert result["resultados"] == []
    assert result["productos_invalidos"][0]["publication_id"] == "BAD"
    assert "diagnostico" not in result


def test_classification_service_merges_model_output(monkeypatch):
    transformed = pd.DataFrame(
        {
            "publication_id": ["1", "2", "3"],
            "ventas_30d": [3, 2, 1],
            "visitas_30d": [9, 6, 3],
            "precio_actual": [10.0, 20.0, 30.0],
            "stock_actual": [1, 1, 1],
            "en_promocion": [0, 0, 0],
        }
    )
    monkeypatch.setattr(
        classification_service,
        "ejecutar_preprocesamiento",
        lambda data: {
            "hay_validos": True,
            "productos_validos": [1, 2, 3],
            "productos_invalidos": [],
            "df_transformado": transformed,
            "X_modelo": transformed[
                ["ventas_30d", "visitas_30d", "precio_actual"]
            ],
        },
    )
    monkeypatch.setattr(
        classification_service,
        "ejecutar_ss_kmeans",
        lambda X: (
            pd.DataFrame(
                {
                    "categoria": ["A", "B", "C"],
                    "score_inicial": [2.0, 1.0, 0.0],
                }
            ),
            {
                "capacidades_objetivo": {"A": 1, "B": 1, "C": 1},
                "conteos_finales": {"A": 1, "B": 1, "C": 1},
                "iteraciones": 2,
                "inertia": 1.0,
                "metricas": {"silhouette": 0.4},
            },
        ),
    )
    result = classification_service.ejecutar_clasificacion(object())
    assert [row["categoria"] for row in result["resultados"]] == [
        "A",
        "B",
        "C",
    ]
    assert result["diagnostico"]["metricas"]["silhouette"] == 0.4


def test_post_to_worker_returns_json_and_disables_client_timeout(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"status": "trained"}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(explainability_service.requests, "post", fake_post)
    result = explainability_service._post_to_worker(
        "/train",
        {"rows": []},
        timeout_seconds=123,
    )
    assert result == {"status": "trained"}
    assert captured["url"].endswith("/train")
    assert captured["timeout"] is None


def test_post_to_worker_converts_network_and_http_failures(monkeypatch):
    def connection_error(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(
        explainability_service.requests,
        "post",
        connection_error,
    )
    with pytest.raises(RuntimeError, match="No se pudo conectar"):
        explainability_service._post_to_worker("/train", {}, 30)

    response = SimpleNamespace(
        status_code=500,
        text="worker error",
        json=lambda: {},
    )
    monkeypatch.setattr(
        explainability_service.requests,
        "post",
        lambda *args, **kwargs: response,
    )
    with pytest.raises(RuntimeError, match="worker error"):
        explainability_service._post_to_worker("/train", {}, 30)


def test_explain_service_sends_only_current_features(monkeypatch):
    captured = {}

    def fake_post(path, payload, timeout_seconds):
        captured.update(
            path=path,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return {"ok": True}

    monkeypatch.setattr(explainability_service, "_post_to_worker", fake_post)
    result = explainability_service.explicar_con_surrogate_autosklearn(
        request_with_one_product(),
        top_n=9,
    )
    assert result == {"ok": True}
    assert captured["path"] == "/explain"
    assert captured["payload"]["top_n"] == 3
    assert set(captured["payload"]["rows"][0]) == {
        "publication_id",
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    }
