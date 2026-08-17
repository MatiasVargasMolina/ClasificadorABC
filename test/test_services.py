from types import SimpleNamespace

import pandas as pd
import pytest
import requests

import app.services.clasificacion_service as classification_service
import app.services.explainability_service as explainability_service
import app.services.optimization_service as optimization_service
import app.services.kmeans_service as kmeans_service
from app.schemas.input_schema import RequestInput


class FakeKMeans:
    received_config = None
    received_metodo_asignacion = None

    def __init__(
        self,
        config,
        metodo_asignacion="global",
    ):
        type(self).received_config = config
        type(self).received_metodo_asignacion = metodo_asignacion

        self.proportions = config.proportions
        self.max_iter = config.max_iter
        self.tol = config.tol
        self.n_init = config.n_init
        self.random_state = config.random_state
        self.shuffle_unlabeled = config.shuffle_unlabeled
        self.metodo_asignacion = metodo_asignacion

        self.capacities_ = {
            "A": 1,
            "B": 1,
            "C": 1,
        }

        self.counts_ = {
            "A": 1,
            "B": 1,
            "C": 1,
        }

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


def test_kmeans_service_uses_central_production_configuration(
    monkeypatch,
):
    monkeypatch.setattr(
        kmeans_service,
        "SSEKMeans",
        FakeKMeans,
    )

    monkeypatch.setattr(
        kmeans_service,
        "evaluate_internal_metrics",
        lambda X, labels: {
            "silhouette": 0.5,
        },
    )

    X = pd.DataFrame(
        {
            "ventas_30d": [
                1.0,
                0.0,
                -1.0,
            ],
            "visitas_30d": [
                1.0,
                0.0,
                -1.0,
            ],
            "precio_actual": [
                -1.0,
                1.0,
                0.0,
            ],
        }
    )

    results, diagnostics = (
        kmeans_service.ejecutar_ss_kmeans(X)
    )

    assert results["categoria"].tolist() == [
        "A",
        "B",
        "C",
    ]

    assert FakeKMeans.received_config.n_init == 10

    assert (
        FakeKMeans.received_config.shuffle_unlabeled
        is True
    )

    assert (
        FakeKMeans.received_metodo_asignacion
        == "global"
    )

    assert (
        diagnostics["metodo_asignacion_utilizado"]
        == "global"
    )

    assert (
        diagnostics["configuracion"]["metodo_asignacion"]
        == "global"
    )

    assert diagnostics["convergio"] is True

    assert diagnostics["metricas"] == {
        "silhouette": 0.5,
    }


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

    result = classification_service.ejecutar_clasificacion(
        request
    )

    assert result["resultados"] == []
    assert result["productos_validos"] == 0
    assert result["minimo_operacional"] == 20

    assert (
        result["productos_invalidos"][0]["publication_id"]
        == "BAD"
    )

    assert "diagnostico" not in result


def test_classification_service_merges_model_output(
    monkeypatch,
):
    transformed = pd.DataFrame(
        {
            "publication_id": [
                str(i)
                for i in range(20)
            ],
            "ventas_30d": list(
                range(20, 0, -1)
            ),
            "visitas_30d": list(
                range(40, 20, -1)
            ),
            "precio_actual": [
                float(i)
                for i in range(20)
            ],
            "stock_actual": [1] * 20,
            "en_promocion": [0] * 20,
        }
    )

    monkeypatch.setattr(
        classification_service,
        "ejecutar_preprocesamiento",
        lambda data: {
            "hay_validos": True,
            "productos_validos": list(
                range(20)
            ),
            "productos_invalidos": [
                {
                    "publication_id": "INVALID",
                    "errores": [
                        "inconsistencia"
                    ],
                }
            ],
            "df_transformado": transformed,
            "X_modelo": transformed[
                [
                    "ventas_30d",
                    "visitas_30d",
                    "precio_actual",
                ]
            ],
        },
    )

    monkeypatch.setattr(
        classification_service,
        "ejecutar_ss_kmeans",
        lambda X, metodo_asignacion="global": (
            pd.DataFrame(
                {
                    "categoria": (
                        ["A"] * 4
                        + ["B"] * 6
                        + ["C"] * 10
                    ),
                    "score_inicial": [
                        float(i)
                        for i in range(20)
                    ],
                }
            ),
            {
                "metodo_asignacion_utilizado": (
                    metodo_asignacion
                ),
                "capacidades_objetivo": {
                    "A": 4,
                    "B": 6,
                    "C": 10,
                },
                "conteos_finales": {
                    "A": 4,
                    "B": 6,
                    "C": 10,
                },
                "iteraciones": 2,
                "inertia": 1.0,
                "metricas": {
                    "silhouette": 0.4,
                },
            },
        ),
    )

    result = (
        classification_service
        .ejecutar_clasificacion(
            object()
        )
    )

    assert [
        row["categoria"]
        for row in result["resultados"]
    ] == (
        ["A"] * 4
        + ["B"] * 6
        + ["C"] * 10
    )

    assert (
        len(result["resultados"])
        == result["productos_validos"]
    )

    assert (
        sum(
            result[
                "diagnostico"
            ][
                "conteos_finales"
            ].values()
        )
        == 20
    )

    assert (
        len(result["resultados"])
        + len(
            result["productos_invalidos"]
        )
        == 21
    )

    assert (
        result[
            "diagnostico"
        ][
            "metricas"
        ][
            "silhouette"
        ]
        == 0.4
    )

    assert (
        result["metodo_asignacion_utilizado"]
        == "global"
    )

    assert (
        result[
            "diagnostico"
        ][
            "metodo_asignacion_utilizado"
        ]
        == "global"
    )


def test_classification_service_propagates_sequential_assignment(
    monkeypatch,
):
    transformed = pd.DataFrame(
        {
            "publication_id": [
                str(i)
                for i in range(20)
            ],
            "ventas_30d": [1] * 20,
            "visitas_30d": [1] * 20,
            "precio_actual": [1000.0] * 20,
        }
    )

    monkeypatch.setattr(
        classification_service,
        "ejecutar_preprocesamiento",
        lambda data: {
            "hay_validos": True,
            "productos_validos": list(
                range(20)
            ),
            "productos_invalidos": [],
            "df_transformado": transformed,
            "X_modelo": transformed[
                [
                    "ventas_30d",
                    "visitas_30d",
                    "precio_actual",
                ]
            ],
        },
    )

    metodo_recibido = {}

    def fake_ejecutar_ss_kmeans(
        X,
        metodo_asignacion="global",
    ):
        metodo_recibido["valor"] = metodo_asignacion

        return (
            pd.DataFrame(
                {
                    "categoria": (
                        ["A"] * 4
                        + ["B"] * 6
                        + ["C"] * 10
                    ),
                    "score_inicial": [
                        float(i)
                        for i in range(20)
                    ],
                }
            ),
            {
                "metodo_asignacion_utilizado": (
                    metodo_asignacion
                ),
                "capacidades_objetivo": {
                    "A": 4,
                    "B": 6,
                    "C": 10,
                },
                "conteos_finales": {
                    "A": 4,
                    "B": 6,
                    "C": 10,
                },
                "iteraciones": 2,
                "inertia": 1.0,
                "metricas": {
                    "silhouette": 0.4,
                },
            },
        )

    monkeypatch.setattr(
        classification_service,
        "ejecutar_ss_kmeans",
        fake_ejecutar_ss_kmeans,
    )

    result = (
        classification_service
        .ejecutar_clasificacion(
            object(),
            metodo_asignacion="secuencial",
        )
    )

    assert (
        metodo_recibido["valor"]
        == "secuencial"
    )

    assert (
        result["metodo_asignacion_utilizado"]
        == "secuencial"
    )

    assert (
        result[
            "diagnostico"
        ][
            "metodo_asignacion_utilizado"
        ]
        == "secuencial"
    )


def test_post_to_worker_returns_json_and_disables_client_timeout(
    monkeypatch,
):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "status": "trained",
            }

    def fake_post(
        url,
        json,
        timeout,
    ):
        captured.update(
            url=url,
            json=json,
            timeout=timeout,
        )

        return Response()

    monkeypatch.setattr(
        explainability_service.requests,
        "post",
        fake_post,
    )

    result = (
        explainability_service
        ._post_to_worker(
            "/train",
            {
                "rows": [],
            },
            timeout_seconds=123,
        )
    )

    assert result == {
        "status": "trained",
    }

    assert captured[
        "url"
    ].endswith(
        "/train"
    )

    assert captured["timeout"] is None


def test_post_to_worker_converts_network_and_http_failures(
    monkeypatch,
):
    def connection_error(
        *args,
        **kwargs,
    ):
        raise requests.ConnectionError(
            "offline"
        )

    monkeypatch.setattr(
        explainability_service.requests,
        "post",
        connection_error,
    )

    with pytest.raises(
        RuntimeError,
        match="No se pudo conectar",
    ):
        explainability_service._post_to_worker(
            "/train",
            {},
            30,
        )

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

    with pytest.raises(
        RuntimeError,
        match="worker error",
    ):
        explainability_service._post_to_worker(
            "/train",
            {},
            30,
        )


def test_explain_service_sends_only_current_features(
    monkeypatch,
):
    captured = {}

    def fake_post(
        path,
        payload,
        timeout_seconds,
    ):
        captured.update(
            path=path,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )

        return {
            "ok": True,
        }

    monkeypatch.setattr(
        explainability_service,
        "_post_to_worker",
        fake_post,
    )

    result = (
        explainability_service
        .explicar_con_surrogate_autosklearn(
            request_with_one_product(),
            top_n=9,
        )
    )

    assert result == {
        "ok": True,
    }

    assert captured[
        "path"
    ] == "/explain"

    assert (
        captured[
            "payload"
        ][
            "top_n"
        ]
        == 3
    )

    assert set(
        captured[
            "payload"
        ][
            "rows"
        ][0]
    ) == {
        "publication_id",
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    }


def test_classification_service_requires_twenty_valid_rows_after_exclusions(
    monkeypatch,
):
    monkeypatch.setattr(
        classification_service,
        "ejecutar_preprocesamiento",
        lambda data: {
            "hay_validos": True,
            "productos_validos": list(
                range(19)
            ),
            "productos_invalidos": [
                {
                    "publication_id": "INVALID",
                    "errores": [
                        "inconsistencia"
                    ],
                }
            ],
        },
    )

    clustering_called = False

    def fail_if_called(X):
        nonlocal clustering_called
        clustering_called = True

        raise AssertionError(
            "SS-EKMeans no debe ejecutarse"
        )

    monkeypatch.setattr(
        classification_service,
        "ejecutar_ss_kmeans",
        fail_if_called,
    )

    result = (
        classification_service
        .ejecutar_clasificacion(
            object()
        )
    )

    assert clustering_called is False
    assert result["productos_validos"] == 19
    assert result["minimo_operacional"] == 20
    assert result["resultados"] == []

    assert (
        "después de excluir"
        in result["mensaje"]
    )


def test_optimization_service_requires_twenty_valid_rows(
    monkeypatch,
):
    monkeypatch.setattr(
        optimization_service,
        "ejecutar_preprocesamiento",
        lambda request: {
            "hay_validos": True,
            "productos_validos": list(
                range(19)
            ),
            "productos_invalidos": [],
        },
    )

    result = (
        optimization_service
        .OptimizationService()
        .optimize(
            object()
        )
    )

    assert result["productos_validos"] == 19
    assert result["minimo_operacional"] == 20
    assert result["best_params"] is None
    assert result["trials"] == []


def test_surrogate_training_inherits_operational_minimum(
    monkeypatch,
):
    monkeypatch.setattr(
        explainability_service,
        "ejecutar_clasificacion",
        lambda data, metodo_asignacion="global": {
            "mensaje": (
                "Se obtuvieron 19 "
                "publicaciones válidas."
            ),
            "productos_validos": 19,
            "productos_invalidos": [],
            "minimo_operacional": 20,
            "resultados": [],
        },
    )

    worker_called = False

    def fail_if_called(
        *args,
        **kwargs,
    ):
        nonlocal worker_called
        worker_called = True

        raise AssertionError(
            "El worker no debe entrenarse"
        )

    monkeypatch.setattr(
        explainability_service,
        "_post_to_worker",
        fail_if_called,
    )

    result = (
        explainability_service
        .entrenar_surrogate_autosklearn(
            object()
        )
    )

    assert worker_called is False

    assert (
        result[
            "clasificacion"
        ][
            "productos_validos"
        ]
        == 19
    )

    assert (
        result[
            "clasificacion"
        ][
            "minimo_operacional"
        ]
        == 20
    )

    assert (
        "No hay resultados válidos"
        in result["mensaje"]
    )