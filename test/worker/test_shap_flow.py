from __future__ import annotations

import numpy as np
import pandas as pd

from autosklearn_worker import main as worker


class DeterministicSurrogate:
    """
    Modelo sustituto pequeño y determinista utilizado únicamente para
    comprobar el flujo real de explicación del worker.
    """

    classes_ = np.array(
        ["A", "B", "C"],
        dtype=object,
    )

    def predict_proba(self, data):
        values = np.asarray(
            data,
            dtype=float,
        )

        if values.ndim == 1:
            values = values.reshape(1, -1)

        ventas = values[:, 0]
        visitas = values[:, 1]
        precio = values[:, 2]

        score_a = (
            0.80 * ventas
            + 0.05 * visitas
            - precio / 100000
        )

        score_b = (
            precio / 10000
            - 0.20 * ventas
            - 0.01 * visitas
        )

        score_c = (
            2.0
            - 0.50 * ventas
            - 0.02 * visitas
            - precio / 20000
        )

        logits = np.column_stack(
            [
                score_a,
                score_b,
                score_c,
            ]
        )

        logits = (
            logits
            - logits.max(
                axis=1,
                keepdims=True,
            )
        )

        exponentials = np.exp(logits)

        return (
            exponentials
            / exponentials.sum(
                axis=1,
                keepdims=True,
            )
        )

    def predict(self, data):
        probabilities = self.predict_proba(data)
        indexes = np.argmax(
            probabilities,
            axis=1,
        )
        return self.classes_[indexes]


def test_complete_kernel_shap_explanation_flow(
    monkeypatch,
):
    model = DeterministicSurrogate()

    background = pd.DataFrame(
        [
            {
                "ventas_30d": 20,
                "visitas_30d": 100,
                "precio_actual": 10000,
            },
            {
                "ventas_30d": 0,
                "visitas_30d": 0,
                "precio_actual": 30000,
            },
            {
                "ventas_30d": 0,
                "visitas_30d": 1,
                "precio_actual": 8000,
            },
        ],
        columns=worker.FEATURE_COLUMNS,
    )

    metadata = {
        "artifact_version": worker.ARTIFACT_VERSION,
        "feature_columns": (
            worker.FEATURE_COLUMNS.copy()
        ),
        "categoria_ss_ekmeans_por_publicacion": {
            "MLC-A": "A",
            "MLC-B": "B",
            "MLC-C": "A",
        },
        "metrics": {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
        },
        "model_info": {
            "model": "DeterministicSurrogate",
        },
        "shap_config": {
            "background_random_state": 42,
        },
    }

    monkeypatch.setattr(
        worker,
        "load_artifacts",
        lambda: (
            model,
            background,
            metadata,
        ),
    )

    monkeypatch.setattr(
        worker,
        "SHAP_NSAMPLES",
        6,
    )

    monkeypatch.setattr(
        worker,
        "SHAP_MAX_EXPLAIN_ROWS",
        3,
    )

    request = worker.ExplainRequest(
        rows=[
            worker.PredictRow(
                publication_id="MLC-A",
                ventas_30d=20,
                visitas_30d=100,
                precio_actual=10000,
            ),
            worker.PredictRow(
                publication_id="MLC-B",
                ventas_30d=0,
                visitas_30d=0,
                precio_actual=30000,
            ),
            worker.PredictRow(
                publication_id="MLC-C",
                ventas_30d=0,
                visitas_30d=1,
                precio_actual=8000,
            ),
        ],
        top_n=3,
    )

    result = worker.explain(request)

    assert (
        result["mensaje"]
        == "Explicaciones generadas correctamente"
    )

    assert result["modelo"] == (
        "AutoSklearnClassifier"
    )

    assert result["variables_explicadas"] == [
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    ]

    shap_config = result["shap_config"]

    assert shap_config["explainer"] == (
        "KernelExplainer"
    )
    assert shap_config["link"] == "identity"
    assert shap_config["nsamples"] == 6
    assert shap_config["productos_recibidos"] == 3
    assert shap_config["productos_explicados"] == 3
    assert shap_config["truncated"] is False

    predictions = result["predicciones"]

    assert len(predictions) == 3

    predicted_categories = {
        prediction["categoria_sustituto"]
        for prediction in predictions
    }

    assert predicted_categories == {
        "A",
        "B",
        "C",
    }

    expected_ss_categories = {
        "MLC-A": "A",
        "MLC-B": "B",
        "MLC-C": "A",
    }

    for prediction in predictions:
        assert (
            prediction["categoria_ss_ekmeans"]
            == expected_ss_categories[
                prediction["publication_id"]
            ]
        )
        assert (
            prediction["prediccion"]
            == prediction["categoria_sustituto"]
        )
        assert (
            prediction["concordancia"]
            == (
                prediction["categoria_ss_ekmeans"]
                == prediction["categoria_sustituto"]
            )
        )

        assert set(
            prediction["probabilidades"]
        ) == {
            "A",
            "B",
            "C",
        }

        assert np.isclose(
            sum(
                prediction[
                    "probabilidades"
                ].values()
            ),
            1.0,
        )

        assert len(
            prediction["contribuciones"]
        ) == 3

        assert len(
            prediction["top_contribuciones"]
        ) == 3

        assert (
            prediction["error_aditividad"]
            <= 1e-4
        )

        contribution_features = {
            contribution["feature"]
            for contribution
            in prediction["contribuciones"]
        }

        assert contribution_features == {
            "ventas_30d",
            "visitas_30d",
            "precio_actual",
        }

    concordance = result["resumen_concordancia"]
    assert concordance["total_explicados"] == 3
    assert concordance["total_comparables"] == 3
    assert concordance["coincidencias"] == 2
    assert concordance["discrepancias"] == 1
    assert concordance[
        "sin_categoria_ss_ekmeans"
    ] == 0
    assert np.isclose(
        concordance["tasa_concordancia"],
        2 / 3,
    )
    assert np.isclose(
        concordance["porcentaje_concordancia"],
        200 / 3,
    )

    assert {
        item["feature"]
        for item in result["importancia_global"]
    } == {
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    }

    assert set(
        result["importancia_por_clase"]
    ) == {
        "A",
        "B",
        "C",
    }

    additivity = (
        result["validacion_aditividad"]
    )

    assert (
        additivity["cumple_tolerancia"]
        is True
    )
    assert (
        additivity["max_absolute_error"]
        <= additivity["tolerance"]
    )