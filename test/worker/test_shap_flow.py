from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from autosklearn_worker import main as worker


class DeterministicSurrogate:
    """
    Modelo sustituto pequeño utilizado para probar
    explicabilidad sin entrenar AutoSklearn.
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
            values = values.reshape(
                1,
                -1,
            )

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

        exponentials = np.exp(
            logits
        )

        return (
            exponentials
            / exponentials.sum(
                axis=1,
                keepdims=True,
            )
        )

    def predict(self, data):
        probabilities = (
            self.predict_proba(
                data
            )
        )

        indexes = np.argmax(
            probabilities,
            axis=1,
        )

        return self.classes_[
            indexes
        ]


class UnknownClassSurrogate(
    DeterministicSurrogate
):
    def predict(self, data):
        values = np.asarray(
            data
        )

        rows = (
            1
            if values.ndim == 1
            else len(values)
        )

        return np.array(
            ["Z"] * rows,
            dtype=object,
        )


class FakeAutoSklearnClassifier:
    """
    Sustituye AutoSklearn en las pruebas de
    entrenamiento, evitando una búsqueda real.
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.config = kwargs

        self.classes_ = np.array(
            ["A", "B", "C"],
            dtype=object,
        )

    def fit(
        self,
        data,
        target,
    ):
        self.classes_ = np.array(
            sorted(
                {
                    str(value)
                    for value
                    in target
                }
            ),
            dtype=object,
        )

        return self

    def predict(
        self,
        data,
    ):
        frame = (
            worker.to_feature_dataframe(
                data
            )
        )

        ventas = (
            frame[
                "ventas_30d"
            ]
            .to_numpy(
                dtype=float
            )
        )

        visitas = (
            frame[
                "visitas_30d"
            ]
            .to_numpy(
                dtype=float
            )
        )

        return np.where(
            ventas >= 10,
            "A",
            np.where(
                visitas >= 10,
                "B",
                "C",
            ),
        )

    def predict_proba(
        self,
        data,
    ):
        predictions = self.predict(
            data
        )

        probabilities = np.full(
            (
                len(predictions),
                len(self.classes_),
            ),
            0.05,
            dtype=float,
        )

        for (
            row_index,
            prediction,
        ) in enumerate(
            predictions
        ):
            class_index = (
                list(
                    self.classes_
                )
                .index(
                    str(
                        prediction
                    )
                )
            )

            probabilities[
                row_index,
                class_index,
            ] = 0.90

        return probabilities

    def sprint_statistics(self):
        return (
            "AutoSklearn falso "
            "para pruebas"
        )

    def leaderboard(self):
        return pd.DataFrame(
            [
                {
                    "rank": 1,
                    "type": "fake",
                    "cost": 0.0,
                }
            ]
        )

    def show_models(self):
        return {
            "fake": {
                "weight": 1.0,
            }
        }


class ModelWithoutIntrospection:
    def sprint_statistics(self):
        raise RuntimeError(
            "sin estadísticas"
        )

    def leaderboard(self):
        raise RuntimeError(
            "sin leaderboard"
        )

    def show_models(self):
        raise RuntimeError(
            "sin modelos"
        )


class FastKernelExplainer:
    """
    Explainer determinista y aditivo utilizado para
    probar las ramas alternativas sin costo SHAP.
    """

    def __init__(
        self,
        predict_proba_fn,
        background,
        link="identity",
    ):
        self.predict_proba_fn = (
            predict_proba_fn
        )

        self.link = link

        background_outputs = np.asarray(
            predict_proba_fn(
                background
            ),
            dtype=float,
        )

        self.expected_value = (
            background_outputs.mean(
                axis=0
            )
        )

    def shap_values(
        self,
        data,
        nsamples=None,
        silent=True,
    ):
        outputs = np.asarray(
            self.predict_proba_fn(
                data
            ),
            dtype=float,
        )

        n_samples = len(
            outputs
        )

        n_features = len(
            worker.FEATURE_COLUMNS
        )

        n_classes = (
            outputs.shape[1]
        )

        values = np.zeros(
            (
                n_samples,
                n_features,
                n_classes,
            ),
            dtype=float,
        )

        values[
            :,
            0,
            :,
        ] = (
            outputs
            - self.expected_value
        )

        return values


def configure_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
) -> None:
    monkeypatch.setattr(
        worker,
        "ARTIFACT_DIR",
        directory,
    )

    monkeypatch.setattr(
        worker,
        "MODEL_PATH",
        directory
        / "autosklearn_model.joblib",
    )

    monkeypatch.setattr(
        worker,
        "BACKGROUND_PATH",
        directory
        / "autosklearn_background.joblib",
    )

    monkeypatch.setattr(
        worker,
        "META_PATH",
        directory
        / "autosklearn_meta.joblib",
    )


def valid_metadata() -> dict[
    str,
    Any,
]:
    return {
        "artifact_version": (
            worker.ARTIFACT_VERSION
        ),
        "feature_columns": (
            worker.FEATURE_COLUMNS.copy()
        ),
        (
            "categoria_ss_ekmeans_"
            "por_publicacion"
        ): {
            "MLC-A": "A",
            "MLC-B": "B",
            "MLC-C": "C",
        },
        (
            "id_ejecucion_"
            "entrenamiento"
        ): (
            "train-"
            "20260807T010000000000Z-"
            "abcdef123456"
        ),
        (
            "fecha_ejecucion_"
            "entrenamiento_utc"
        ): (
            "2026-08-07"
            "T01:00:00.000Z"
        ),
        "metrics": {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
        },
        "model_info": {
            "model": (
                "DeterministicSurrogate"
            ),
        },
        "shap_config": {
            (
                "background_"
                "random_state"
            ): 42,
        },
    }


def default_background() -> (
    pd.DataFrame
):
    return pd.DataFrame(
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
        columns=(
            worker.FEATURE_COLUMNS
        ),
    )


def write_artifacts(
    model: Any | None = None,
    background: (
        pd.DataFrame
        | None
    ) = None,
    metadata: (
        dict[str, Any]
        | None
    ) = None,
) -> tuple[
    Any,
    pd.DataFrame,
    dict[str, Any],
]:
    worker.ensure_artifact_dir()

    selected_model = (
        model
        if model is not None
        else DeterministicSurrogate()
    )

    selected_background = (
        background
        if background is not None
        else default_background()
    )

    selected_metadata = (
        metadata
        if metadata is not None
        else valid_metadata()
    )

    joblib.dump(
        selected_model,
        worker.MODEL_PATH,
    )

    joblib.dump(
        selected_background,
        worker.BACKGROUND_PATH,
    )

    joblib.dump(
        selected_metadata,
        worker.META_PATH,
    )

    return (
        selected_model,
        selected_background,
        selected_metadata,
    )


def build_training_request() -> (
    worker.TrainRequest
):
    rows: list[
        worker.TrainRow
    ] = []

    for index in range(5):
        rows.extend(
            [
                worker.TrainRow(
                    publication_id=(
                        f"MLC-A-{index}"
                    ),
                    ventas_30d=(
                        20 + index
                    ),
                    visitas_30d=30,
                    precio_actual=10000,
                    categoria="A",
                ),
                worker.TrainRow(
                    publication_id=(
                        f"MLC-B-{index}"
                    ),
                    ventas_30d=2,
                    visitas_30d=(
                        40 + index
                    ),
                    precio_actual=30000,
                    categoria="B",
                ),
                worker.TrainRow(
                    publication_id=(
                        f"MLC-C-{index}"
                    ),
                    ventas_30d=0,
                    visitas_30d=1,
                    precio_actual=8000,
                    categoria="C",
                ),
            ]
        )

    return worker.TrainRequest(
        rows=rows,
        time_left_for_this_task=30,
        per_run_time_limit=10,
    )


def build_explain_request(
    include_third_row: bool = True,
) -> worker.ExplainRequest:
    rows = [
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
    ]

    if include_third_row:
        rows.append(
            worker.PredictRow(
                publication_id="MLC-C",
                ventas_30d=0,
                visitas_30d=1,
                precio_actual=8000,
            )
        )

    return worker.ExplainRequest(
        rows=rows,
        top_n=3,
    )


def test_create_execution_trace_is_unique_and_utc() -> None:
    (
        first_id,
        first_timestamp,
    ) = worker.create_execution_trace(
        "shap"
    )

    (
        second_id,
        second_timestamp,
    ) = worker.create_execution_trace(
        "shap"
    )

    assert (
        first_id
        != second_id
    )

    assert re.fullmatch(
        (
            r"shap-"
            r"\d{8}T"
            r"\d{12}Z-"
            r"[0-9a-f]{12}"
        ),
        first_id,
    )

    assert (
        first_timestamp.endswith(
            "Z"
        )
    )

    assert (
        second_timestamp.endswith(
            "Z"
        )
    )


def test_dataframe_and_prediction_helpers() -> None:
    rows = build_explain_request(
        include_third_row=False
    ).rows

    frame = (
        worker.rows_to_dataframe(
            rows
        )
    )

    assert (
        list(frame.columns)
        == [
            "publication_id",
            *worker.FEATURE_COLUMNS,
        ]
    )

    selected = (
        worker.to_feature_dataframe(
            frame
        )
    )

    assert (
        list(selected.columns)
        == worker.FEATURE_COLUMNS
    )

    from_array = (
        worker.to_feature_dataframe(
            selected.to_numpy()
        )
    )

    pd.testing.assert_frame_equal(
        from_array.reset_index(
            drop=True
        ),
        selected.reset_index(
            drop=True
        ),
        check_dtype=False,
    )

    model = (
        DeterministicSurrogate()
    )

    predictions = (
        worker.build_predict_fn(
            model
        )(
            selected
        )
    )

    probabilities = (
        worker.build_predict_proba_fn(
            model
        )(
            selected
        )
    )

    assert (
        len(predictions)
        == len(selected)
    )

    assert (
        probabilities.shape
        == (
            len(selected),
            3,
        )
    )

    assert np.allclose(
        probabilities.sum(
            axis=1
        ),
        1.0,
    )


def test_build_background_is_bounded_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "ventas_30d": index,
                "visitas_30d": index,
                "precio_actual": (
                    1000 + index
                ),
            }
            for index in range(10)
        ]
    )

    monkeypatch.setattr(
        worker,
        "SHAP_BACKGROUND_SIZE",
        4,
    )

    monkeypatch.setattr(
        worker,
        (
            "SHAP_BACKGROUND_"
            "RANDOM_STATE"
        ),
        42,
    )

    first = (
        worker.build_background(
            frame
        )
    )

    second = (
        worker.build_background(
            frame
        )
    )

    assert (
        len(first)
        == 4
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_health_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    result = worker.health()

    assert (
        result["status"]
        == "ok"
    )

    assert (
        result["model_exists"]
        is False
    )

    assert (
        result[
            "background_exists"
        ]
        is False
    )

    assert (
        result["meta_exists"]
        is False
    )

    assert (
        result[
            "artifacts_compatible"
        ]
        is False
    )


def test_health_with_compatible_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    write_artifacts()

    result = worker.health()

    assert (
        result["model_exists"]
        is True
    )

    assert (
        result[
            "background_exists"
        ]
        is True
    )

    assert (
        result["meta_exists"]
        is True
    )

    assert (
        result[
            "artifact_feature_columns"
        ]
        == worker.FEATURE_COLUMNS
    )

    assert (
        result[
            (
                "categories_ss_ekmeans_"
                "available"
            )
        ]
        is True
    )

    assert (
        result[
            (
                "training_traceability_"
                "available"
            )
        ]
        is True
    )

    assert (
        result[
            "artifacts_compatible"
        ]
        is True
    )


def test_health_handles_a_corrupted_metadata_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    worker.ensure_artifact_dir()

    joblib.dump(
        DeterministicSurrogate(),
        worker.MODEL_PATH,
    )

    joblib.dump(
        default_background(),
        worker.BACKGROUND_PATH,
    )

    worker.META_PATH.write_bytes(
        b"metadata-invalida"
    )

    result = worker.health()

    assert (
        result["meta_exists"]
        is True
    )

    assert (
        result[
            "artifacts_compatible"
        ]
        is False
    )


def test_load_artifacts_rejects_missing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Faltan artefactos",
    ):
        worker.load_artifacts()


def test_load_artifacts_returns_compatible_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    (
        _,
        expected_background,
        expected_metadata,
    ) = write_artifacts()

    (
        model,
        background,
        metadata,
    ) = worker.load_artifacts()

    assert isinstance(
        model,
        DeterministicSurrogate,
    )

    pd.testing.assert_frame_equal(
        background,
        expected_background,
    )

    assert (
        metadata
        == expected_metadata
    )


def test_load_artifacts_rejects_feature_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    metadata = (
        valid_metadata()
    )

    metadata[
        "feature_columns"
    ] = [
        "otra_variable"
    ]

    write_artifacts(
        metadata=metadata
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "entrenados "
            "con variables"
        ),
    ):
        worker.load_artifacts()


def test_load_artifacts_rejects_background_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    background = (
        default_background()
        .rename(
            columns={
                "precio_actual": (
                    "precio_anterior"
                )
            }
        )
    )

    write_artifacts(
        background=background
    )

    with pytest.raises(
        RuntimeError,
        match="background SHAP",
    ):
        worker.load_artifacts()


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "artifact_version",
            3,
        ),
        (
            (
                "categoria_ss_ekmeans_"
                "por_publicacion"
            ),
            {},
        ),
        (
            (
                "id_ejecucion_"
                "entrenamiento"
            ),
            "",
        ),
        (
            (
                "fecha_ejecucion_"
                "entrenamiento_utc"
            ),
            "",
        ),
    ],
)
def test_load_artifacts_rejects_incomplete_traceability(
    field: str,
    value: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    metadata = (
        valid_metadata()
    )

    metadata[field] = value

    write_artifacts(
        metadata=metadata
    )

    with pytest.raises(
        RuntimeError,
        match="trazabilidad",
    ):
        worker.load_artifacts()


def test_get_autosklearn_model_info_when_available() -> None:
    model = (
        FakeAutoSklearnClassifier()
    )

    info = (
        worker.get_autosklearn_model_info(
            model
        )
    )

    assert (
        info["automl_class"]
        == (
            "FakeAutoSklearn"
            "Classifier"
        )
    )

    assert (
        "AutoSklearn falso"
        in info[
            "sprint_statistics"
        ]
    )

    assert (
        info[
            "leaderboard"
        ][0]["rank"]
        == 1
    )

    assert (
        "fake"
        in info[
            "models_with_weights"
        ]
    )


def test_get_autosklearn_model_info_handles_errors() -> None:
    info = (
        worker.get_autosklearn_model_info(
            ModelWithoutIntrospection()
        )
    )

    assert (
        info[
            "sprint_statistics"
        ]
        .startswith(
            "No disponible:"
        )
    )

    assert (
        info["leaderboard"]
        .startswith(
            "No disponible:"
        )
    )

    assert (
        info[
            "models_with_weights"
        ]
        .startswith(
            "No disponible:"
        )
    )


def test_train_rejects_invalid_time_limits() -> None:
    request = (
        build_training_request()
    )

    request.per_run_time_limit = 40
    request.time_left_for_this_task = 30

    with pytest.raises(
        worker.HTTPException
    ) as captured:
        worker.train(
            request
        )

    assert (
        captured.value.status_code
        == 400
    )

    assert (
        "per_run_time_limit"
        in captured.value.detail
    )


def test_train_rejects_a_single_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    request = worker.TrainRequest(
        rows=[
            worker.TrainRow(
                publication_id=(
                    f"MLC-{index}"
                ),
                ventas_30d=index,
                visitas_30d=index,
                precio_actual=10000,
                categoria="A",
            )
            for index in range(3)
        ],
        time_left_for_this_task=30,
        per_run_time_limit=10,
    )

    with pytest.raises(
        worker.HTTPException
    ) as captured:
        worker.train(
            request
        )

    assert (
        captured.value.status_code
        == 400
    )

    assert (
        "dos clases"
        in captured.value.detail
    )


def test_train_complete_flow_without_real_autosklearn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    monkeypatch.setattr(
        worker
        .autosklearn
        .classification,
        "AutoSklearnClassifier",
        FakeAutoSklearnClassifier,
    )

    result = worker.train(
        build_training_request()
    )

    assert (
        result["tipo_ejecucion"]
        == (
            "entrenamiento_"
            "autosklearn"
        )
    )

    assert (
        result[
            "version_artefacto"
        ]
        == worker.ARTIFACT_VERSION
    )

    assert (
        result["metrics"][
            "n_total"
        ]
        == 15
    )

    assert (
        result["metrics"][
            "accuracy"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        worker.MODEL_PATH.exists()
    )

    assert (
        worker
        .BACKGROUND_PATH
        .exists()
    )

    assert (
        worker.META_PATH.exists()
    )

    (
        _,
        background,
        metadata,
    ) = worker.load_artifacts()

    assert (
        list(
            background.columns
        )
        == worker.FEATURE_COLUMNS
    )

    assert (
        metadata[
            "artifact_version"
        ]
        == worker.ARTIFACT_VERSION
    )

    assert (
        len(
            metadata[
                (
                    "categoria_ss_ekmeans_"
                    "por_publicacion"
                )
            ]
        )
        == 15
    )


def test_train_falls_back_to_unstratified_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact_paths(
        monkeypatch,
        tmp_path,
    )

    monkeypatch.setattr(
        worker
        .autosklearn
        .classification,
        "AutoSklearnClassifier",
        FakeAutoSklearnClassifier,
    )

    real_split = (
        worker.train_test_split
    )

    calls = {
        "count": 0,
    }

    def split_with_first_failure(
        *args,
        **kwargs,
    ):
        calls["count"] += 1

        if calls["count"] == 1:
            raise ValueError(
                (
                    "fallo estratificado "
                    "simulado"
                )
            )

        return real_split(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        worker,
        "train_test_split",
        split_with_first_failure,
    )

    result = worker.train(
        build_training_request()
    )

    assert (
        calls["count"]
        == 2
    )

    assert (
        result["metrics"][
            "n_total"
        ]
        == 15
    )

    assert (
        result["mensaje"]
        == (
            "AutoSklearn "
            "entrenado correctamente"
        )
    )


def test_explain_translates_missing_artifacts_to_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load():
        raise FileNotFoundError(
            "faltan artefactos"
        )

    monkeypatch.setattr(
        worker,
        "load_artifacts",
        fail_load,
    )

    with pytest.raises(
        worker.HTTPException
    ) as captured:
        worker.explain(
            build_explain_request()
        )

    assert (
        captured.value.status_code
        == 404
    )

    assert (
        "faltan artefactos"
        in captured.value.detail
    )


def test_explain_translates_incompatible_artifacts_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load():
        raise RuntimeError(
            "artefactos incompatibles"
        )

    monkeypatch.setattr(
        worker,
        "load_artifacts",
        fail_load,
    )

    with pytest.raises(
        worker.HTTPException
    ) as captured:
        worker.explain(
            build_explain_request()
        )

    assert (
        captured.value.status_code
        == 409
    )

    assert (
        "incompatibles"
        in captured.value.detail
    )


def test_explain_truncates_and_allows_missing_ss_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = (
        DeterministicSurrogate()
    )

    background = (
        default_background()
    )

    metadata = (
        valid_metadata()
    )

    metadata[
        (
            "categoria_ss_ekmeans_"
            "por_publicacion"
        )
    ] = {
        "MLC-A": "A",
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
        worker.shap,
        "KernelExplainer",
        FastKernelExplainer,
    )

    monkeypatch.setattr(
        worker,
        "SHAP_MAX_EXPLAIN_ROWS",
        2,
    )

    monkeypatch.setattr(
        worker,
        "SHAP_NSAMPLES",
        6,
    )

    result = worker.explain(
        build_explain_request(
            include_third_row=True
        )
    )

    assert (
        result["shap_config"][
            "productos_recibidos"
        ]
        == 3
    )

    assert (
        result["shap_config"][
            "productos_explicados"
        ]
        == 2
    )

    assert (
        result["shap_config"][
            "truncated"
        ]
        is True
    )

    assert (
        result[
            "resumen_concordancia"
        ][
            "total_comparables"
        ]
        == 1
    )

    assert (
        result[
            "resumen_concordancia"
        ][
            "sin_categoria_ss_ekmeans"
        ]
        == 1
    )

    assert (
        result[
            "predicciones"
        ][1][
            "categoria_ss_ekmeans"
        ]
        is None
    )

    assert (
        result[
            "predicciones"
        ][1][
            "concordancia"
        ]
        is None
    )


def test_explain_handles_a_batch_without_comparable_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = (
        DeterministicSurrogate()
    )

    background = (
        default_background()
    )

    metadata = (
        valid_metadata()
    )

    metadata[
        (
            "categoria_ss_ekmeans_"
            "por_publicacion"
        )
    ] = {
        "OTRA": "A",
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
        worker.shap,
        "KernelExplainer",
        FastKernelExplainer,
    )

    monkeypatch.setattr(
        worker,
        "SHAP_MAX_EXPLAIN_ROWS",
        3,
    )

    request = (
        worker.ExplainRequest(
            rows=[
                worker.PredictRow(
                    publication_id=(
                        "SIN-CATEGORIA"
                    ),
                    ventas_30d=0,
                    visitas_30d=1,
                    precio_actual=8000,
                )
            ],
            top_n=1,
        )
    )

    result = worker.explain(
        request
    )

    summary = result[
        "resumen_concordancia"
    ]

    assert (
        summary[
            "total_comparables"
        ]
        == 0
    )

    assert (
        summary[
            "tasa_concordancia"
        ]
        is None
    )

    assert (
        summary[
            "porcentaje_concordancia"
        ]
        is None
    )

    assert (
        summary[
            (
                "discrepancias_"
                "por_transicion"
            )
        ]
        == []
    )


def test_explain_rejects_a_class_unknown_to_the_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = (
        UnknownClassSurrogate()
    )

    background = (
        default_background()
    )

    metadata = (
        valid_metadata()
    )

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
        worker.shap,
        "KernelExplainer",
        FastKernelExplainer,
    )

    with pytest.raises(
        worker.HTTPException
    ) as captured:
        worker.explain(
            build_explain_request(
                include_third_row=False
            )
        )

    assert (
        captured.value.status_code
        == 500
    )

    assert (
        "clase desconocida"
        in captured.value.detail
    )


def test_complete_kernel_shap_explanation_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Mantiene una prueba que utiliza el
    KernelExplainer real del contenedor.
    """

    model = (
        DeterministicSurrogate()
    )

    background = (
        default_background()
    )

    metadata = (
        valid_metadata()
    )

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

    result = worker.explain(
        build_explain_request(
            include_third_row=True
        )
    )

    assert (
        result["mensaje"]
        == (
            "Explicaciones "
            "generadas correctamente"
        )
    )

    assert (
        result["modelo"]
        == "AutoSklearnClassifier"
    )

    assert (
        result[
            "variables_explicadas"
        ]
        == worker.FEATURE_COLUMNS
    )

    shap_config = result[
        "shap_config"
    ]

    assert (
        shap_config["explainer"]
        == "KernelExplainer"
    )

    assert (
        shap_config["link"]
        == "identity"
    )

    assert (
        shap_config["nsamples"]
        == 6
    )

    assert (
        shap_config[
            "productos_recibidos"
        ]
        == 3
    )

    assert (
        shap_config[
            "productos_explicados"
        ]
        == 3
    )

    assert (
        shap_config["truncated"]
        is False
    )

    predictions = result[
        "predicciones"
    ]

    assert (
        len(predictions)
        == 3
    )

    assert {
        row[
            "categoria_sustituto"
        ]
        for row
        in predictions
    } == {
        "A",
        "B",
        "C",
    }

    for prediction in predictions:
        assert (
            prediction[
                "prediccion"
            ]
            == prediction[
                "categoria_sustituto"
            ]
        )

        assert (
            prediction[
                "concordancia"
            ]
            == (
                prediction[
                    "categoria_ss_ekmeans"
                ]
                == prediction[
                    "categoria_sustituto"
                ]
            )
        )

        assert (
            set(
                prediction[
                    "probabilidades"
                ]
            )
            == {
                "A",
                "B",
                "C",
            }
        )

        assert np.isclose(
            sum(
                prediction[
                    "probabilidades"
                ].values()
            ),
            1.0,
        )

        assert (
            len(
                prediction[
                    "contribuciones"
                ]
            )
            == len(
                worker.FEATURE_COLUMNS
            )
        )

        assert (
            len(
                prediction[
                    "top_contribuciones"
                ]
            )
            == 3
        )

        reconstructed = (
            prediction[
                "valor_base"
            ]
            + sum(
                contribution[
                    "shap_value"
                ]
                for contribution
                in prediction[
                    "contribuciones"
                ]
            )
        )

        expected_probability = (
            prediction[
                "probabilidades"
            ][
                prediction[
                    "explicacion_clase"
                ]
            ]
        )

        assert np.isclose(
            reconstructed,
            prediction[
                "probabilidad_reconstruida"
            ],
            atol=1e-6,
        )

        assert np.isclose(
            reconstructed,
            expected_probability,
            atol=1e-6,
        )

    summary = result[
        "resumen_concordancia"
    ]

    assert (
        summary[
            "total_explicados"
        ]
        == 3
    )

    assert (
        summary[
            "total_comparables"
        ]
        == 3
    )

    assert (
        summary[
            "coincidencias"
        ]
        + summary[
            "discrepancias"
        ]
        == 3
    )

    assert (
        result[
            "validacion_aditividad"
        ][
            "cumple_tolerancia"
        ]
        is True
    )