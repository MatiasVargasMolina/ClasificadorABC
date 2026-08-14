from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


os.environ.setdefault("MPLBACKEND", "Agg")


from generate_shap_result_figures import (  # noqa: E402
    CLASSES,
    build_additivity_rows,
    build_concordance_summary,
    build_concordance_detail_rows,
    build_concordance_transition_rows,
    choose_representative_cases,
    main as generate_figures_main,
    validate_response,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESPONSE_PATH = (
    PROJECT_ROOT
    / "data"
    / "shap_explain_response.json"
)

RESPONSE_PATH = Path(
    os.getenv(
        "SHAP_EXPLAIN_RESPONSE_PATH",
        str(DEFAULT_RESPONSE_PATH),
    )
)


EXPECTED_TOTAL = 832
EXPECTED_AGREEMENTS = 823
EXPECTED_DISCREPANCIES = 9
EXPECTED_CONCORDANCE_RATE = 823 / 832

EXPECTED_MATRIX = {
    "A": {
        "A": 165,
        "B": 1,
        "C": 0,
    },
    "B": {
        "A": 0,
        "B": 249,
        "C": 1,
    },
    "C": {
        "A": 7,
        "B": 0,
        "C": 409,
    },
}

EXPECTED_TRANSITIONS = [
    {
        "categoria_ss_ekmeans": "A",
        "categoria_sustituto": "B",
        "cantidad": 1,
    },
    {
        "categoria_ss_ekmeans": "B",
        "categoria_sustituto": "C",
        "cantidad": 1,
    },
    {
        "categoria_ss_ekmeans": "C",
        "categoria_sustituto": "A",
        "cantidad": 7,
    },
]

EXPECTED_SS_COUNTS = {
    "A": 166,
    "B": 250,
    "C": 416,
}

EXPECTED_SURROGATE_COUNTS = {
    "A": 172,
    "B": 250,
    "C": 410,
}

EXPECTED_REPRESENTATIVE_IDS = {
    "A": "MLC3737992360",
    "B": "MLC1887635545",
    "C": "MLC1878041405",
}

EXPECTED_FIGURES = {
    "figura_shap_01_fidelidad_sustituto.png",
    "figura_shap_02_importancia_global.png",
    "figura_shap_03_importancia_por_categoria.png",
    "figura_shap_04_casos_representativos.png",
    "figura_shap_05_validacion_aditividad.png",
}

EXPECTED_TABLES = {
    "tabla_shap_01_fidelidad_sustituto.csv",
    "tabla_shap_02_importancia_global.csv",
    "tabla_shap_03_importancia_por_categoria.csv",
    "tabla_shap_04_casos_representativos.csv",
    "tabla_shap_05_excepciones_aditividad.csv",
    "tabla_shap_06_concordancia_categorias.csv",
    "tabla_shap_07_concordancia_publicaciones.csv",
}


def read_response() -> dict[str, Any]:
    if not RESPONSE_PATH.exists():
        pytest.fail(
            "No se encontró la respuesta SHAP final en "
            f"{RESPONSE_PATH}."
        )

    with RESPONSE_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as input_file:
        response = json.load(input_file)

    if not isinstance(response, dict):
        pytest.fail(
            "shap_explain_response.json debe contener "
            "un objeto JSON."
        )

    return response


@pytest.fixture(scope="module")
def response() -> dict[str, Any]:
    return read_response()


def parse_utc_timestamp(
    value: str,
) -> datetime:
    return datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00",
        )
    )


def read_csv_rows(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(
            input_file
        )
        columns = list(
            reader.fieldnames
            or []
        )
        rows = list(
            reader
        )

    return (
        columns,
        rows,
    )


def test_final_response_is_complete_and_traceable(
    response: dict[str, Any],
) -> None:
    validate_response(
        response,
        expected_products=EXPECTED_TOTAL,
    )

    assert (
        response["version_artefacto"]
        == 4
    )

    assert (
        response["tipo_ejecucion"]
        == "explicabilidad_shap"
    )

    shap_execution_id = (
        response["id_ejecucion"]
    )

    training_execution_id = response[
        "id_ejecucion_entrenamiento"
    ]

    assert re.fullmatch(
        (
            r"shap-"
            r"\d{8}T"
            r"\d{12}Z-"
            r"[0-9a-f]{12}"
        ),
        shap_execution_id,
    )

    assert re.fullmatch(
        (
            r"train-"
            r"\d{8}T"
            r"\d{12}Z-"
            r"[0-9a-f]{12}"
        ),
        training_execution_id,
    )

    shap_timestamp = (
        parse_utc_timestamp(
            response[
                "fecha_ejecucion_utc"
            ]
        )
    )

    training_timestamp = (
        parse_utc_timestamp(
            response[
                "fecha_ejecucion_entrenamiento_utc"
            ]
        )
    )

    assert (
        training_timestamp
        <= shap_timestamp
    )

    shap_config = response[
        "shap_config"
    ]

    assert (
        shap_config[
            "productos_recibidos"
        ]
        == EXPECTED_TOTAL
    )

    assert (
        shap_config[
            "productos_explicados"
        ]
        == EXPECTED_TOTAL
    )

    assert (
        shap_config[
            "limite_productos_explicados"
        ]
        == EXPECTED_TOTAL
    )

    assert (
        shap_config["truncated"]
        is False
    )

    predictions = response[
        "predicciones"
    ]

    assert (
        len(predictions)
        == EXPECTED_TOTAL
    )

    publication_ids = {
        prediction[
            "publication_id"
        ]
        for prediction
        in predictions
    }

    assert (
        len(publication_ids)
        == EXPECTED_TOTAL
    )

    assert all(
        (
            prediction[
                "id_ejecucion"
            ]
            == shap_execution_id
        )
        for prediction
        in predictions
    )

    assert all(
        (
            prediction[
                "id_ejecucion_entrenamiento"
            ]
            == training_execution_id
        )
        for prediction
        in predictions
    )

    assert all(
        (
            prediction[
                "categoria_ss_ekmeans"
            ]
            in {
                "A",
                "B",
                "C",
            }
        )
        for prediction
        in predictions
    )

    assert all(
        (
            prediction[
                "categoria_sustituto"
            ]
            in {
                "A",
                "B",
                "C",
            }
        )
        for prediction
        in predictions
    )

    assert all(
        (
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
        for prediction
        in predictions
    )


def test_training_metrics_preserve_the_official_class_order(
    response: dict[str, Any],
) -> None:
    metrics = response[
        "metrics_entrenamiento"
    ]

    assert (
        metrics["classes"]
        == list(CLASSES)
    )

    assert (
        metrics["n_total"]
        == EXPECTED_TOTAL
    )

    assert (
        metrics["n_train"]
        + metrics["n_test"]
        == EXPECTED_TOTAL
    )

    confusion_matrix = metrics[
        "confusion_matrix"
    ]

    assert (
        len(confusion_matrix)
        == len(CLASSES)
    )

    assert all(
        (
            len(row)
            == len(CLASSES)
        )
        for row
        in confusion_matrix
    )

    assert (
        sum(
            sum(
                int(value)
                for value
                in row
            )
            for row
            in confusion_matrix
        )
        == metrics["n_test"]
    )

    assert (
        0.0
        <= float(
            metrics["accuracy"]
        )
        <= 1.0
    )

    assert (
        0.0
        <= float(
            metrics[
                "balanced_accuracy"
            ]
        )
        <= 1.0
    )

    assert (
        0.0
        <= float(
            metrics["macro_f1"]
        )
        <= 1.0
    )


def test_surrogate_category_uses_full_precision_probabilities(
    response: dict[str, Any],
) -> None:
    for prediction in response[
        "predicciones"
    ]:
        probabilities = prediction[
            "probabilidades"
        ]

        assert (
            list(probabilities)
            == list(CLASSES)
        )

        probability_sum = sum(
            float(
                probabilities[
                    class_name
                ]
            )
            for class_name
            in CLASSES
        )

        assert (
            probability_sum
            == pytest.approx(
                1.0,
                abs=1e-9,
            )
        )

        expected_category = max(
            CLASSES,
            key=lambda class_name: float(
                probabilities[
                    class_name
                ]
            ),
        )

        assert (
            prediction[
                "prediccion"
            ]
            == expected_category
        )

        assert (
            prediction[
                "categoria_sustituto"
            ]
            == expected_category
        )

        assert (
            prediction[
                "explicacion_clase"
            ]
            == expected_category
        )

        ordered_probabilities = sorted(
            (
                float(
                    probabilities[
                        class_name
                    ]
                )
                for class_name
                in CLASSES
            ),
            reverse=True,
        )

        probability_margin = (
            ordered_probabilities[0]
            - ordered_probabilities[1]
        )

        assert (
            0.0
            <= probability_margin
            <= 1.0
        )


def test_additivity_is_valid_for_every_explanation(
    response: dict[str, Any],
) -> None:
    (
        rows,
        exceptions,
        tolerance,
    ) = build_additivity_rows(
        response
    )

    validation = response[
        "validacion_aditividad"
    ]

    assert (
        len(rows)
        == EXPECTED_TOTAL
    )

    assert (
        exceptions
        == []
    )

    assert (
        tolerance
        == pytest.approx(
            float(
                validation[
                    "tolerance"
                ]
            ),
            abs=0.0,
        )
    )

    assert (
        validation[
            "cumple_tolerancia"
        ]
        is True
    )

    calculated_errors: list[
        float
    ] = []

    for prediction in response[
        "predicciones"
    ]:
        explained_class = prediction[
            "explicacion_clase"
        ]

        predicted_probability = float(
            prediction[
                "probabilidades"
            ][
                explained_class
            ]
        )

        reconstructed_probability = float(
            prediction[
                "probabilidad_reconstruida"
            ]
        )

        reconstructed_from_contributions = (
            float(
                prediction[
                    "valor_base"
                ]
            )
            + sum(
                float(
                    contribution[
                        "shap_value"
                    ]
                )
                for contribution
                in prediction[
                    "contribuciones"
                ]
            )
        )

        calculated_error = abs(
            predicted_probability
            - reconstructed_probability
        )

        calculated_errors.append(
            calculated_error
        )

        assert (
            reconstructed_from_contributions
            == pytest.approx(
                reconstructed_probability,
                abs=1e-12,
            )
        )

        assert (
            abs(
                float(
                    prediction[
                        "error_aditividad"
                    ]
                )
            )
            == pytest.approx(
                calculated_error,
                abs=1e-12,
            )
        )

        assert (
            calculated_error
            <= tolerance
        )

    assert (
        max(calculated_errors)
        == pytest.approx(
            float(
                validation[
                    "max_absolute_error"
                ]
            ),
            abs=1e-12,
        )
    )


def test_final_concordance_matches_official_results(
    response: dict[str, Any],
) -> None:
    summary = response[
        "resumen_concordancia"
    ]

    assert (
        summary[
            "total_explicados"
        ]
        == EXPECTED_TOTAL
    )

    assert (
        summary[
            "total_comparables"
        ]
        == EXPECTED_TOTAL
    )

    assert (
        summary[
            "coincidencias"
        ]
        == EXPECTED_AGREEMENTS
    )

    assert (
        summary[
            "discrepancias"
        ]
        == EXPECTED_DISCREPANCIES
    )

    assert (
        summary[
            "sin_categoria_ss_ekmeans"
        ]
        == 0
    )

    assert (
        summary[
            "tasa_concordancia"
        ]
        == pytest.approx(
            EXPECTED_CONCORDANCE_RATE,
            abs=1e-12,
        )
    )

    assert (
        summary[
            "porcentaje_concordancia"
        ]
        == pytest.approx(
            (
                EXPECTED_CONCORDANCE_RATE
                * 100
            ),
            abs=1e-12,
        )
    )

    assert (
        summary[
            "matriz_concordancia"
        ]
        == EXPECTED_MATRIX
    )

    assert (
        summary[
            "discrepancias_por_transicion"
        ]
        == EXPECTED_TRANSITIONS
    )

    predictions = response[
        "predicciones"
    ]

    ss_counts = Counter(
        prediction[
            "categoria_ss_ekmeans"
        ]
        for prediction
        in predictions
    )

    surrogate_counts = Counter(
        prediction[
            "categoria_sustituto"
        ]
        for prediction
        in predictions
    )

    assert (
        dict(ss_counts)
        == EXPECTED_SS_COUNTS
    )

    assert (
        dict(surrogate_counts)
        == EXPECTED_SURROGATE_COUNTS
    )

    discrepancies = [
        prediction
        for prediction
        in predictions
        if (
            prediction[
                "concordancia"
            ]
            is False
        )
    ]

    assert (
        len(discrepancies)
        == EXPECTED_DISCREPANCIES
    )


def test_concordance_summary_is_reconstructed_from_predictions(
    response: dict[str, Any],
) -> None:
    reconstructed = (
        build_concordance_summary(
            response[
                "predicciones"
            ]
        )
    )

    received = response[
        "resumen_concordancia"
    ]

    for (
        key,
        value,
    ) in reconstructed.items():
        if key in {
            "tasa_concordancia",
            "porcentaje_concordancia",
        }:
            assert (
                received[key]
                == pytest.approx(
                    value,
                    abs=1e-12,
                )
            )
        else:
            assert (
                received[key]
                == value
            )

    assert (
        received[
            "id_ejecucion"
        ]
        == response[
            "id_ejecucion"
        ]
    )

    assert (
        received[
            "id_ejecucion_entrenamiento"
        ]
        == response[
            "id_ejecucion_entrenamiento"
        ]
    )


def test_representative_cases_are_concordant(
    response: dict[str, Any],
) -> None:
    cases = (
        choose_representative_cases(
            response
        )
    )

    assert (
        set(cases)
        == {
            "A",
            "B",
            "C",
        }
    )

    for (
        category,
        expected_publication_id,
    ) in (
        EXPECTED_REPRESENTATIVE_IDS.items()
    ):
        case = cases[
            category
        ]

        assert (
            case[
                "publication_id"
            ]
            == expected_publication_id
        )

        assert (
            case[
                "categoria_ss_ekmeans"
            ]
            == category
        )

        assert (
            case[
                "categoria_sustituto"
            ]
            == category
        )

        assert (
            case[
                "concordancia"
            ]
            is True
        )

        assert (
            case[
                "id_ejecucion"
            ]
            == response[
                "id_ejecucion"
            ]
        )

        assert (
            case[
                "id_ejecucion_entrenamiento"
            ]
            == response[
                "id_ejecucion_entrenamiento"
            ]
        )


def test_concordance_rows_preserve_execution_ids(
    response: dict[str, Any],
) -> None:
    transition_rows = (
        build_concordance_transition_rows(
            response
        )
    )

    detail_rows = (
        build_concordance_detail_rows(
            response[
                "predicciones"
            ]
        )
    )

    assert (
        len(transition_rows)
        == 9
    )

    assert (
        sum(
            int(
                row[
                    "cantidad"
                ]
            )
            for row
            in transition_rows
        )
        == EXPECTED_TOTAL
    )

    assert (
        len(detail_rows)
        == EXPECTED_TOTAL
    )

    assert (
        sum(
            (
                row[
                    "concordancia"
                ]
                is False
            )
            for row
            in detail_rows
        )
        == EXPECTED_DISCREPANCIES
    )

    for row in [
        *transition_rows,
        *detail_rows,
    ]:
        assert (
            row[
                "id_ejecucion"
            ]
            == response[
                "id_ejecucion"
            ]
        )

        assert (
            row[
                "id_ejecucion_entrenamiento"
            ]
            == response[
                "id_ejecucion_entrenamiento"
            ]
        )


def test_validation_rejects_an_inconsistent_execution_id(
    response: dict[str, Any],
) -> None:
    altered = deepcopy(
        response
    )

    altered[
        "predicciones"
    ][0][
        "id_ejecucion"
    ] = "shap-alterado"

    with pytest.raises(
        ValueError,
        match=(
            "id_ejecucion "
            "inconsistente"
        ),
    ):
        validate_response(
            altered,
            expected_products=EXPECTED_TOTAL,
        )


def test_validation_rejects_an_inconsistent_training_execution_id(
    response: dict[str, Any],
) -> None:
    altered = deepcopy(
        response
    )

    altered[
        "predicciones"
    ][0][
        "id_ejecucion_entrenamiento"
    ] = "train-alterado"

    with pytest.raises(
        ValueError,
        match=(
            "id_ejecucion_entrenamiento "
            "inconsistente"
        ),
    ):
        validate_response(
            altered,
            expected_products=EXPECTED_TOTAL,
        )


def test_validation_rejects_a_duplicated_publication_id(
    response: dict[str, Any],
) -> None:
    altered = deepcopy(
        response
    )

    altered[
        "predicciones"
    ][1][
        "publication_id"
    ] = altered[
        "predicciones"
    ][0][
        "publication_id"
    ]

    with pytest.raises(
        ValueError,
        match=(
            "publication_id "
            "duplicado"
        ),
    ):
        validate_response(
            altered,
            expected_products=EXPECTED_TOTAL,
        )


def test_validation_rejects_an_inconsistent_concordance_indicator(
    response: dict[str, Any],
) -> None:
    altered = deepcopy(
        response
    )

    altered[
        "predicciones"
    ][0][
        "concordancia"
    ] = not altered[
        "predicciones"
    ][0][
        "concordancia"
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Indicador de concordancia "
            "inconsistente"
        ),
    ):
        validate_response(
            altered,
            expected_products=EXPECTED_TOTAL,
        )


def test_validation_rejects_a_truncated_response(
    response: dict[str, Any],
) -> None:
    altered = deepcopy(
        response
    )

    altered[
        "shap_config"
    ][
        "truncated"
    ] = True

    with pytest.raises(
        ValueError,
        match="truncada",
    ):
        validate_response(
            altered,
            expected_products=EXPECTED_TOTAL,
        )


def test_generator_creates_traceable_final_artifacts(
    response: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = (
        tmp_path
        / "shap_results"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            (
                "generate_shap_"
                "result_figures.py"
            ),
            "--explain-response",
            str(
                RESPONSE_PATH
            ),
            "--output-dir",
            str(
                output_dir
            ),
            "--dpi",
            "72",
            "--expected-products",
            str(
                EXPECTED_TOTAL
            ),
        ],
    )

    generate_figures_main()

    generated_files = {
        path.name
        for path
        in output_dir.iterdir()
        if path.is_file()
    }

    assert (
        EXPECTED_FIGURES
        <= generated_files
    )

    assert (
        EXPECTED_TABLES
        <= generated_files
    )

    assert (
        "resumen_resultados_shap.json"
        in generated_files
    )

    for figure_name in (
        EXPECTED_FIGURES
    ):
        figure_path = (
            output_dir
            / figure_name
        )

        assert (
            figure_path.stat().st_size
            > 0
        )

    for table_name in (
        EXPECTED_TABLES
    ):
        (
            columns,
            rows,
        ) = read_csv_rows(
            output_dir
            / table_name
        )

        assert (
            columns[:2]
            == [
                "id_ejecucion",
                "id_ejecucion_entrenamiento",
            ]
        )

        for row in rows:
            assert (
                row[
                    "id_ejecucion"
                ]
                == response[
                    "id_ejecucion"
                ]
            )

            assert (
                row[
                    "id_ejecucion_entrenamiento"
                ]
                == response[
                    "id_ejecucion_entrenamiento"
                ]
            )

    (
        detail_columns,
        detail_rows,
    ) = read_csv_rows(
        output_dir
        / (
            "tabla_shap_07_"
            "concordancia_publicaciones.csv"
        )
    )

    assert (
        detail_columns[:2]
        == [
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
        ]
    )

    assert (
        len(detail_rows)
        == EXPECTED_TOTAL
    )

    assert (
        sum(
            (
                row[
                    "concordancia"
                ]
                == "False"
            )
            for row
            in detail_rows
        )
        == EXPECTED_DISCREPANCIES
    )

    summary_path = (
        output_dir
        / "resumen_resultados_shap.json"
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        generated_summary = (
            json.load(
                input_file
            )
        )

    assert (
        generated_summary[
            "id_ejecucion"
        ]
        == response[
            "id_ejecucion"
        ]
    )

    assert (
        generated_summary[
            "id_ejecucion_entrenamiento"
        ]
        == response[
            "id_ejecucion_entrenamiento"
        ]
    )

    assert (
        generated_summary[
            "version_artefacto"
        ]
        == 4
    )

    assert (
        generated_summary[
            "resumen_concordancia"
        ][
            "coincidencias"
        ]
        == EXPECTED_AGREEMENTS
    )

    assert (
        generated_summary[
            "resumen_concordancia"
        ][
            "discrepancias"
        ]
        == EXPECTED_DISCREPANCIES
    )

    assert (
        generated_summary[
            "validacion_aditividad"
        ][
            "explicaciones_evaluadas"
        ]
        == EXPECTED_TOTAL
    )

    assert (
        generated_summary[
            "validacion_aditividad"
        ][
            "explicaciones_fuera_tolerancia"
        ]
        == 0
    )