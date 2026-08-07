from __future__ import annotations

"""
Genera las figuras y tablas finales de explicabilidad a partir de una
respuesta JSON ya producida por:

    POST /api/explainability/autosklearn/explain

Este script NO vuelve a entrenar AutoSklearn, NO ejecuta SS-E-KMeans y
NO recalcula SHAP. Solo reutiliza los resultados guardados en el JSON.

Ejemplo en PowerShell:

    python generate_shap_result_figures.py `
        --explain-response data/shap_explain_response.json `
        --output-dir artifacts/shap_results
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


CLASSES = ("A", "B", "C")
REQUIRED_ARTIFACT_VERSION = 4

FEATURE_LABELS = {
    "ventas_30d": "Ventas (30 días)",
    "visitas_30d": "Visitas (30 días)",
    "precio_actual": "Precio actual",
    "stock_actual": "Stock actual",
    "en_promocion": "En promoción",
}

CLASS_COLORS = {
    "A": "#163A5F",
    "B": "#4F78A8",
    "C": "#A7B4C2",
}

COLOR_TEXT = "#1F2933"
COLOR_AXIS = "#66727D"
COLOR_GRID = "#D9DEE3"
COLOR_POSITIVE = "#315F8C"
COLOR_NEGATIVE = "#A0443E"
COLOR_NEUTRAL = "#A7B4C2"
COLOR_SUCCESS = "#496F5D"
COLOR_FAILURE = "#A0443E"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera figuras y tablas de SHAP reutilizando una respuesta "
            "JSON existente del endpoint de explicabilidad."
        )
    )

    parser.add_argument(
        "--explain-response",
        type=Path,
        required=True,
        help=(
            "Respuesta JSON de "
            "POST /api/explainability/autosklearn/explain."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/shap_results"),
        help="Directorio donde se guardarán figuras y tablas.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolución de las figuras PNG.",
    )

    parser.add_argument(
        "--expected-products",
        type=int,
        default=832,
        help=(
            "Cantidad exacta de publicaciones esperada. "
            "Para los resultados finales de la memoria se utilizan 832."
        ),
    )

    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.edgecolor": COLOR_AXIS,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "text.color": COLOR_TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}"
        )

    raw_text = path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not raw_text:
        raise ValueError(
            f"El archivo está vacío: {path}"
        )

    try:
        content = json.loads(raw_text)

    except json.JSONDecodeError as error:
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")

        if first_brace == -1 or last_brace == -1:
            raise ValueError(
                f"No se encontró un objeto JSON válido en {path}."
            ) from error

        try:
            content = json.loads(
                raw_text[
                    first_brace:last_brace + 1
                ]
            )

        except json.JSONDecodeError as nested_error:
            raise ValueError(
                "El archivo contiene JSON inválido. "
                f"Línea {nested_error.lineno}, "
                f"columna {nested_error.colno}: "
                f"{nested_error.msg}"
            ) from nested_error

    if not isinstance(content, dict):
        raise TypeError(
            "La respuesta de explicabilidad debe ser un objeto JSON."
        )

    return content


def build_concordance_summary(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matrix = {
        ss_category: {
            surrogate_category: 0
            for surrogate_category in CLASSES
        }
        for ss_category in CLASSES
    }

    comparable = 0
    agreements = 0

    for prediction in predictions:
        ss_category = prediction.get(
            "categoria_ss_ekmeans"
        )

        surrogate_category = prediction.get(
            "categoria_sustituto"
        )

        if (
            ss_category is None
            or surrogate_category is None
        ):
            continue

        ss_category = str(ss_category)
        surrogate_category = str(
            surrogate_category
        )

        if (
            ss_category not in CLASSES
            or surrogate_category not in CLASSES
        ):
            raise ValueError(
                "Las categorías deben pertenecer a "
                f"{CLASSES}: "
                f"SS-EKMeans={ss_category!r}, "
                f"sustituto={surrogate_category!r}."
            )

        comparable += 1
        agreements += (
            ss_category == surrogate_category
        )

        matrix[
            ss_category
        ][
            surrogate_category
        ] += 1

    discrepancies = (
        comparable - agreements
    )

    rate = (
        agreements / comparable
        if comparable
        else None
    )

    discrepancy_transitions = [
        {
            "categoria_ss_ekmeans": (
                ss_category
            ),
            "categoria_sustituto": (
                surrogate_category
            ),
            "cantidad": (
                matrix[
                    ss_category
                ][
                    surrogate_category
                ]
            ),
        }
        for ss_category in CLASSES
        for surrogate_category in CLASSES
        if (
            ss_category != surrogate_category
            and matrix[
                ss_category
            ][
                surrogate_category
            ] > 0
        )
    ]

    return {
        "total_explicados": len(
            predictions
        ),

        "total_comparables": (
            comparable
        ),

        "coincidencias": (
            agreements
        ),

        "discrepancias": (
            discrepancies
        ),

        "sin_categoria_ss_ekmeans": (
            len(predictions)
            - comparable
        ),

        "tasa_concordancia": (
            rate
        ),

        "porcentaje_concordancia": (
            rate * 100
            if rate is not None
            else None
        ),

        "matriz_concordancia": (
            matrix
        ),

        "discrepancias_por_transicion": (
            discrepancy_transitions
        ),
    }


def execution_trace(
    response: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "id_ejecucion": str(
            response[
                "id_ejecucion"
            ]
        ),

        "id_ejecucion_entrenamiento": str(
            response[
                "id_ejecucion_entrenamiento"
            ]
        ),
    }


def validate_response(
    response: Mapping[str, Any],
    expected_products: int | None = None,
) -> None:
    required_keys = (
        "id_ejecucion",
        "tipo_ejecucion",
        "fecha_ejecucion_utc",
        "id_ejecucion_entrenamiento",
        "fecha_ejecucion_entrenamiento_utc",
        "version_artefacto",
        "variables_explicadas",
        "metrics_entrenamiento",
        "importancia_global",
        "importancia_por_clase",
        "predicciones",
        "shap_config",
        "validacion_aditividad",
        "resumen_concordancia",
    )

    missing = [
        key
        for key in required_keys
        if key not in response
    ]

    if missing:
        raise ValueError(
            "La respuesta no contiene las claves requeridas: "
            f"{missing}"
        )

    string_trace_keys = (
        "id_ejecucion",
        "tipo_ejecucion",
        "fecha_ejecucion_utc",
        "id_ejecucion_entrenamiento",
        "fecha_ejecucion_entrenamiento_utc",
    )

    for key in string_trace_keys:
        value = response.get(key)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"La trazabilidad requiere {key!r} "
                "como texto no vacío."
            )

    if (
        response["tipo_ejecucion"]
        != "explicabilidad_shap"
    ):
        raise ValueError(
            "tipo_ejecucion debe ser "
            "'explicabilidad_shap'."
        )

    if (
        int(
            response[
                "version_artefacto"
            ]
        )
        != REQUIRED_ARTIFACT_VERSION
    ):
        raise ValueError(
            "Se requieren artefactos versión "
            f"{REQUIRED_ARTIFACT_VERSION}; "
            "la respuesta informa "
            f"{response['version_artefacto']!r}."
        )

    variables = response[
        "variables_explicadas"
    ]

    predictions = response[
        "predicciones"
    ]

    if (
        not isinstance(variables, list)
        or not variables
    ):
        raise ValueError(
            "variables_explicadas debe ser "
            "una lista no vacía."
        )

    if (
        not isinstance(predictions, list)
        or not predictions
    ):
        raise ValueError(
            "predicciones debe ser "
            "una lista no vacía."
        )

    shap_config = response[
        "shap_config"
    ]

    if not isinstance(
        shap_config,
        Mapping,
    ):
        raise ValueError(
            "shap_config debe ser un objeto."
        )

    products_received = int(
        shap_config.get(
            "productos_recibidos",
            -1,
        )
    )

    products_explained = int(
        shap_config.get(
            "productos_explicados",
            -1,
        )
    )

    truncated = shap_config.get(
        "truncated"
    )

    if products_explained != len(
        predictions
    ):
        raise ValueError(
            "productos_explicados no coincide "
            "con la cantidad de predicciones."
        )

    if (
        products_received != len(predictions)
        or truncated is not False
    ):
        raise ValueError(
            "La respuesta está truncada o no "
            "contiene todas las publicaciones recibidas."
        )

    if expected_products is not None:
        if expected_products < 1:
            raise ValueError(
                "expected_products debe ser "
                "mayor que cero."
            )

        if len(predictions) != expected_products:
            raise ValueError(
                f"Se esperaban {expected_products} "
                "publicaciones, pero la respuesta contiene "
                f"{len(predictions)}."
            )

    response_execution_id = str(
        response[
            "id_ejecucion"
        ]
    )

    training_execution_id = str(
        response[
            "id_ejecucion_entrenamiento"
        ]
    )

    publication_ids: set[str] = set()

    for position, prediction in enumerate(
        predictions
    ):
        required_prediction_keys = (
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
            "publication_id",
            "categoria_ss_ekmeans",
            "categoria_sustituto",
            "concordancia",
        )

        missing_prediction_keys = [
            key
            for key in required_prediction_keys
            if key not in prediction
        ]

        if missing_prediction_keys:
            raise ValueError(
                f"predicciones[{position}] "
                "no contiene: "
                f"{missing_prediction_keys}. "
                "El JSON debe provenir del worker "
                "con artefactos versión 4."
            )

        if (
            str(
                prediction[
                    "id_ejecucion"
                ]
            )
            != response_execution_id
        ):
            raise ValueError(
                "id_ejecucion inconsistente "
                f"en la fila {position}."
            )

        if (
            str(
                prediction[
                    "id_ejecucion_entrenamiento"
                ]
            )
            != training_execution_id
        ):
            raise ValueError(
                "id_ejecucion_entrenamiento "
                "inconsistente en la fila "
                f"{position}."
            )

        publication_id = str(
            prediction[
                "publication_id"
            ]
        ).strip()

        if not publication_id:
            raise ValueError(
                "publication_id vacío "
                f"en la fila {position}."
            )

        if publication_id in publication_ids:
            raise ValueError(
                "publication_id duplicado: "
                f"{publication_id}."
            )

        publication_ids.add(
            publication_id
        )

        ss_category = str(
            prediction[
                "categoria_ss_ekmeans"
            ]
        )

        surrogate_category = str(
            prediction[
                "categoria_sustituto"
            ]
        )

        concordance = prediction[
            "concordancia"
        ]

        if ss_category not in CLASSES:
            raise ValueError(
                "Categoría SS-EKMeans inválida "
                f"en la fila {position}: "
                f"{ss_category!r}."
            )

        if surrogate_category not in CLASSES:
            raise ValueError(
                "Categoría sustituta inválida "
                f"en la fila {position}: "
                f"{surrogate_category!r}."
            )

        if not isinstance(
            concordance,
            bool,
        ):
            raise ValueError(
                "concordancia debe ser booleana "
                f"en la fila {position}."
            )

        if (
            concordance
            != (
                ss_category
                == surrogate_category
            )
        ):
            raise ValueError(
                "Indicador de concordancia "
                "inconsistente en la fila "
                f"{position}."
            )

        if (
            "prediccion" in prediction
            and str(
                prediction[
                    "prediccion"
                ]
            )
            != surrogate_category
        ):
            raise ValueError(
                "prediccion y categoria_sustituto "
                "difieren en la fila "
                f"{position}."
            )

    expected_summary = (
        build_concordance_summary(
            predictions
        )
    )

    received_summary = response[
        "resumen_concordancia"
    ]

    if not isinstance(
        received_summary,
        Mapping,
    ):
        raise ValueError(
            "resumen_concordancia debe ser un objeto."
        )

    if (
        received_summary.get(
            "id_ejecucion"
        )
        != response_execution_id
    ):
        raise ValueError(
            "El ID del resumen de concordancia "
            "no coincide con la ejecución SHAP."
        )

    if (
        received_summary.get(
            "id_ejecucion_entrenamiento"
        )
        != training_execution_id
    ):
        raise ValueError(
            "El ID de entrenamiento del resumen "
            "de concordancia es inconsistente."
        )

    exact_summary_keys = (
        "total_explicados",
        "total_comparables",
        "coincidencias",
        "discrepancias",
        "sin_categoria_ss_ekmeans",
        "matriz_concordancia",
        "discrepancias_por_transicion",
    )

    for key in exact_summary_keys:
        if (
            received_summary.get(key)
            != expected_summary[key]
        ):
            raise ValueError(
                "resumen_concordancia"
                f"[{key!r}] no coincide "
                "con las predicciones."
            )

    for key in (
        "tasa_concordancia",
        "porcentaje_concordancia",
    ):
        expected_value = (
            expected_summary[key]
        )

        received_value = (
            received_summary.get(key)
        )

        if (
            expected_value is None
            or received_value is None
        ):
            if expected_value is not received_value:
                raise ValueError(
                    "resumen_concordancia"
                    f"[{key!r}] es inconsistente."
                )

        elif not math.isclose(
            float(received_value),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "resumen_concordancia"
                f"[{key!r}] no coincide "
                "con las predicciones."
            )


def feature_label(
    feature: str,
) -> str:
    return FEATURE_LABELS.get(
        feature,
        feature.replace(
            "_",
            " ",
        ).capitalize(),
    )


def format_decimal(
    value: float,
    digits: int = 4,
) -> str:
    return (
        f"{value:.{digits}f}"
        .replace(
            ".",
            ",",
        )
    )


def format_feature_value(
    value: Any,
) -> str:
    if isinstance(value, bool):
        return (
            "Sí"
            if value
            else "No"
        )

    if isinstance(
        value,
        (int, np.integer),
    ):
        return (
            f"{int(value):,}"
            .replace(
                ",",
                ".",
            )
        )

    if isinstance(
        value,
        (float, np.floating),
    ):
        numeric = float(value)

        if numeric.is_integer():
            return (
                f"{int(numeric):,}"
                .replace(
                    ",",
                    ".",
                )
            )

        return format_decimal(
            numeric,
            2,
        )

    return str(value)


def save_figure(
    figure: Figure,
    path: Path,
    dpi: int,
) -> Path:
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(
        figure
    )

    return path


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Path:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(columns),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def to_json_serializable(
    value: Any,
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): to_json_serializable(
                content
            )
            for key, content in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            to_json_serializable(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    return value


def write_json(
    path: Path,
    content: Mapping[str, Any],
) -> Path:
    path.write_text(
        json.dumps(
            to_json_serializable(
                content
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def add_grid(
    axis: Axes,
    axis_name: str = "y",
) -> None:
    axis.grid(
        axis=axis_name,
        color=COLOR_GRID,
        linewidth=0.8,
        alpha=0.75,
    )

    axis.set_axisbelow(
        True
    )


def generate_surrogate_fidelity_figure(
    response: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
) -> tuple[
    Path,
    list[dict[str, Any]],
]:
    metrics = response[
        "metrics_entrenamiento"
    ]

    trace = execution_trace(
        response
    )

    metric_rows = [
        {
            **trace,
            "metrica": "Exactitud",
            "valor": float(
                metrics["accuracy"]
            ),
        },
        {
            **trace,
            "metrica": (
                "Exactitud balanceada"
            ),
            "valor": float(
                metrics[
                    "balanced_accuracy"
                ]
            ),
        },
        {
            **trace,
            "metrica": "F1 macro",
            "valor": float(
                metrics["macro_f1"]
            ),
        },
    ]

    matrix = np.asarray(
        metrics[
            "confusion_matrix"
        ],
        dtype=int,
    )

    classes = [
        str(item)
        for item in metrics.get(
            "classes",
            CLASSES,
        )
    ]

    if matrix.shape != (
        len(classes),
        len(classes),
    ):
        raise ValueError(
            "La matriz de confusión no "
            "coincide con las clases informadas."
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            11.4,
            5.2,
        ),
        gridspec_kw={
            "width_ratios": [
                1.05,
                1.0,
            ]
        },
    )

    (
        axis_metrics,
        axis_matrix,
    ) = axes

    metric_names = [
        row["metrica"]
        for row in metric_rows
    ]

    metric_values = [
        row["valor"]
        for row in metric_rows
    ]

    positions = np.arange(
        len(metric_rows)
    )

    bars = axis_metrics.barh(
        positions,
        metric_values,
        color="#4F78A8",
        edgecolor="#315F8C",
        height=0.58,
    )

    axis_metrics.set_yticks(
        positions
    )

    axis_metrics.set_yticklabels(
        metric_names
    )

    axis_metrics.invert_yaxis()

    axis_metrics.set_xlim(
        0.0,
        1.05,
    )

    axis_metrics.set_xlabel(
        "Valor de la métrica"
    )

    axis_metrics.set_title(
        "Fidelidad en el conjunto de prueba"
    )

    add_grid(
        axis_metrics,
        "x",
    )

    for bar, value in zip(
        bars,
        metric_values,
    ):
        axis_metrics.text(
            min(
                value + 0.015,
                1.015,
            ),
            (
                bar.get_y()
                + bar.get_height() / 2
            ),
            format_decimal(
                value,
                4,
            ),
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
        )

    image = axis_matrix.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=max(
            int(matrix.max()),
            1,
        ),
        aspect="equal",
    )

    axis_matrix.set_xticks(
        np.arange(
            len(classes)
        )
    )

    axis_matrix.set_yticks(
        np.arange(
            len(classes)
        )
    )

    axis_matrix.set_xticklabels(
        classes
    )

    axis_matrix.set_yticklabels(
        classes
    )

    axis_matrix.set_xlabel(
        "Predicción del modelo sustituto"
    )

    axis_matrix.set_ylabel(
        "Categoría de SS-E-KMeans"
    )

    axis_matrix.set_title(
        "Matriz de confusión"
    )

    threshold = (
        float(
            matrix.max()
        )
        / 2.0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = int(
                matrix[
                    row_index,
                    column_index,
                ]
            )

            axis_matrix.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=(
                    "white"
                    if value > threshold
                    else COLOR_TEXT
                ),
            )

    colorbar = figure.colorbar(
        image,
        ax=axis_matrix,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Cantidad de publicaciones"
    )

    figure.suptitle(
        "Fidelidad del modelo sustituto AutoSklearn",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    figure.tight_layout()

    path = (
        output_dir
        / "figura_shap_01_fidelidad_sustituto.png"
    )

    return (
        save_figure(
            figure,
            path,
            dpi,
        ),
        metric_rows,
    )


def normalize_global_importance(
    response: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_rows = response[
        "importancia_global"
    ]

    trace = execution_trace(
        response
    )

    total = sum(
        float(
            row[
                "mean_abs_shap"
            ]
        )
        for row in raw_rows
    )

    rows: list[
        dict[str, Any]
    ] = []

    for row in raw_rows:
        value = float(
            row[
                "mean_abs_shap"
            ]
        )

        rows.append(
            {
                **trace,

                "feature": str(
                    row["feature"]
                ),

                "variable": feature_label(
                    str(
                        row["feature"]
                    )
                ),

                "mean_abs_shap": (
                    value
                ),

                "porcentaje_relativo": (
                    value / total * 100.0
                    if total > 0
                    else 0.0
                ),
            }
        )

    return sorted(
        rows,
        key=lambda item: (
            item[
                "mean_abs_shap"
            ]
        ),
        reverse=True,
    )


def generate_global_importance_figure(
    global_rows: Sequence[
        Mapping[str, Any]
    ],
    output_dir: Path,
    dpi: int,
) -> Path:
    ordered = list(
        reversed(
            global_rows
        )
    )

    labels = [
        str(
            row["variable"]
        )
        for row in ordered
    ]

    values = [
        float(
            row[
                "mean_abs_shap"
            ]
        )
        for row in ordered
    ]

    percentages = [
        float(
            row[
                "porcentaje_relativo"
            ]
        )
        for row in ordered
    ]

    figure, axis = plt.subplots(
        figsize=(
            8.4,
            5.4,
        )
    )

    positions = np.arange(
        len(ordered)
    )

    bars = axis.barh(
        positions,
        values,
        color=[
            "#A7B4C2",
            "#4F78A8",
            "#163A5F",
        ][
            -len(ordered):
        ],
        edgecolor=COLOR_AXIS,
        height=0.62,
    )

    axis.set_yticks(
        positions
    )

    axis.set_yticklabels(
        labels
    )

    axis.set_xlabel(
        "Importancia media absoluta de SHAP"
    )

    axis.set_title(
        "Importancia global de las variables"
    )

    add_grid(
        axis,
        "x",
    )

    maximum = (
        max(values)
        if values
        else 1.0
    )

    axis.set_xlim(
        0,
        maximum * 1.32,
    )

    for (
        bar,
        value,
        percentage,
    ) in zip(
        bars,
        values,
        percentages,
    ):
        axis.text(
            (
                value
                + maximum * 0.025
            ),
            (
                bar.get_y()
                + bar.get_height() / 2
            ),
            (
                f"{format_decimal(value, 4)} "
                f"({format_decimal(percentage, 2)} %)"
            ),
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
        )

    figure.text(
        0.5,
        0.01,
        (
            "La magnitud indica influencia promedio, "
            "no causalidad ni dirección del efecto."
        ),
        ha="center",
        fontsize=10,
        color=COLOR_AXIS,
    )

    figure.tight_layout(
        rect=(
            0,
            0.06,
            1,
            1,
        )
    )

    path = (
        output_dir
        / "figura_shap_02_importancia_global.png"
    )

    return save_figure(
        figure,
        path,
        dpi,
    )


def normalize_class_importance(
    response: Mapping[str, Any],
    features: Sequence[str],
) -> list[dict[str, Any]]:
    importance_by_class = response[
        "importancia_por_clase"
    ]

    trace = execution_trace(
        response
    )

    rows: list[
        dict[str, Any]
    ] = []

    for class_name in CLASSES:
        class_rows = (
            importance_by_class.get(
                class_name,
                [],
            )
        )

        value_by_feature = {
            str(
                row["feature"]
            ): float(
                row[
                    "mean_abs_shap"
                ]
            )
            for row in class_rows
        }

        for feature in features:
            rows.append(
                {
                    **trace,

                    "categoria": (
                        class_name
                    ),

                    "feature": feature,

                    "variable": (
                        feature_label(
                            feature
                        )
                    ),

                    "mean_abs_shap": float(
                        value_by_feature.get(
                            feature,
                            0.0,
                        )
                    ),
                }
            )

    return rows


def generate_class_importance_figure(
    class_rows: Sequence[
        Mapping[str, Any]
    ],
    features: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axis = plt.subplots(
        figsize=(
            9.2,
            5.8,
        )
    )

    base_positions = np.arange(
        len(features)
    )

    bar_width = 0.22

    for (
        offset_index,
        class_name,
    ) in enumerate(
        CLASSES
    ):
        values = [
            next(
                float(
                    row[
                        "mean_abs_shap"
                    ]
                )
                for row in class_rows
                if (
                    row[
                        "categoria"
                    ]
                    == class_name
                    and row[
                        "feature"
                    ]
                    == feature
                )
            )
            for feature in features
        ]

        positions = (
            base_positions
            + (
                offset_index - 1
            )
            * bar_width
        )

        bars = axis.bar(
            positions,
            values,
            width=bar_width,
            label=(
                f"Categoría {class_name}"
            ),
            color=(
                CLASS_COLORS[
                    class_name
                ]
            ),
            edgecolor=COLOR_AXIS,
            linewidth=0.6,
        )

        for bar, value in zip(
            bars,
            values,
        ):
            axis.text(
                (
                    bar.get_x()
                    + bar.get_width() / 2
                ),
                value + 0.008,
                format_decimal(
                    value,
                    3,
                ),
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=(
                    90
                    if len(features) > 4
                    else 0
                ),
            )

    axis.set_xticks(
        base_positions
    )

    axis.set_xticklabels(
        [
            feature_label(feature)
            for feature in features
        ]
    )

    axis.set_ylabel(
        "Importancia media absoluta de SHAP"
    )

    axis.set_title(
        "Importancia de las variables por categoría"
    )

    axis.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
    )

    add_grid(
        axis,
        "y",
    )

    maximum = max(
        float(
            row[
                "mean_abs_shap"
            ]
        )
        for row in class_rows
    )

    axis.set_ylim(
        0,
        maximum * 1.24,
    )

    figure.tight_layout()

    path = (
        output_dir
        / "figura_shap_03_importancia_por_categoria.png"
    )

    return save_figure(
        figure,
        path,
        dpi,
    )


def choose_representative_cases(
    response: Mapping[str, Any],
) -> dict[
    str,
    dict[str, Any],
]:
    predictions = response[
        "predicciones"
    ]

    selected: dict[
        str,
        dict[str, Any],
    ] = {}

    for class_name in CLASSES:
        class_predictions = [
            row
            for row in predictions
            if (
                str(
                    row.get(
                        "categoria_sustituto"
                    )
                )
                == class_name
                and class_name
                in row.get(
                    "probabilidades",
                    {},
                )
            )
        ]

        if not class_predictions:
            continue

        probabilities = [
            float(
                row[
                    "probabilidades"
                ][
                    class_name
                ]
            )
            for row in class_predictions
        ]

        class_median = float(
            median(
                probabilities
            )
        )

        concordant_candidates = [
            row
            for row in class_predictions
            if (
                row.get(
                    "concordancia"
                )
                is True
            )
        ]

        if not concordant_candidates:
            continue

        selected[
            class_name
        ] = min(
            concordant_candidates,
            key=lambda row: (
                abs(
                    float(
                        row[
                            "probabilidades"
                        ][
                            class_name
                        ]
                    )
                    - class_median
                ),
                str(
                    row.get(
                        "publication_id",
                        "",
                    )
                ),
            ),
        )

    missing_classes = [
        class_name
        for class_name in CLASSES
        if class_name not in selected
    ]

    if missing_classes:
        raise ValueError(
            "No existen casos concordantes para "
            "seleccionar las categorías: "
            f"{missing_classes}."
        )

    return selected


def contribution_color(
    value: float,
) -> str:
    if value > 0:
        return COLOR_POSITIVE

    if value < 0:
        return COLOR_NEGATIVE

    return COLOR_NEUTRAL


def representative_case_rows(
    cases: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for class_name in CLASSES:
        case = cases.get(
            class_name
        )

        if case is None:
            continue

        contributions = case.get(
            "contribuciones",
            case.get(
                "top_contribuciones",
                [],
            ),
        )

        probability = float(
            case[
                "probabilidades"
            ][
                class_name
            ]
        )

        for contribution in contributions:
            rows.append(
                {
                    "id_ejecucion": str(
                        case[
                            "id_ejecucion"
                        ]
                    ),

                    "id_ejecucion_entrenamiento": str(
                        case[
                            "id_ejecucion_entrenamiento"
                        ]
                    ),

                    "publication_id": str(
                        case.get(
                            "publication_id",
                            "",
                        )
                    ),

                    "categoria_ss_ekmeans": str(
                        case[
                            "categoria_ss_ekmeans"
                        ]
                    ),

                    "categoria_sustituto": str(
                        case[
                            "categoria_sustituto"
                        ]
                    ),

                    "concordancia": bool(
                        case[
                            "concordancia"
                        ]
                    ),

                    "probabilidad_predicha": (
                        probability
                    ),

                    "valor_base": float(
                        case[
                            "valor_base"
                        ]
                    ),

                    "probabilidad_reconstruida": float(
                        case[
                            "probabilidad_reconstruida"
                        ]
                    ),

                    "error_aditividad": float(
                        case[
                            "error_aditividad"
                        ]
                    ),

                    "feature": str(
                        contribution[
                            "feature"
                        ]
                    ),

                    "feature_value": (
                        contribution[
                            "feature_value"
                        ]
                    ),

                    "shap_value": float(
                        contribution[
                            "shap_value"
                        ]
                    ),
                }
            )

    return rows


def generate_local_cases_figure(
    cases: Mapping[
        str,
        Mapping[str, Any],
    ],
    output_dir: Path,
    dpi: int,
) -> Path:
    available_classes = [
        class_name
        for class_name in CLASSES
        if class_name in cases
    ]

    if not available_classes:
        raise ValueError(
            "No fue posible seleccionar "
            "casos locales representativos."
        )

    figure, axes = plt.subplots(
        len(
            available_classes
        ),
        1,
        figsize=(
            8.4,
            3.25
            * len(
                available_classes
            ),
        ),
        squeeze=False,
    )

    all_values = [
        abs(
            float(
                item[
                    "shap_value"
                ]
            )
        )
        for class_name in available_classes
        for item in cases[
            class_name
        ].get(
            "contribuciones",
            cases[
                class_name
            ].get(
                "top_contribuciones",
                [],
            ),
        )
    ]

    global_limit = (
        max(
            all_values,
            default=1.0,
        )
        * 1.25
    )

    for (
        row_index,
        class_name,
    ) in enumerate(
        available_classes
    ):
        axis = axes[
            row_index,
            0,
        ]

        case = cases[
            class_name
        ]

        contributions = list(
            case.get(
                "contribuciones",
                case.get(
                    "top_contribuciones",
                    [],
                ),
            )
        )

        contributions.sort(
            key=lambda item: abs(
                float(
                    item[
                        "shap_value"
                    ]
                )
            )
        )

        labels = [
            (
                f"{feature_label(str(item['feature']))} = "
                f"{format_feature_value(item['feature_value'])}"
            )
            for item in contributions
        ]

        values = [
            float(
                item[
                    "shap_value"
                ]
            )
            for item in contributions
        ]

        colors = [
            contribution_color(
                value
            )
            for value in values
        ]

        positions = np.arange(
            len(
                contributions
            )
        )

        bars = axis.barh(
            positions,
            values,
            color=colors,
            edgecolor=COLOR_AXIS,
            height=0.58,
        )

        axis.axvline(
            0,
            color=COLOR_AXIS,
            linewidth=1.0,
        )

        axis.set_yticks(
            positions
        )

        axis.set_yticklabels(
            labels
        )

        axis.set_xlim(
            -global_limit,
            global_limit,
        )

        axis.set_xlabel(
            "Contribución hacia la categoría sustituta "
            f"{class_name}"
        )

        axis.set_title(
            (
                "SS-EKMeans "
                f"{case['categoria_ss_ekmeans']} | "
                "Sustituto "
                f"{case['categoria_sustituto']} | "
                "Concordancia: Sí\n"
                f"{case.get('publication_id', '')}"
            ),
            loc="left",
        )

        add_grid(
            axis,
            "x",
        )

        for bar, value in zip(
            bars,
            values,
        ):
            horizontal_alignment = (
                "left"
                if value >= 0
                else "right"
            )

            offset = (
                global_limit
                * 0.018
            )

            x_position = (
                value + offset
                if value >= 0
                else value - offset
            )

            axis.text(
                x_position,
                (
                    bar.get_y()
                    + bar.get_height() / 2
                ),
                (
                    f"{value:+.4f}"
                    .replace(
                        ".",
                        ",",
                    )
                ),
                ha=horizontal_alignment,
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        probability = float(
            case[
                "probabilidades"
            ][
                class_name
            ]
        )

        base_value = float(
            case[
                "valor_base"
            ]
        )

        reconstructed = float(
            case[
                "probabilidad_reconstruida"
            ]
        )

        axis.text(
            0.02,
            0.93,
            (
                "Base = "
                f"{format_decimal(base_value, 4)}\n"
                "Predicha = "
                f"{format_decimal(probability, 4)}  |  "
                "Reconstruida = "
                f"{format_decimal(reconstructed, 4)}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color=COLOR_AXIS,
            bbox={
                "boxstyle": (
                    "round,pad=0.28"
                ),
                "facecolor": "white",
                "edgecolor": (
                    COLOR_GRID
                ),
                "alpha": 0.92,
            },
        )

    legend_items = [
        Line2D(
            [0],
            [0],
            color=COLOR_POSITIVE,
            linewidth=8,
            label=(
                "Aumenta la probabilidad "
                "de la clase explicada"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_NEGATIVE,
            linewidth=8,
            label=(
                "Disminuye la probabilidad "
                "de la clase explicada"
            ),
        ),
    ]

    figure.legend(
        handles=legend_items,
        loc="upper center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.985,
        ),
    )

    figure.suptitle(
        "Explicaciones SHAP locales "
        "de casos representativos",
        fontsize=18,
        fontweight="bold",
        y=1.025,
    )

    figure.tight_layout(
        rect=(
            0,
            0,
            1,
            0.94,
        )
    )

    path = (
        output_dir
        / "figura_shap_04_casos_representativos.png"
    )

    return save_figure(
        figure,
        path,
        dpi,
    )


def build_additivity_rows(
    response: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
]:
    tolerance = float(
        response[
            "validacion_aditividad"
        ][
            "tolerance"
        ]
    )

    predictions = response[
        "predicciones"
    ]

    rows: list[
        dict[str, Any]
    ] = []

    for prediction in predictions:
        error = abs(
            float(
                prediction[
                    "error_aditividad"
                ]
            )
        )

        rows.append(
            {
                "id_ejecucion": str(
                    prediction[
                        "id_ejecucion"
                    ]
                ),

                "id_ejecucion_entrenamiento": str(
                    prediction[
                        "id_ejecucion_entrenamiento"
                    ]
                ),

                "publication_id": str(
                    prediction.get(
                        "publication_id",
                        "",
                    )
                ),

                "categoria_ss_ekmeans": str(
                    prediction[
                        "categoria_ss_ekmeans"
                    ]
                ),

                "categoria_sustituto": str(
                    prediction[
                        "categoria_sustituto"
                    ]
                ),

                "concordancia": bool(
                    prediction[
                        "concordancia"
                    ]
                ),

                "error_aditividad": (
                    error
                ),

                "cumple_tolerancia": (
                    error <= tolerance
                ),
            }
        )

    exceptions = [
        row
        for row in rows
        if not row[
            "cumple_tolerancia"
        ]
    ]

    exceptions.sort(
        key=lambda row: (
            row[
                "error_aditividad"
            ]
        ),
        reverse=True,
    )

    return (
        rows,
        exceptions,
        tolerance,
    )


def generate_additivity_figure(
    rows: Sequence[
        Mapping[str, Any]
    ],
    tolerance: float,
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            11.4,
            5.3,
        ),
        gridspec_kw={
            "width_ratios": [
                0.9,
                1.35,
            ]
        },
    )

    (
        axis_compliance,
        axis_errors,
    ) = axes

    total = len(rows)

    compliant = sum(
        bool(
            row[
                "cumple_tolerancia"
            ]
        )
        for row in rows
    )

    failed = (
        total - compliant
    )

    compliance_percentage = (
        compliant
        / total
        * 100.0
    )

    bars = axis_compliance.bar(
        [
            "Cumplen",
            "Superan",
        ],
        [
            compliant,
            failed,
        ],
        color=[
            COLOR_SUCCESS,
            COLOR_FAILURE,
        ],
        edgecolor=COLOR_AXIS,
        width=0.58,
    )

    axis_compliance.set_ylabel(
        "Cantidad de explicaciones"
    )

    axis_compliance.set_title(
        "Cumplimiento de la tolerancia"
    )

    add_grid(
        axis_compliance,
        "y",
    )

    for bar, value in zip(
        bars,
        [
            compliant,
            failed,
        ],
    ):
        axis_compliance.text(
            (
                bar.get_x()
                + bar.get_width() / 2
            ),
            (
                value
                + max(
                    total * 0.015,
                    1,
                )
            ),
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    first_bar = bars[0]

    axis_compliance.text(
        (
            first_bar.get_x()
            + first_bar.get_width() / 2
        ),
        compliant * 0.50,
        (
            f"{format_decimal(compliance_percentage, 2)} %"
        ),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )

    top_rows = sorted(
        rows,
        key=lambda row: float(
            row[
                "error_aditividad"
            ]
        ),
        reverse=True,
    )[:10]

    top_rows = list(
        reversed(
            top_rows
        )
    )

    epsilon = max(
        tolerance * 1e-4,
        1e-12,
    )

    error_values = [
        max(
            float(
                row[
                    "error_aditividad"
                ]
            ),
            epsilon,
        )
        for row in top_rows
    ]

    labels = [
        str(
            row[
                "publication_id"
            ]
        )
        for row in top_rows
    ]

    colors = [
        (
            COLOR_FAILURE
            if float(
                row[
                    "error_aditividad"
                ]
            )
            > tolerance
            else COLOR_SUCCESS
        )
        for row in top_rows
    ]

    positions = np.arange(
        len(top_rows)
    )

    axis_errors.barh(
        positions,
        error_values,
        color=colors,
        edgecolor=COLOR_AXIS,
        height=0.62,
    )

    axis_errors.axvline(
        tolerance,
        color=COLOR_FAILURE,
        linestyle="--",
        linewidth=2,
        label=(
            "Tolerancia = "
            f"{tolerance:.1e}"
            .replace(
                ".",
                ",",
            )
        ),
    )

    axis_errors.set_xscale(
        "log"
    )

    axis_errors.set_yticks(
        positions
    )

    axis_errors.set_yticklabels(
        labels,
        fontsize=9,
    )

    axis_errors.set_xlabel(
        "Error absoluto de aditividad "
        "(escala logarítmica)"
    )

    axis_errors.set_title(
        "Mayores errores observados"
    )

    axis_errors.legend(
        frameon=False,
        loc="lower right",
    )

    add_grid(
        axis_errors,
        "x",
    )

    figure.suptitle(
        "Validación de aditividad "
        "de las explicaciones SHAP",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    figure.tight_layout()

    path = (
        output_dir
        / "figura_shap_05_validacion_aditividad.png"
    )

    return save_figure(
        figure,
        path,
        dpi,
    )


def count_predictions_by_class(
    predictions: Iterable[
        Mapping[str, Any]
    ],
    category_field: str,
) -> dict[str, int]:
    counts = {
        class_name: 0
        for class_name in CLASSES
    }

    for prediction in predictions:
        class_name = str(
            prediction.get(
                category_field,
                "",
            )
        )

        counts[
            class_name
        ] = (
            counts.get(
                class_name,
                0,
            )
            + 1
        )

    return counts


def build_concordance_transition_rows(
    response: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matrix = response[
        "resumen_concordancia"
    ][
        "matriz_concordancia"
    ]

    total = int(
        response[
            "resumen_concordancia"
        ][
            "total_comparables"
        ]
    )

    trace = execution_trace(
        response
    )

    rows: list[
        dict[str, Any]
    ] = []

    for ss_category in CLASSES:
        for surrogate_category in CLASSES:
            count = int(
                matrix[
                    ss_category
                ][
                    surrogate_category
                ]
            )

            rows.append(
                {
                    **trace,

                    "categoria_ss_ekmeans": (
                        ss_category
                    ),

                    "categoria_sustituto": (
                        surrogate_category
                    ),

                    "concordancia": (
                        ss_category
                        == surrogate_category
                    ),

                    "cantidad": (
                        count
                    ),

                    "porcentaje_total": (
                        count
                        / total
                        * 100.0
                        if total
                        else 0.0
                    ),
                }
            )

    return rows


def build_concordance_detail_rows(
    predictions: Iterable[
        Mapping[str, Any]
    ],
) -> list[dict[str, Any]]:
    return [
        {
            "id_ejecucion": str(
                prediction[
                    "id_ejecucion"
                ]
            ),

            "id_ejecucion_entrenamiento": str(
                prediction[
                    "id_ejecucion_entrenamiento"
                ]
            ),

            "publication_id": str(
                prediction.get(
                    "publication_id",
                    "",
                )
            ),

            "categoria_ss_ekmeans": str(
                prediction[
                    "categoria_ss_ekmeans"
                ]
            ),

            "categoria_sustituto": str(
                prediction[
                    "categoria_sustituto"
                ]
            ),

            "concordancia": bool(
                prediction[
                    "concordancia"
                ]
            ),
        }
        for prediction in predictions
    ]


def main() -> None:
    args = parse_arguments()

    apply_style()

    output_dir: Path = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    response = read_json(
        args.explain_response
    )

    validate_response(
        response,
        expected_products=(
            args.expected_products
        ),
    )

    features = [
        str(feature)
        for feature in response[
            "variables_explicadas"
        ]
    ]

    figures: list[Path] = []
    tables: list[Path] = []

    (
        fidelity_path,
        metric_rows,
    ) = (
        generate_surrogate_fidelity_figure(
            response=response,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    figures.append(
        fidelity_path
    )

    global_rows = (
        normalize_global_importance(
            response
        )
    )

    figures.append(
        generate_global_importance_figure(
            global_rows=global_rows,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    class_rows = (
        normalize_class_importance(
            response=response,
            features=features,
        )
    )

    figures.append(
        generate_class_importance_figure(
            class_rows=class_rows,
            features=features,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    representative_cases = (
        choose_representative_cases(
            response
        )
    )

    representative_rows = (
        representative_case_rows(
            representative_cases
        )
    )

    figures.append(
        generate_local_cases_figure(
            cases=representative_cases,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    (
        additivity_rows,
        exceptions,
        tolerance,
    ) = build_additivity_rows(
        response
    )

    figures.append(
        generate_additivity_figure(
            rows=additivity_rows,
            tolerance=tolerance,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    fidelity_csv = write_csv(
        (
            output_dir
            / "tabla_shap_01_fidelidad_sustituto.csv"
        ),
        metric_rows,
        (
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
            "metrica",
            "valor",
        ),
    )

    tables.append(
        fidelity_csv
    )

    global_csv = write_csv(
        (
            output_dir
            / "tabla_shap_02_importancia_global.csv"
        ),
        global_rows,
        (
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
            "feature",
            "variable",
            "mean_abs_shap",
            "porcentaje_relativo",
        ),
    )

    tables.append(
        global_csv
    )

    class_csv = write_csv(
        (
            output_dir
            / "tabla_shap_03_importancia_por_categoria.csv"
        ),
        class_rows,
        (
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
            "categoria",
            "feature",
            "variable",
            "mean_abs_shap",
        ),
    )

    tables.append(
        class_csv
    )

    local_csv = write_csv(
        (
            output_dir
            / "tabla_shap_04_casos_representativos.csv"
        ),
        representative_rows,
        (
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
            "publication_id",
            "categoria_ss_ekmeans",
            "categoria_sustituto",
            "concordancia",
            "probabilidad_predicha",
            "valor_base",
            "probabilidad_reconstruida",
            "error_aditividad",
            "feature",
            "feature_value",
            "shap_value",
        ),
    )

    tables.append(
        local_csv
    )

    exceptions_csv = write_csv(
        (
            output_dir
            / "tabla_shap_05_excepciones_aditividad.csv"
        ),
        exceptions,
        (
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
            "publication_id",
            "categoria_ss_ekmeans",
            "categoria_sustituto",
            "concordancia",
            "error_aditividad",
            "cumple_tolerancia",
        ),
    )

    tables.append(
        exceptions_csv
    )

    concordance_transition_rows = (
        build_concordance_transition_rows(
            response
        )
    )

    transition_csv = write_csv(
        (
            output_dir
            / "tabla_shap_06_concordancia_categorias.csv"
        ),
        concordance_transition_rows,
        (
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
            "categoria_ss_ekmeans",
            "categoria_sustituto",
            "concordancia",
            "cantidad",
            "porcentaje_total",
        ),
    )

    tables.append(
        transition_csv
    )

    concordance_detail_rows = (
        build_concordance_detail_rows(
            response[
                "predicciones"
            ]
        )
    )

    concordance_detail_csv = write_csv(
        (
            output_dir
            / "tabla_shap_07_concordancia_publicaciones.csv"
        ),
        concordance_detail_rows,
        (
            "id_ejecucion",
            "id_ejecucion_entrenamiento",
            "publication_id",
            "categoria_ss_ekmeans",
            "categoria_sustituto",
            "concordancia",
        ),
    )

    tables.append(
        concordance_detail_csv
    )

    ss_ekmeans_counts = (
        count_predictions_by_class(
            response[
                "predicciones"
            ],
            "categoria_ss_ekmeans",
        )
    )

    surrogate_counts = (
        count_predictions_by_class(
            response[
                "predicciones"
            ],
            "categoria_sustituto",
        )
    )

    shap_config = response[
        "shap_config"
    ]

    additivity = response[
        "validacion_aditividad"
    ]

    summary = {
        "source": str(
            args.explain_response
        ),

        "id_ejecucion": (
            response[
                "id_ejecucion"
            ]
        ),

        "tipo_ejecucion": (
            response[
                "tipo_ejecucion"
            ]
        ),

        "fecha_ejecucion_utc": (
            response[
                "fecha_ejecucion_utc"
            ]
        ),

        "id_ejecucion_entrenamiento": (
            response[
                "id_ejecucion_entrenamiento"
            ]
        ),

        "fecha_ejecucion_entrenamiento_utc": (
            response[
                "fecha_ejecucion_entrenamiento_utc"
            ]
        ),

        "version_artefacto": (
            response[
                "version_artefacto"
            ]
        ),

        "variables_explicadas": (
            features
        ),

        "modelo_sustituto": (
            response.get(
                "modelo"
            )
        ),

        "metricas_entrenamiento": (
            response[
                "metrics_entrenamiento"
            ]
        ),

        "configuracion_shap": (
            shap_config
        ),

        "conteos_categoria_ss_ekmeans": (
            ss_ekmeans_counts
        ),

        "conteos_categoria_sustituto": (
            surrogate_counts
        ),

        "resumen_concordancia": (
            response[
                "resumen_concordancia"
            ]
        ),

        "transiciones_concordancia": (
            concordance_transition_rows
        ),

        "importancia_global": (
            global_rows
        ),

        "casos_representativos": {
            class_name: {
                "id_ejecucion": case.get(
                    "id_ejecucion"
                ),

                "id_ejecucion_entrenamiento": (
                    case.get(
                        "id_ejecucion_entrenamiento"
                    )
                ),

                "publication_id": case.get(
                    "publication_id"
                ),

                "categoria_ss_ekmeans": (
                    case.get(
                        "categoria_ss_ekmeans"
                    )
                ),

                "categoria_sustituto": (
                    case.get(
                        "categoria_sustituto"
                    )
                ),

                "concordancia": (
                    case.get(
                        "concordancia"
                    )
                ),

                "probabilidad": (
                    case.get(
                        "probabilidades",
                        {},
                    ).get(
                        class_name
                    )
                ),

                "valor_base": (
                    case.get(
                        "valor_base"
                    )
                ),

                "probabilidad_reconstruida": (
                    case.get(
                        "probabilidad_reconstruida"
                    )
                ),

                "error_aditividad": (
                    case.get(
                        "error_aditividad"
                    )
                ),
            }
            for (
                class_name,
                case,
            ) in representative_cases.items()
        },

        "validacion_aditividad": {
            **additivity,

            "explicaciones_evaluadas": (
                len(
                    additivity_rows
                )
            ),

            "explicaciones_dentro_tolerancia": (
                sum(
                    bool(
                        row[
                            "cumple_tolerancia"
                        ]
                    )
                    for row in additivity_rows
                )
            ),

            "explicaciones_fuera_tolerancia": (
                len(
                    exceptions
                )
            ),

            "excepciones": (
                exceptions
            ),
        },

        "figuras_generadas": [
            path.name
            for path in figures
        ],

        "tablas_generadas": [
            path.name
            for path in tables
        ],
    }

    summary_path = write_json(
        (
            output_dir
            / "resumen_resultados_shap.json"
        ),
        summary,
    )

    print(
        "\nFiguras SHAP generadas:"
    )

    for path in figures:
        print(
            f"  - {path}"
        )

    print(
        "\nTablas generadas:"
    )

    for path in tables:
        print(
            f"  - {path}"
        )

    print(
        "\nResumen:"
    )

    print(
        f"  - {summary_path}"
    )

    print(
        "\nTrazabilidad:"
    )

    print(
        "  - Ejecución SHAP: "
        f"{response['id_ejecucion']}"
    )

    print(
        "  - Entrenamiento origen: "
        f"{response['id_ejecucion_entrenamiento']}"
    )

    print(
        "  - Artefacto: versión "
        f"{response['version_artefacto']}"
    )

    print(
        "\nCasos representativos seleccionados:"
    )

    for class_name in CLASSES:
        case = representative_cases.get(
            class_name
        )

        if case is None:
            continue

        probability = float(
            case[
                "probabilidades"
            ][
                class_name
            ]
        )

        print(
            "  - SS-EKMeans "
            f"{case['categoria_ss_ekmeans']} | "
            "sustituto "
            f"{case['categoria_sustituto']} | "
            "concordancia: "
            f"{'sí' if case['concordancia'] else 'no'} | "
            f"{case['publication_id']} "
            f"(p={probability:.6f})"
        )

    concordance = response[
        "resumen_concordancia"
    ]

    print(
        "\nConcordancia SS-EKMeans / sustituto:"
    )

    print(
        "  - "
        f"{concordance['coincidencias']}/"
        f"{concordance['total_comparables']} "
        "publicaciones "
        f"({concordance['porcentaje_concordancia']:.2f} %)."
    )

    print(
        "  - Discrepancias: "
        f"{concordance['discrepancias']}"
    )

    total = len(
        additivity_rows
    )

    compliant = (
        total
        - len(
            exceptions
        )
    )

    print(
        "\nAditividad:"
    )

    print(
        "  - "
        f"{compliant}/{total} explicaciones "
        "dentro de la tolerancia."
    )

    print(
        "  - Excepciones: "
        f"{len(exceptions)}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )

        raise