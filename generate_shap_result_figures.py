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
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    raw_text = path.read_text(encoding="utf-8-sig").strip()

    if not raw_text:
        raise ValueError(f"El archivo está vacío: {path}")

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
            content = json.loads(raw_text[first_brace:last_brace + 1])
        except json.JSONDecodeError as nested_error:
            raise ValueError(
                "El archivo contiene JSON inválido. "
                f"Línea {nested_error.lineno}, "
                f"columna {nested_error.colno}: "
                f"{nested_error.msg}"
            ) from nested_error

    if not isinstance(content, dict):
        raise TypeError("La respuesta de explicabilidad debe ser un objeto JSON.")

    return content


def validate_response(response: Mapping[str, Any]) -> None:
    required_keys = (
        "variables_explicadas",
        "metrics_entrenamiento",
        "importancia_global",
        "importancia_por_clase",
        "predicciones",
        "shap_config",
        "validacion_aditividad",
    )

    missing = [key for key in required_keys if key not in response]

    if missing:
        raise ValueError(
            "La respuesta no contiene las claves requeridas: "
            f"{missing}"
        )

    variables = response["variables_explicadas"]
    predictions = response["predicciones"]

    if not isinstance(variables, list) or not variables:
        raise ValueError("variables_explicadas debe ser una lista no vacía.")

    if not isinstance(predictions, list) or not predictions:
        raise ValueError("predicciones debe ser una lista no vacía.")


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def format_decimal(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def format_feature_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Sí" if value else "No"

    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", ".")

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if numeric.is_integer():
            return f"{int(numeric):,}".replace(",", ".")
        return format_decimal(numeric, 2)

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
    plt.close(figure)
    return path


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)

    return path


def to_json_serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): to_json_serializable(content)
            for key, content in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_json_serializable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def write_json(
    path: Path,
    content: Mapping[str, Any],
) -> Path:
    path.write_text(
        json.dumps(
            to_json_serializable(content),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def add_grid(axis: Axes, axis_name: str = "y") -> None:
    axis.grid(
        axis=axis_name,
        color=COLOR_GRID,
        linewidth=0.8,
        alpha=0.75,
    )
    axis.set_axisbelow(True)


def generate_surrogate_fidelity_figure(
    response: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, list[dict[str, Any]]]:
    metrics = response["metrics_entrenamiento"]

    metric_rows = [
        {
            "metrica": "Exactitud",
            "valor": float(metrics["accuracy"]),
        },
        {
            "metrica": "Exactitud balanceada",
            "valor": float(metrics["balanced_accuracy"]),
        },
        {
            "metrica": "F1 macro",
            "valor": float(metrics["macro_f1"]),
        },
    ]

    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    classes = [str(item) for item in metrics.get("classes", CLASSES)]

    if matrix.shape != (len(classes), len(classes)):
        raise ValueError(
            "La matriz de confusión no coincide con las clases informadas."
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.2),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )

    axis_metrics, axis_matrix = axes

    metric_names = [row["metrica"] for row in metric_rows]
    metric_values = [row["valor"] for row in metric_rows]
    positions = np.arange(len(metric_rows))

    bars = axis_metrics.barh(
        positions,
        metric_values,
        color="#4F78A8",
        edgecolor="#315F8C",
        height=0.58,
    )

    axis_metrics.set_yticks(positions)
    axis_metrics.set_yticklabels(metric_names)
    axis_metrics.invert_yaxis()
    axis_metrics.set_xlim(0.0, 1.05)
    axis_metrics.set_xlabel("Valor de la métrica")
    axis_metrics.set_title("Fidelidad en el conjunto de prueba")
    add_grid(axis_metrics, "x")

    for bar, value in zip(bars, metric_values):
        axis_metrics.text(
            min(value + 0.015, 1.015),
            bar.get_y() + bar.get_height() / 2,
            format_decimal(value, 4),
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
        )

    image = axis_matrix.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=max(int(matrix.max()), 1),
        aspect="equal",
    )

    axis_matrix.set_xticks(np.arange(len(classes)))
    axis_matrix.set_yticks(np.arange(len(classes)))
    axis_matrix.set_xticklabels(classes)
    axis_matrix.set_yticklabels(classes)
    axis_matrix.set_xlabel("Predicción del modelo sustituto")
    axis_matrix.set_ylabel("Categoría de SS-E-KMeans")
    axis_matrix.set_title("Matriz de confusión")

    threshold = float(matrix.max()) / 2.0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis_matrix.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if value > threshold else COLOR_TEXT,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis_matrix,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Cantidad de publicaciones")

    figure.suptitle(
        "Fidelidad del modelo sustituto AutoSklearn",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    figure.tight_layout()

    path = output_dir / "figura_shap_01_fidelidad_sustituto.png"
    return save_figure(figure, path, dpi), metric_rows


def normalize_global_importance(
    response: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_rows = response["importancia_global"]
    total = sum(float(row["mean_abs_shap"]) for row in raw_rows)

    rows: list[dict[str, Any]] = []

    for row in raw_rows:
        value = float(row["mean_abs_shap"])
        rows.append(
            {
                "feature": str(row["feature"]),
                "variable": feature_label(str(row["feature"])),
                "mean_abs_shap": value,
                "porcentaje_relativo": (
                    value / total * 100.0
                    if total > 0
                    else 0.0
                ),
            }
        )

    return sorted(
        rows,
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )


def generate_global_importance_figure(
    global_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    ordered = list(reversed(global_rows))
    labels = [str(row["variable"]) for row in ordered]
    values = [float(row["mean_abs_shap"]) for row in ordered]
    percentages = [
        float(row["porcentaje_relativo"])
        for row in ordered
    ]

    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    positions = np.arange(len(ordered))

    bars = axis.barh(
        positions,
        values,
        color=["#A7B4C2", "#4F78A8", "#163A5F"][-len(ordered):],
        edgecolor=COLOR_AXIS,
        height=0.62,
    )

    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Importancia media absoluta de SHAP")
    axis.set_title("Importancia global de las variables")
    add_grid(axis, "x")

    maximum = max(values) if values else 1.0
    axis.set_xlim(0, maximum * 1.32)

    for bar, value, percentage in zip(bars, values, percentages):
        axis.text(
            value + maximum * 0.025,
            bar.get_y() + bar.get_height() / 2,
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
            "La magnitud indica influencia promedio, no causalidad "
            "ni dirección del efecto."
        ),
        ha="center",
        fontsize=10,
        color=COLOR_AXIS,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))

    path = output_dir / "figura_shap_02_importancia_global.png"
    return save_figure(figure, path, dpi)


def normalize_class_importance(
    response: Mapping[str, Any],
    features: Sequence[str],
) -> list[dict[str, Any]]:
    importance_by_class = response["importancia_por_clase"]
    rows: list[dict[str, Any]] = []

    for class_name in CLASSES:
        class_rows = importance_by_class.get(class_name, [])
        value_by_feature = {
            str(row["feature"]): float(row["mean_abs_shap"])
            for row in class_rows
        }

        for feature in features:
            rows.append(
                {
                    "categoria": class_name,
                    "feature": feature,
                    "variable": feature_label(feature),
                    "mean_abs_shap": float(
                        value_by_feature.get(feature, 0.0)
                    ),
                }
            )

    return rows


def generate_class_importance_figure(
    class_rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 5.8))

    base_positions = np.arange(len(features))
    bar_width = 0.22

    for offset_index, class_name in enumerate(CLASSES):
        values = [
            next(
                float(row["mean_abs_shap"])
                for row in class_rows
                if (
                    row["categoria"] == class_name
                    and row["feature"] == feature
                )
            )
            for feature in features
        ]

        positions = (
            base_positions
            + (offset_index - 1) * bar_width
        )

        bars = axis.bar(
            positions,
            values,
            width=bar_width,
            label=f"Categoría {class_name}",
            color=CLASS_COLORS[class_name],
            edgecolor=COLOR_AXIS,
            linewidth=0.6,
        )

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                format_decimal(value, 3),
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90 if len(features) > 4 else 0,
            )

    axis.set_xticks(base_positions)
    axis.set_xticklabels(
        [feature_label(feature) for feature in features]
    )
    axis.set_ylabel("Importancia media absoluta de SHAP")
    axis.set_title("Importancia de las variables por categoría")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    add_grid(axis, "y")

    maximum = max(
        float(row["mean_abs_shap"])
        for row in class_rows
    )
    axis.set_ylim(0, maximum * 1.24)

    figure.tight_layout()
    path = output_dir / "figura_shap_03_importancia_por_categoria.png"
    return save_figure(figure, path, dpi)


def choose_representative_cases(
    response: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    predictions = response["predicciones"]
    selected: dict[str, dict[str, Any]] = {}

    for class_name in CLASSES:
        candidates = [
            row
            for row in predictions
            if str(row.get("prediccion")) == class_name
            and class_name in row.get("probabilidades", {})
        ]

        if not candidates:
            continue

        probabilities = [
            float(row["probabilidades"][class_name])
            for row in candidates
        ]
        class_median = float(median(probabilities))

        selected[class_name] = min(
            candidates,
            key=lambda row: (
                abs(
                    float(row["probabilidades"][class_name])
                    - class_median
                ),
                str(row.get("publication_id", "")),
            ),
        )

    return selected


def contribution_color(value: float) -> str:
    if value > 0:
        return COLOR_POSITIVE
    if value < 0:
        return COLOR_NEGATIVE
    return COLOR_NEUTRAL


def representative_case_rows(
    cases: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for class_name in CLASSES:
        case = cases.get(class_name)

        if case is None:
            continue

        contributions = case.get(
            "contribuciones",
            case.get("top_contribuciones", []),
        )

        probability = float(
            case["probabilidades"][class_name]
        )

        for contribution in contributions:
            rows.append(
                {
                    "categoria": class_name,
                    "publication_id": str(
                        case.get("publication_id", "")
                    ),
                    "probabilidad_predicha": probability,
                    "valor_base": float(case["valor_base"]),
                    "probabilidad_reconstruida": float(
                        case["probabilidad_reconstruida"]
                    ),
                    "error_aditividad": float(
                        case["error_aditividad"]
                    ),
                    "feature": str(contribution["feature"]),
                    "feature_value": contribution["feature_value"],
                    "shap_value": float(
                        contribution["shap_value"]
                    ),
                }
            )

    return rows


def generate_local_cases_figure(
    cases: Mapping[str, Mapping[str, Any]],
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
            "No fue posible seleccionar casos locales representativos."
        )

    figure, axes = plt.subplots(
        len(available_classes),
        1,
        figsize=(8.4, 3.25 * len(available_classes)),
        squeeze=False,
    )

    all_values = [
        abs(float(item["shap_value"]))
        for class_name in available_classes
        for item in cases[class_name].get(
            "contribuciones",
            cases[class_name].get("top_contribuciones", []),
        )
    ]
    global_limit = max(all_values, default=1.0) * 1.25

    for row_index, class_name in enumerate(available_classes):
        axis = axes[row_index, 0]
        case = cases[class_name]
        contributions = list(
            case.get(
                "contribuciones",
                case.get("top_contribuciones", []),
            )
        )
        contributions.sort(
            key=lambda item: abs(float(item["shap_value"]))
        )

        labels = [
            (
                f"{feature_label(str(item['feature']))} = "
                f"{format_feature_value(item['feature_value'])}"
            )
            for item in contributions
        ]
        values = [
            float(item["shap_value"])
            for item in contributions
        ]
        colors = [
            contribution_color(value)
            for value in values
        ]
        positions = np.arange(len(contributions))

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
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.set_xlim(-global_limit, global_limit)
        axis.set_xlabel(
            f"Contribución hacia la categoría {class_name}"
        )
        axis.set_title(
            (
                f"Categoría {class_name}: "
                f"{case.get('publication_id', '')}"
            ),
            loc="left",
        )
        add_grid(axis, "x")

        for bar, value in zip(bars, values):
            horizontal_alignment = "left" if value >= 0 else "right"
            offset = global_limit * 0.018
            x_position = value + offset if value >= 0 else value - offset

            axis.text(
                x_position,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.4f}".replace(".", ","),
                ha=horizontal_alignment,
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        probability = float(
            case["probabilidades"][class_name]
        )
        base_value = float(case["valor_base"])
        reconstructed = float(
            case["probabilidad_reconstruida"]
        )

        axis.text(
            0.02,
            0.93,
            (
                f"Base = {format_decimal(base_value, 4)}\n"
                f"Predicha = {format_decimal(probability, 4)}  |  "
                f"Reconstruida = {format_decimal(reconstructed, 4)}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color=COLOR_AXIS,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": COLOR_GRID,
                "alpha": 0.92,
            },
        )

    legend_items = [
        Line2D(
            [0],
            [0],
            color=COLOR_POSITIVE,
            linewidth=8,
            label="Aumenta la probabilidad de la clase explicada",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_NEGATIVE,
            linewidth=8,
            label="Disminuye la probabilidad de la clase explicada",
        ),
    ]

    figure.legend(
        handles=legend_items,
        loc="upper center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.suptitle(
        "Explicaciones SHAP locales de casos representativos",
        fontsize=18,
        fontweight="bold",
        y=1.025,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    path = output_dir / "figura_shap_04_casos_representativos.png"
    return save_figure(figure, path, dpi)


def build_additivity_rows(
    response: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
]:
    tolerance = float(
        response["validacion_aditividad"]["tolerance"]
    )
    predictions = response["predicciones"]

    rows: list[dict[str, Any]] = []

    for prediction in predictions:
        error = abs(float(prediction["error_aditividad"]))
        rows.append(
            {
                "publication_id": str(
                    prediction.get("publication_id", "")
                ),
                "categoria": str(
                    prediction.get("prediccion", "")
                ),
                "error_aditividad": error,
                "cumple_tolerancia": error <= tolerance,
            }
        )

    exceptions = [
        row
        for row in rows
        if not row["cumple_tolerancia"]
    ]
    exceptions.sort(
        key=lambda row: row["error_aditividad"],
        reverse=True,
    )

    return rows, exceptions, tolerance


def generate_additivity_figure(
    rows: Sequence[Mapping[str, Any]],
    tolerance: float,
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.3),
        gridspec_kw={"width_ratios": [0.9, 1.35]},
    )
    axis_compliance, axis_errors = axes

    total = len(rows)
    compliant = sum(
        bool(row["cumple_tolerancia"])
        for row in rows
    )
    failed = total - compliant
    compliance_percentage = compliant / total * 100.0

    bars = axis_compliance.bar(
        ["Cumplen", "Superan"],
        [compliant, failed],
        color=[COLOR_SUCCESS, COLOR_FAILURE],
        edgecolor=COLOR_AXIS,
        width=0.58,
    )
    axis_compliance.set_ylabel("Cantidad de explicaciones")
    axis_compliance.set_title("Cumplimiento de la tolerancia")
    add_grid(axis_compliance, "y")

    for bar, value in zip(bars, [compliant, failed]):
        axis_compliance.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(total * 0.015, 1),
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    first_bar = bars[0]
    axis_compliance.text(
        first_bar.get_x() + first_bar.get_width() / 2,
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
        key=lambda row: float(row["error_aditividad"]),
        reverse=True,
    )[:10]
    top_rows = list(reversed(top_rows))

    epsilon = max(tolerance * 1e-4, 1e-12)
    error_values = [
        max(float(row["error_aditividad"]), epsilon)
        for row in top_rows
    ]
    labels = [
        str(row["publication_id"])
        for row in top_rows
    ]
    colors = [
        (
            COLOR_FAILURE
            if float(row["error_aditividad"]) > tolerance
            else COLOR_SUCCESS
        )
        for row in top_rows
    ]

    positions = np.arange(len(top_rows))
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
            f"{tolerance:.1e}".replace(".", ",")
        ),
    )
    axis_errors.set_xscale("log")
    axis_errors.set_yticks(positions)
    axis_errors.set_yticklabels(labels, fontsize=9)
    axis_errors.set_xlabel("Error absoluto de aditividad (escala logarítmica)")
    axis_errors.set_title("Mayores errores observados")
    axis_errors.legend(frameon=False, loc="lower right")
    add_grid(axis_errors, "x")

    figure.suptitle(
        "Validación de aditividad de las explicaciones SHAP",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    figure.tight_layout()

    path = output_dir / "figura_shap_05_validacion_aditividad.png"
    return save_figure(figure, path, dpi)


def count_predictions_by_class(
    predictions: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASSES}

    for prediction in predictions:
        class_name = str(prediction.get("prediccion", ""))
        counts[class_name] = counts.get(class_name, 0) + 1

    return counts


def main() -> None:
    args = parse_arguments()
    apply_style()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    response = read_json(args.explain_response)
    validate_response(response)

    features = [
        str(feature)
        for feature in response["variables_explicadas"]
    ]

    figures: list[Path] = []
    tables: list[Path] = []

    fidelity_path, metric_rows = generate_surrogate_fidelity_figure(
        response=response,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    figures.append(fidelity_path)

    global_rows = normalize_global_importance(response)
    figures.append(
        generate_global_importance_figure(
            global_rows=global_rows,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    class_rows = normalize_class_importance(
        response=response,
        features=features,
    )
    figures.append(
        generate_class_importance_figure(
            class_rows=class_rows,
            features=features,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    representative_cases = choose_representative_cases(response)
    representative_rows = representative_case_rows(
        representative_cases
    )
    figures.append(
        generate_local_cases_figure(
            cases=representative_cases,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    additivity_rows, exceptions, tolerance = build_additivity_rows(
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
        output_dir / "tabla_shap_01_fidelidad_sustituto.csv",
        metric_rows,
        ("metrica", "valor"),
    )
    tables.append(fidelity_csv)

    global_csv = write_csv(
        output_dir / "tabla_shap_02_importancia_global.csv",
        global_rows,
        (
            "feature",
            "variable",
            "mean_abs_shap",
            "porcentaje_relativo",
        ),
    )
    tables.append(global_csv)

    class_csv = write_csv(
        output_dir / "tabla_shap_03_importancia_por_categoria.csv",
        class_rows,
        (
            "categoria",
            "feature",
            "variable",
            "mean_abs_shap",
        ),
    )
    tables.append(class_csv)

    local_csv = write_csv(
        output_dir / "tabla_shap_04_casos_representativos.csv",
        representative_rows,
        (
            "categoria",
            "publication_id",
            "probabilidad_predicha",
            "valor_base",
            "probabilidad_reconstruida",
            "error_aditividad",
            "feature",
            "feature_value",
            "shap_value",
        ),
    )
    tables.append(local_csv)

    exceptions_csv = write_csv(
        output_dir / "tabla_shap_05_excepciones_aditividad.csv",
        exceptions,
        (
            "publication_id",
            "categoria",
            "error_aditividad",
            "cumple_tolerancia",
        ),
    )
    tables.append(exceptions_csv)

    prediction_counts = count_predictions_by_class(
        response["predicciones"]
    )
    shap_config = response["shap_config"]
    additivity = response["validacion_aditividad"]

    summary = {
        "source": str(args.explain_response),
        "variables_explicadas": features,
        "modelo_sustituto": response.get("modelo"),
        "metricas_entrenamiento": response["metrics_entrenamiento"],
        "configuracion_shap": shap_config,
        "conteos_predichos": prediction_counts,
        "importancia_global": global_rows,
        "casos_representativos": {
            class_name: {
                "publication_id": case.get("publication_id"),
                "probabilidad": case.get(
                    "probabilidades",
                    {},
                ).get(class_name),
                "valor_base": case.get("valor_base"),
                "probabilidad_reconstruida": case.get(
                    "probabilidad_reconstruida"
                ),
                "error_aditividad": case.get(
                    "error_aditividad"
                ),
            }
            for class_name, case in representative_cases.items()
        },
        "validacion_aditividad": {
            **additivity,
            "explicaciones_evaluadas": len(additivity_rows),
            "explicaciones_dentro_tolerancia": sum(
                bool(row["cumple_tolerancia"])
                for row in additivity_rows
            ),
            "explicaciones_fuera_tolerancia": len(exceptions),
            "excepciones": exceptions,
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
        output_dir / "resumen_resultados_shap.json",
        summary,
    )

    print("\nFiguras SHAP generadas:")
    for path in figures:
        print(f"  - {path}")

    print("\nTablas generadas:")
    for path in tables:
        print(f"  - {path}")

    print("\nResumen:")
    print(f"  - {summary_path}")

    print("\nCasos representativos seleccionados:")
    for class_name in CLASSES:
        case = representative_cases.get(class_name)
        if case is None:
            continue
        probability = float(case["probabilidades"][class_name])
        print(
            f"  - {class_name}: {case['publication_id']} "
            f"(p={probability:.6f})"
        )

    total = len(additivity_rows)
    compliant = total - len(exceptions)
    print("\nAditividad:")
    print(
        f"  - {compliant}/{total} explicaciones "
        "dentro de la tolerancia."
    )
    print(f"  - Excepciones: {len(exceptions)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
