from __future__ import annotations

"""
Compara la línea base ABC por score normalizado con SS-EKMeans y genera
los CSV y las figuras para el capítulo de evaluación de la memoria.

Ejecución desde la raíz del proyecto:

    python evaluate_score_baseline_comparison.py

También se pueden cambiar las rutas:

    python evaluate_score_baseline_comparison.py \
        --input data/input_request.json \
        --output data/score_comparison

Archivos principales generados:

    comparison_summary.csv
    class_profiles.csv
    labels_comparison.csv
    transition_matrix.csv
    changed_publications.csv
    method_comparison.csv

    tabla_concentracion_comercial.csv
    tabla_estadisticos_descriptivos.csv
    tabla_calidad_concordancia.csv

    figura_distribucion_comercial.png
    figura_distribucion_comercial.pdf
    figura_matriz_transicion.png
    figura_matriz_transicion.pdf
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

# Permite ejecutar el script sin una interfaz gráfica abierta.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import adjusted_rand_score

from app.ml.core.config import get_production_config
from app.ml.core.ss_kmeans import SSEKMeans
from app.ml.metrics.clustering_metrics import evaluate_internal_metrics
from app.ml.score_ranking import ScoreRankingABC
from app.schemas.input_schema import RequestInput
from app.services.preprocessing_service import ejecutar_preprocesamiento


DEFAULT_INPUT_PATH = Path("data/input_request.json")
DEFAULT_OUTPUT_DIRECTORY = Path("data/score_comparison")

LABELS_ABC = ("A", "B", "C")

METHOD_ORDER = (
    "score_normalizado",
    "ss_ekmeans",
)

METHOD_DISPLAY = {
    "score_normalizado": "Score normalizado",
    "ss_ekmeans": "SS-EKMeans",
}


# ---------------------------------------------------------------------------
# Configuración general de las figuras
# ---------------------------------------------------------------------------


def configurar_matplotlib() -> None:
    """Aplica un formato sobrio y legible para una memoria académica."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ---------------------------------------------------------------------------
# Entrada y preprocesamiento
# ---------------------------------------------------------------------------


def cargar_payload(
    input_path: Path,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {input_path}"
        )

    raw_text = input_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not raw_text:
        raise ValueError(
            f"El archivo {input_path} está vacío."
        )

    try:
        return json.loads(raw_text)

    except json.JSONDecodeError:
        # También admite archivos que contengan texto
        # antes o después del objeto JSON.
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")

        if first_brace == -1 or last_brace == -1:
            raise ValueError(
                "No se encontró un objeto JSON válido."
            )

        json_text = raw_text[
            first_brace:last_brace + 1
        ]

        try:
            return json.loads(json_text)

        except json.JSONDecodeError as error:
            raise ValueError(
                "El archivo contiene un JSON inválido. "
                f"Línea {error.lineno}, "
                f"columna {error.colno}: "
                f"{error.msg}"
            ) from error


def cargar_datos(
    input_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    request_data = RequestInput.model_validate(
        cargar_payload(input_path)
    )

    preprocessing = ejecutar_preprocesamiento(
        request_data
    )

    if not preprocessing["hay_validos"]:
        raise RuntimeError(
            "El dataset no contiene publicaciones válidas."
        )

    X_modelo = (
        preprocessing["X_modelo"]
        .copy()
        .reset_index(drop=True)
    )

    datos_comerciales = (
        preprocessing["df_transformado"]
        .copy()
        .reset_index(drop=True)
    )

    columnas_requeridas = [
        "publication_id",
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    ]

    faltantes = [
        column
        for column in columnas_requeridas
        if column not in datos_comerciales.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas comerciales necesarias: "
            f"{faltantes}"
        )

    datos_comerciales = datos_comerciales[
        columnas_requeridas
    ].copy()

    for column in [
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    ]:
        datos_comerciales[column] = pd.to_numeric(
            datos_comerciales[column],
            errors="raise",
        )

    datos_comerciales["publication_id"] = (
        datos_comerciales["publication_id"]
        .astype(str)
    )

    datos_comerciales["valor_transado"] = (
        datos_comerciales["ventas_30d"]
        * datos_comerciales["precio_actual"]
    )

    if len(X_modelo) != len(datos_comerciales):
        raise RuntimeError(
            "La matriz del modelo y los datos comerciales "
            "tienen distinta cantidad de registros."
        )

    print(
        "Dataset preparado:",
        len(X_modelo),
        "publicaciones válidas.",
    )

    print(
        "Variables del modelo:",
        list(X_modelo.columns),
    )

    return X_modelo, datos_comerciales


# ---------------------------------------------------------------------------
# Ejecución de los dos métodos
# ---------------------------------------------------------------------------


def ejecutar_score_ranking(
    X_modelo: pd.DataFrame,
) -> tuple[
    ScoreRankingABC,
    pd.DataFrame,
    float,
]:
    config = get_production_config()

    model = ScoreRankingABC(
        proportions=config.proportions,
    )

    start = time.perf_counter()

    results = model.fit_predict(
        X_modelo
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return model, results, elapsed


def ejecutar_ss_ekmeans(
    X_modelo: pd.DataFrame,
) -> tuple[
    SSEKMeans,
    pd.DataFrame,
    float,
]:
    config = get_production_config()

    model = SSEKMeans(
        config=config
    )

    start = time.perf_counter()

    results = model.fit_predict(
        X_modelo
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return model, results, elapsed


# ---------------------------------------------------------------------------
# Utilidades numéricas y de formato
# ---------------------------------------------------------------------------


def porcentaje(
    numerador: float,
    denominador: float,
) -> float:
    if denominador == 0:
        return 0.0

    return float(
        numerador
        / denominador
        * 100
    )


def numero_seguro(
    value: Any,
) -> float:
    """
    Convierte NaN o infinito en cero para evitar problemas al exportar.
    """

    number = float(value)

    if not np.isfinite(number):
        return 0.0

    return number


def formato_decimal_es(
    value: Any,
    decimals: int = 2,
) -> str:
    """
    Formatea un número utilizando coma como separador decimal.
    """

    if value is None or pd.isna(value):
        return "—"

    number = float(value)

    if not np.isfinite(number):
        return "—"

    return (
        f"{number:.{decimals}f}"
        .replace(".", ",")
    )


def formato_entero_es(
    value: Any,
) -> str:
    """
    Formatea un entero utilizando punto como separador de miles.
    """

    if value is None or pd.isna(value):
        return "—"

    number = int(
        round(float(value))
    )

    return (
        f"{number:,}"
        .replace(",", ".")
    )


def formato_porcentaje_es(
    value: Any,
    decimals: int = 2,
) -> str:
    if value is None or pd.isna(value):
        return "—"

    return (
        f"{formato_decimal_es(value, decimals)} %"
    )


def formato_moneda_clp(
    value: Any,
) -> str:
    if value is None or pd.isna(value):
        return "—"

    return (
        f"${formato_entero_es(value)}"
    )


# ---------------------------------------------------------------------------
# Construcción de resultados
# ---------------------------------------------------------------------------


def construir_perfiles(
    datos_comerciales: pd.DataFrame,
    labels: pd.Series,
    metodo: str,
) -> pd.DataFrame:
    df = datos_comerciales.copy()

    df["categoria"] = (
        labels
        .astype(str)
        .reset_index(drop=True)
    )

    df["tiene_ventas"] = (
        df["ventas_30d"] > 0
    )

    total_ventas = float(
        df["ventas_30d"].sum()
    )

    total_visitas = float(
        df["visitas_30d"].sum()
    )

    total_valor = float(
        df["valor_transado"].sum()
    )

    rows: list[
        dict[str, Any]
    ] = []

    for categoria in LABELS_ABC:
        group = df[
            df["categoria"] == categoria
        ]

        ventas_total = float(
            group["ventas_30d"].sum()
        )

        visitas_total = float(
            group["visitas_30d"].sum()
        )

        valor_total = float(
            group["valor_transado"].sum()
        )

        publicaciones = int(
            len(group)
        )

        publicaciones_con_ventas = int(
            group["tiene_ventas"].sum()
        )

        rows.append(
            {
                "metodo": metodo,
                "categoria": categoria,
                "publicaciones": publicaciones,
                "publicaciones_pct": porcentaje(
                    publicaciones,
                    len(df),
                ),
                "ventas_total": ventas_total,
                "ventas_pct_total": porcentaje(
                    ventas_total,
                    total_ventas,
                ),
                "ventas_media": numero_seguro(
                    group["ventas_30d"].mean()
                ),
                "ventas_mediana": numero_seguro(
                    group["ventas_30d"].median()
                ),
                "visitas_total": visitas_total,
                "visitas_pct_total": porcentaje(
                    visitas_total,
                    total_visitas,
                ),
                "visitas_media": numero_seguro(
                    group["visitas_30d"].mean()
                ),
                "visitas_mediana": numero_seguro(
                    group["visitas_30d"].median()
                ),
                "precio_media": numero_seguro(
                    group["precio_actual"].mean()
                ),
                "precio_mediana": numero_seguro(
                    group["precio_actual"].median()
                ),
                "publicaciones_con_ventas": (
                    publicaciones_con_ventas
                ),
                "publicaciones_con_ventas_pct": porcentaje(
                    publicaciones_con_ventas,
                    publicaciones,
                ),
                "valor_transado_total": valor_total,
                "valor_transado_pct_total": porcentaje(
                    valor_total,
                    total_valor,
                ),
                "valor_transado_media": numero_seguro(
                    group["valor_transado"].mean()
                ),
                "valor_transado_mediana": numero_seguro(
                    group["valor_transado"].median()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def construir_resumen_metodo(
    metodo: str,
    X_modelo: pd.DataFrame,
    labels: pd.Series,
    profiles: pd.DataFrame,
    elapsed_seconds: float,
    inertia: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = evaluate_internal_metrics(
        X_modelo,
        labels,
    )

    profile_a = (
        profiles[
            profiles["categoria"] == "A"
        ]
        .iloc[0]
    )

    row: dict[str, Any] = {
        "metodo": metodo,
        "publicaciones": len(labels),
        "cantidad_A": int(
            (labels == "A").sum()
        ),
        "cantidad_B": int(
            (labels == "B").sum()
        ),
        "cantidad_C": int(
            (labels == "C").sum()
        ),
        "silhouette": metrics[
            "silhouette"
        ],
        "davies_bouldin": metrics[
            "davies_bouldin"
        ],
        "calinski_harabasz": metrics[
            "calinski_harabasz"
        ],
        "inercia": float(inertia),
        "tiempo_segundos": float(
            elapsed_seconds
        ),
        "ventas_en_A": float(
            profile_a["ventas_total"]
        ),
        "ventas_en_A_pct": float(
            profile_a["ventas_pct_total"]
        ),
        "visitas_en_A": float(
            profile_a["visitas_total"]
        ),
        "visitas_en_A_pct": float(
            profile_a["visitas_pct_total"]
        ),
        "valor_transado_en_A": float(
            profile_a["valor_transado_total"]
        ),
        "valor_transado_en_A_pct": float(
            profile_a[
                "valor_transado_pct_total"
            ]
        ),
        "publicaciones_con_ventas_en_A": int(
            profile_a[
                "publicaciones_con_ventas"
            ]
        ),
        "publicaciones_con_ventas_en_A_pct": float(
            profile_a[
                "publicaciones_con_ventas_pct"
            ]
        ),
    }

    if extra:
        row.update(extra)

    return row


def construir_comparacion_individual(
    datos_comerciales: pd.DataFrame,
    score_results: pd.DataFrame,
    ss_results: pd.DataFrame,
) -> pd.DataFrame:
    comparison = (
        datos_comerciales.copy()
    )

    comparison["score_inicial"] = (
        score_results["score_inicial"]
        .reset_index(drop=True)
    )

    comparison["categoria_score"] = (
        score_results["categoria"]
        .astype(str)
        .reset_index(drop=True)
    )

    comparison["categoria_ss_ekmeans"] = (
        ss_results["categoria"]
        .astype(str)
        .reset_index(drop=True)
    )

    comparison["coincide"] = (
        comparison["categoria_score"]
        == comparison[
            "categoria_ss_ekmeans"
        ]
    )

    return comparison


# ---------------------------------------------------------------------------
# Tablas formateadas para la memoria
# ---------------------------------------------------------------------------


def construir_tabla_concentracion(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye la tabla comercial ya formateada para la memoria.
    """

    tabla = profiles[
        [
            "metodo",
            "categoria",
            "publicaciones",
            "ventas_total",
            "ventas_pct_total",
            "visitas_total",
            "visitas_pct_total",
            "publicaciones_con_ventas",
            "publicaciones_con_ventas_pct",
            "valor_transado_total",
            "valor_transado_pct_total",
        ]
    ].copy()

    tabla["metodo"] = (
        tabla["metodo"]
        .map(METHOD_DISPLAY)
    )

    tabla = tabla.rename(
        columns={
            "metodo": "Método",
            "categoria": "Categoría",
            "publicaciones": "Publicaciones",
            "ventas_total": "Ventas totales",
            "ventas_pct_total": "Ventas (%)",
            "visitas_total": "Visitas totales",
            "visitas_pct_total": "Visitas (%)",
            "publicaciones_con_ventas": (
                "Publicaciones con ventas"
            ),
            "publicaciones_con_ventas_pct": (
                "Publicaciones con ventas (%)"
            ),
            "valor_transado_total": (
                "Valor transado total"
            ),
            "valor_transado_pct_total": (
                "Valor transado (%)"
            ),
        }
    )

    for column in [
        "Publicaciones",
        "Ventas totales",
        "Visitas totales",
        "Publicaciones con ventas",
    ]:
        tabla[column] = (
            tabla[column]
            .map(formato_entero_es)
        )

    for column in [
        "Ventas (%)",
        "Visitas (%)",
        "Publicaciones con ventas (%)",
        "Valor transado (%)",
    ]:
        tabla[column] = (
            tabla[column]
            .map(
                lambda value: formato_porcentaje_es(
                    value,
                    2,
                )
            )
        )

    tabla["Valor transado total"] = (
        tabla["Valor transado total"]
        .map(formato_moneda_clp)
    )

    return tabla


def construir_tabla_descriptivos(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye los estadísticos descriptivos listos para la memoria.
    """

    tabla = profiles[
        [
            "metodo",
            "categoria",
            "ventas_media",
            "ventas_mediana",
            "visitas_media",
            "visitas_mediana",
            "precio_media",
            "precio_mediana",
            "valor_transado_media",
            "valor_transado_mediana",
        ]
    ].copy()

    tabla["metodo"] = (
        tabla["metodo"]
        .map(METHOD_DISPLAY)
    )

    tabla = tabla.rename(
        columns={
            "metodo": "Método",
            "categoria": "Categoría",
            "ventas_media": "Media ventas",
            "ventas_mediana": "Mediana ventas",
            "visitas_media": "Media visitas",
            "visitas_mediana": "Mediana visitas",
            "precio_media": "Precio medio",
            "precio_mediana": "Precio mediano",
            "valor_transado_media": (
                "Valor transado medio"
            ),
            "valor_transado_mediana": (
                "Valor transado mediano"
            ),
        }
    )

    for column in [
        "Media ventas",
        "Mediana ventas",
        "Media visitas",
        "Mediana visitas",
    ]:
        tabla[column] = (
            tabla[column]
            .map(
                lambda value: formato_decimal_es(
                    value,
                    2,
                )
            )
        )

    for column in [
        "Precio medio",
        "Precio mediano",
        "Valor transado medio",
        "Valor transado mediano",
    ]:
        tabla[column] = (
            tabla[column]
            .map(formato_moneda_clp)
        )

    return tabla


def construir_tabla_calidad(
    summary: pd.DataFrame,
    method_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye la tabla de calidad con los redondeos de la memoria.
    """

    score = (
        summary.loc[
            summary["metodo"]
            == "score_normalizado"
        ]
        .iloc[0]
    )

    ss = (
        summary.loc[
            summary["metodo"]
            == "ss_ekmeans"
        ]
        .iloc[0]
    )

    comparison = (
        method_comparison.iloc[0]
    )

    rows = [
        {
            "Indicador": "Silhouette ↑",
            "Score normalizado": formato_decimal_es(
                score["silhouette"],
                4,
            ),
            "SS-EKMeans": formato_decimal_es(
                ss["silhouette"],
                4,
            ),
            "Interpretación": "Mayor es mejor",
        },
        {
            "Indicador": "Davies-Bouldin ↓",
            "Score normalizado": formato_decimal_es(
                score["davies_bouldin"],
                4,
            ),
            "SS-EKMeans": formato_decimal_es(
                ss["davies_bouldin"],
                4,
            ),
            "Interpretación": "Menor es mejor",
        },
        {
            "Indicador": "Calinski-Harabasz ↑",
            "Score normalizado": formato_decimal_es(
                score["calinski_harabasz"],
                4,
            ),
            "SS-EKMeans": formato_decimal_es(
                ss["calinski_harabasz"],
                4,
            ),
            "Interpretación": "Mayor es mejor",
        },
        {
            "Indicador": "Inercia ↓",
            "Score normalizado": formato_decimal_es(
                score["inercia"],
                2,
            ),
            "SS-EKMeans": formato_decimal_es(
                ss["inercia"],
                2,
            ),
            "Interpretación": "Menor es mejor",
        },
        {
            "Indicador": "Tiempo (s)",
            "Score normalizado": formato_decimal_es(
                score["tiempo_segundos"],
                3,
            ),
            "SS-EKMeans": formato_decimal_es(
                ss["tiempo_segundos"],
                3,
            ),
            "Interpretación": "El score solo ordena",
        },
        {
            "Indicador": "Coincidencia exacta",
            "Score normalizado": "—",
            "SS-EKMeans": formato_porcentaje_es(
                comparison[
                    "coincidencia_exacta_pct"
                ],
                2,
            ),
            "Interpretación": (
                "Comparación entre métodos"
            ),
        },
        {
            "Indicador": "Publicaciones diferentes",
            "Score normalizado": "—",
            "SS-EKMeans": formato_entero_es(
                comparison[
                    "publicaciones_diferentes"
                ]
            ),
            "Interpretación": (
                "Comparación entre métodos"
            ),
        },
        {
            "Indicador": "Adjusted Rand Index",
            "Score normalizado": "—",
            "SS-EKMeans": formato_decimal_es(
                comparison[
                    "adjusted_rand_index"
                ],
                4,
            ),
            "Interpretación": (
                "Concordancia entre particiones"
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def guardar_figura(
    fig: plt.Figure,
    output_directory: Path,
    basename: str,
) -> None:
    fig.savefig(
        output_directory
        / f"{basename}.png",
        bbox_inches="tight",
        dpi=300,
    )

    fig.savefig(
        output_directory
        / f"{basename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def generar_figura_distribucion_comercial(
    profiles: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Genera barras horizontales apiladas al 100 %.

    Los porcentajes pequeños se ubican fuera de las barras con una línea
    guía. Esto evita superposiciones y permite visualizar también el
    porcentaje pequeño de valor transado asignado a B por la línea base.
    """

    indicadores = [
        (
            "Ventas",
            "ventas_pct_total",
        ),
        (
            "Visitas",
            "visitas_pct_total",
        ),
        (
            "Valor transado",
            "valor_transado_pct_total",
        ),
    ]

    rows: list[
        dict[str, Any]
    ] = []

    for (
        indicador_display,
        indicador_column,
    ) in indicadores:
        for metodo in METHOD_ORDER:
            subset = (
                profiles[
                    profiles["metodo"]
                    == metodo
                ]
                .set_index("categoria")
                .reindex(LABELS_ABC)
            )

            row: dict[str, Any] = {
                "etiqueta": (
                    f"{indicador_display} — "
                    f"{METHOD_DISPLAY[metodo]}"
                )
            }

            for categoria in LABELS_ABC:
                row[categoria] = float(
                    subset.loc[
                        categoria,
                        indicador_column,
                    ]
                )

            rows.append(row)

    plot_data = pd.DataFrame(
        rows
    )

    y = np.arange(
        len(plot_data)
    )

    facecolors = {
        "A": "0.25",
        "B": "0.58",
        "C": "0.86",
    }

    hatches = {
        "A": "///",
        "B": "...",
        "C": "",
    }

    fig, ax = plt.subplots(
        figsize=(9.2, 5.4)
    )

    left = np.zeros(
        len(plot_data),
        dtype=float,
    )

    for categoria in LABELS_ABC:
        values = (
            plot_data[categoria]
            .to_numpy(dtype=float)
        )

        segment_left = (
            left.copy()
        )

        bars = ax.barh(
            y,
            values,
            left=segment_left,
            height=0.64,
            label=f"Categoría {categoria}",
            color=facecolors[categoria],
            edgecolor="black",
            linewidth=0.8,
            hatch=hatches[categoria],
        )

        for index, (
            bar,
            value,
        ) in enumerate(
            zip(bars, values)
        ):
            if value <= 0:
                continue

            center_x = (
                segment_left[index]
                + value / 2
            )

            center_y = (
                bar.get_y()
                + bar.get_height() / 2
            )

            label = formato_porcentaje_es(
                value,
                2,
            )

            if categoria == "A":
                ax.text(
                    center_x,
                    center_y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                    clip_on=True,
                )

            else:
                # Los segmentos B y C se rotulan fuera de la barra.
                # Se desplazan en sentidos opuestos para impedir
                # que sus textos se superpongan.
                label_y = (
                    center_y - 0.20
                    if categoria == "B"
                    else center_y + 0.20
                )

                ax.annotate(
                    label,
                    xy=(
                        center_x,
                        center_y,
                    ),
                    xycoords="data",
                    xytext=(
                        101.2,
                        label_y,
                    ),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=7.3,
                    annotation_clip=False,
                    clip_on=False,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "black",
                        "linewidth": 0.7,
                        "shrinkA": 2,
                        "shrinkB": 2,
                    },
                )

        left += values

    ax.set_yticks(y)

    ax.set_yticklabels(
        plot_data["etiqueta"]
    )

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        100,
    )

    ax.set_xticks(
        np.arange(
            0,
            101,
            10,
        )
    )

    ax.set_xlabel(
        "Participación dentro del total del indicador (%)"
    )

    ax.margins(
        y=0.10
    )

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.5,
        alpha=0.55,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.13,
        ),
        ncol=3,
        frameon=False,
    )

    # El título se agrega debajo de la figura en la memoria,
    # por lo que no se repite dentro del gráfico.
    #
    # Se reserva margen derecho para los porcentajes externos.
    fig.tight_layout(
        rect=(
            0,
            0,
            0.90,
            1,
        )
    )

    guardar_figura(
        fig,
        output_directory,
        "figura_distribucion_comercial",
    )


def generar_figura_matriz_transicion(
    transition: pd.DataFrame,
    exact_match_pct: float,
    ari: float,
    output_directory: Path,
) -> None:
    """
    Genera un mapa de calor de 3 x 3.

    El color representa el porcentaje por fila. Cada celda muestra la
    cantidad de publicaciones y el porcentaje respecto de su categoría
    de origen en el método de score normalizado.
    """

    matrix = (
        transition
        .reindex(
            index=LABELS_ABC,
            columns=LABELS_ABC,
            fill_value=0,
        )
        .to_numpy(dtype=float)
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    row_percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_totals != 0,
    ) * 100

    fig, ax = plt.subplots(
        figsize=(6.8, 5.8)
    )

    image = ax.imshow(
        row_percentages,
        cmap="Greys",
        vmin=0,
        vmax=100,
        aspect="equal",
    )

    ax.set_xticks(
        np.arange(
            len(LABELS_ABC)
        )
    )

    ax.set_yticks(
        np.arange(
            len(LABELS_ABC)
        )
    )

    ax.set_xticklabels(
        LABELS_ABC
    )

    ax.set_yticklabels(
        LABELS_ABC
    )

    ax.set_xlabel(
        "Categoría asignada por SS-EKMeans"
    )

    ax.set_ylabel(
        "Categoría asignada por score normalizado"
    )

    coincidencia_display = (
        formato_porcentaje_es(
            exact_match_pct,
            2,
        )
    )

    ari_display = (
        formato_decimal_es(
            ari,
            4,
        )
    )

    ax.set_title(
        "Matriz de transición entre métodos\n"
        f"Coincidencia exacta: "
        f"{coincidencia_display} · "
        f"ARI: {ari_display}"
    )

    # Rejilla entre las celdas.
    ax.set_xticks(
        np.arange(
            -0.5,
            len(LABELS_ABC),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(LABELS_ABC),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="black",
        linestyle="-",
        linewidth=0.8,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for row in range(
        len(LABELS_ABC)
    ):
        for column in range(
            len(LABELS_ABC)
        ):
            count = int(
                matrix[
                    row,
                    column,
                ]
            )

            pct = float(
                row_percentages[
                    row,
                    column,
                ]
            )

            color = (
                "white"
                if pct >= 55
                else "black"
            )

            weight = (
                "bold"
                if row == column
                else "normal"
            )

            ax.text(
                column,
                row,
                (
                    f"{count}\n"
                    f"("
                    f"{formato_porcentaje_es(pct, 1)}"
                    f")"
                ),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight=weight,
            )

            if row == column:
                ax.add_patch(
                    Rectangle(
                        (
                            column - 0.5,
                            row - 0.5,
                        ),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Porcentaje de la categoría de origen (%)"
    )

    fig.tight_layout()

    guardar_figura(
        fig,
        output_directory,
        "figura_matriz_transicion",
    )


# ---------------------------------------------------------------------------
# Escritura de archivos
# ---------------------------------------------------------------------------


def guardar_csv(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> None:
    dataframe.to_csv(
        path,
        index=index,
        encoding="utf-8-sig",
        float_format="%.6f",
    )


# ---------------------------------------------------------------------------
# Argumentos del script
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara score normalizado con SS-EKMeans "
            "y genera tablas y figuras para la memoria."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Ruta del JSON de entrada. "
            f"Predeterminado: {DEFAULT_INPUT_PATH}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directorio de salida. "
            f"Predeterminado: {DEFAULT_OUTPUT_DIRECTORY}"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    input_path: Path = (
        args.input
    )

    output_directory: Path = (
        args.output
    )

    configurar_matplotlib()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_modelo, datos_comerciales = (
        cargar_datos(
            input_path
        )
    )

    print(
        "\n1. Ejecutando línea base "
        "por score normalizado..."
    )

    (
        score_model,
        score_results,
        score_time,
    ) = ejecutar_score_ranking(
        X_modelo
    )

    print(
        "2. Ejecutando SS-EKMeans productivo..."
    )

    (
        ss_model,
        ss_results,
        ss_time,
    ) = ejecutar_ss_ekmeans(
        X_modelo
    )

    # Se verifica que la línea base y SS-EKMeans
    # utilicen exactamente el mismo score inicial.
    if not np.allclose(
        score_results["score_inicial"],
        ss_results["score_inicial"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "El score de la línea base no coincide con "
            "el score utilizado por SS-EKMeans."
        )

    score_labels = (
        score_results["categoria"]
        .astype(str)
        .reset_index(drop=True)
    )

    ss_labels = (
        ss_results["categoria"]
        .astype(str)
        .reset_index(drop=True)
    )

    score_profiles = construir_perfiles(
        datos_comerciales,
        score_labels,
        "score_normalizado",
    )

    ss_profiles = construir_perfiles(
        datos_comerciales,
        ss_labels,
        "ss_ekmeans",
    )

    profiles = pd.concat(
        [
            score_profiles,
            ss_profiles,
        ],
        ignore_index=True,
    )

    summary = pd.DataFrame(
        [
            construir_resumen_metodo(
                metodo="score_normalizado",
                X_modelo=X_modelo,
                labels=score_labels,
                profiles=score_profiles,
                elapsed_seconds=score_time,
                inertia=score_model.inertia_,
                extra={
                    "iteraciones": 0,
                    "convergio": True,
                    "motivo_termino": (
                        "ordenamiento_determinista"
                    ),
                },
            ),
            construir_resumen_metodo(
                metodo="ss_ekmeans",
                X_modelo=X_modelo,
                labels=ss_labels,
                profiles=ss_profiles,
                elapsed_seconds=ss_time,
                inertia=ss_model.inertia_,
                extra={
                    "iteraciones": (
                        ss_model.n_iter_
                    ),
                    "convergio": (
                        ss_model.converged_
                    ),
                    "motivo_termino": (
                        ss_model.stop_reason_
                    ),
                    "corridas_convergentes": (
                        ss_model.converged_runs_
                    ),
                    "corridas_descartadas": (
                        ss_model.discarded_runs_
                    ),
                },
            ),
        ]
    )

    exact_match_pct = float(
        (
            score_labels
            == ss_labels
        ).mean()
        * 100
    )

    changed_count = int(
        (
            score_labels
            != ss_labels
        ).sum()
    )

    ari = float(
        adjusted_rand_score(
            score_labels,
            ss_labels,
        )
    )

    method_comparison = pd.DataFrame(
        [
            {
                "coincidencia_exacta_pct": (
                    exact_match_pct
                ),
                "publicaciones_diferentes": (
                    changed_count
                ),
                "adjusted_rand_index": ari,
            }
        ]
    )

    transition = (
        pd.crosstab(
            score_labels,
            ss_labels,
            rownames=[
                "categoria_score"
            ],
            colnames=[
                "categoria_ss_ekmeans"
            ],
            dropna=False,
        )
        .reindex(
            index=LABELS_ABC,
            columns=LABELS_ABC,
            fill_value=0,
        )
    )

    individual = (
        construir_comparacion_individual(
            datos_comerciales,
            score_results,
            ss_results,
        )
    )

    changed = individual[
        ~individual["coincide"]
    ].copy()

    tabla_concentracion = (
        construir_tabla_concentracion(
            profiles
        )
    )

    tabla_descriptivos = (
        construir_tabla_descriptivos(
            profiles
        )
    )

    tabla_calidad = (
        construir_tabla_calidad(
            summary,
            method_comparison,
        )
    )

    output_files: dict[
        str,
        tuple[pd.DataFrame, bool],
    ] = {
        "comparison_summary.csv": (
            summary,
            False,
        ),
        "class_profiles.csv": (
            profiles,
            False,
        ),
        "labels_comparison.csv": (
            individual,
            False,
        ),
        "transition_matrix.csv": (
            transition,
            True,
        ),
        "changed_publications.csv": (
            changed,
            False,
        ),
        "method_comparison.csv": (
            method_comparison,
            False,
        ),
        "tabla_concentracion_comercial.csv": (
            tabla_concentracion,
            False,
        ),
        "tabla_estadisticos_descriptivos.csv": (
            tabla_descriptivos,
            False,
        ),
        "tabla_calidad_concordancia.csv": (
            tabla_calidad,
            False,
        ),
    }

    for (
        filename,
        (
            dataframe,
            include_index,
        ),
    ) in output_files.items():
        guardar_csv(
            dataframe,
            output_directory
            / filename,
            index=include_index,
        )

    print(
        "3. Generando figuras..."
    )

    generar_figura_distribucion_comercial(
        profiles,
        output_directory,
    )

    generar_figura_matriz_transicion(
        transition,
        exact_match_pct,
        ari,
        output_directory,
    )

    print(
        "\n4. Resumen de métodos"
    )

    print(
        summary[
            [
                "metodo",
                "cantidad_A",
                "cantidad_B",
                "cantidad_C",
                "ventas_en_A_pct",
                "visitas_en_A_pct",
                "valor_transado_en_A_pct",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "inercia",
                "tiempo_segundos",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n5. Comparación de asignaciones"
    )

    print(
        "  Coincidencia exacta: "
        f"{exact_match_pct:.2f}%"
    )

    print(
        "  Publicaciones diferentes: "
        f"{changed_count}"
    )

    print(
        f"  ARI: {ari:.4f}"
    )

    print(
        "\n6. Matriz de transición"
    )

    print(
        transition.to_string()
    )

    print(
        "\nArchivos generados:"
    )

    for filename in output_files:
        print(
            f"  "
            f"{output_directory / filename}"
        )

    for filename in [
        "figura_distribucion_comercial.png",
        "figura_distribucion_comercial.pdf",
        "figura_matriz_transicion.png",
        "figura_matriz_transicion.pdf",
    ]:
        print(
            f"  "
            f"{output_directory / filename}"
        )


if __name__ == "__main__":
    main()