from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_PATH = Path(
    "data/input_request.json"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/shap_performance_results"
)

DEFAULT_EXPLAIN_URL = (
    "http://127.0.0.1:8000"
    "/api/explainability/autosklearn/explain"
)

DEFAULT_HEALTH_URL = (
    "http://127.0.0.1:8000"
    "/api/explainability/health"
)

DEFAULT_SIZES = [
    1,
    10,
    50,
    100,
    832,
]


def cargar_json(
    input_path: Path,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            "No se encontró el archivo de entrada: "
            f"{input_path}"
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
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")

        if (
            first_brace == -1
            or last_brace == -1
        ):
            raise ValueError(
                "No fue posible encontrar un objeto "
                f"JSON dentro de {input_path}."
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


def cargar_productos(
    input_path: Path,
) -> list[dict[str, Any]]:
    payload = cargar_json(input_path)

    productos = payload.get("productos")

    if not isinstance(productos, list):
        raise ValueError(
            "El archivo de entrada debe contener "
            "una lista llamada 'productos'."
        )

    if not productos:
        raise ValueError(
            "La lista de productos está vacía."
        )

    return productos


def verificar_servicios(
    session: requests.Session,
    health_url: str,
) -> dict[str, Any]:
    print(
        "Verificando la API y el worker "
        "de explicabilidad..."
    )

    try:
        response = session.get(
            health_url,
            timeout=30,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "No fue posible consultar el estado "
            f"del worker en {health_url}: {error}"
        ) from error

    if response.status_code >= 400:
        raise RuntimeError(
            "La verificación del worker respondió "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    health = response.json()

    if not health.get(
        "artifacts_compatible",
        False,
    ):
        raise RuntimeError(
            "Los artefactos AutoSklearn no son "
            "compatibles con las variables actuales. "
            "Debes volver a ejecutar el entrenamiento."
        )

    if not health.get("model_exists"):
        raise RuntimeError(
            "No existe el modelo AutoSklearn."
        )

    if not health.get("background_exists"):
        raise RuntimeError(
            "No existe el background SHAP."
        )

    if not health.get("meta_exists"):
        raise RuntimeError(
            "No existe el archivo de metadatos."
        )

    print("Worker disponible.")
    print(
        "Variables:",
        health.get("feature_columns"),
    )
    print(
        "Configuración SHAP:",
        health.get("shap_config"),
    )

    return health


def preparar_muestras(
    productos: list[dict[str, Any]],
    sizes: list[int],
    random_state: int,
) -> dict[int, list[dict[str, Any]]]:
    total_productos = len(productos)

    invalid_sizes = [
        size
        for size in sizes
        if size < 1 or size > total_productos
    ]

    if invalid_sizes:
        raise ValueError(
            "Los siguientes tamaños no son válidos "
            f"para un dataset de {total_productos} "
            f"publicaciones: {invalid_sizes}"
        )

    indices = list(
        range(total_productos)
    )

    random_generator = random.Random(
        random_state
    )

    random_generator.shuffle(indices)

    muestras: dict[
        int,
        list[dict[str, Any]],
    ] = {}

    for size in sorted(set(sizes)):
        selected_indices = indices[:size]

        muestras[size] = [
            productos[index]
            for index in selected_indices
        ]

    return muestras


def ejecutar_peticion(
    session: requests.Session,
    explain_url: str,
    productos: list[dict[str, Any]],
    top_n: int,
) -> tuple[dict[str, Any], float, int]:
    payload = {
        "productos": productos,
    }

    start_time = time.perf_counter()

    try:
        response = session.post(
            explain_url,
            params={
                "top_n": top_n,
            },
            json=payload,
            timeout=None,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "La petición al endpoint de "
            f"explicabilidad falló: {error}"
        ) from error

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if response.status_code >= 400:
        raise RuntimeError(
            "El endpoint respondió "
            f"HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    return (
        response.json(),
        elapsed_seconds,
        response.status_code,
    )


def construir_registro(
    size: int,
    repetition: int,
    response_data: dict[str, Any],
    elapsed_seconds: float,
    status_code: int,
) -> dict[str, Any]:
    shap_config = response_data.get(
        "shap_config",
        {},
    )

    explained_products = int(
        shap_config.get(
            "productos_explicados",
            size,
        )
    )

    shap_seconds = float(
        shap_config.get(
            "shap_seconds",
            0.0,
        )
    )

    seconds_per_product = (
        elapsed_seconds / explained_products
        if explained_products > 0
        else None
    )

    products_per_second = (
        explained_products / elapsed_seconds
        if elapsed_seconds > 0
        else None
    )

    products_per_minute = (
        products_per_second * 60
        if products_per_second is not None
        else None
    )

    api_overhead_seconds = max(
        0.0,
        elapsed_seconds - shap_seconds,
    )

    additivity = response_data.get(
        "validacion_aditividad",
        {},
    )

    return {
        "tamano_solicitado": size,
        "repeticion": repetition,
        "http_status": status_code,
        "productos_recibidos": int(
            shap_config.get(
                "productos_recibidos",
                size,
            )
        ),
        "productos_explicados": (
            explained_products
        ),
        "truncado": bool(
            shap_config.get(
                "truncated",
                False,
            )
        ),
        "background_rows": int(
            shap_config.get(
                "background_rows",
                0,
            )
        ),
        "nsamples": int(
            shap_config.get(
                "nsamples",
                0,
            )
        ),
        "latencia_total_segundos": (
            elapsed_seconds
        ),
        "tiempo_shap_segundos": (
            shap_seconds
        ),
        "sobrecarga_api_segundos": (
            api_overhead_seconds
        ),
        "segundos_por_producto": (
            seconds_per_product
        ),
        "productos_por_segundo": (
            products_per_second
        ),
        "productos_por_minuto": (
            products_per_minute
        ),
        "aditividad_cumplida": bool(
            additivity.get(
                "cumple_tolerancia",
                False,
            )
        ),
        "error_aditividad_maximo": (
            additivity.get(
                "max_absolute_error"
            )
        ),
    }


def guardar_metricas(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    pd.DataFrame(records).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def generar_resumen(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    metrics = pd.DataFrame(records)

    summary = (
        metrics
        .groupby(
            "tamano_solicitado",
            as_index=False,
        )
        .agg(
            repeticiones=(
                "repeticion",
                "count",
            ),
            latencia_total_media_s=(
                "latencia_total_segundos",
                "mean",
            ),
            latencia_total_mediana_s=(
                "latencia_total_segundos",
                "median",
            ),
            latencia_total_desviacion_s=(
                "latencia_total_segundos",
                "std",
            ),
            latencia_total_minima_s=(
                "latencia_total_segundos",
                "min",
            ),
            latencia_total_maxima_s=(
                "latencia_total_segundos",
                "max",
            ),
            tiempo_shap_mediano_s=(
                "tiempo_shap_segundos",
                "median",
            ),
            segundos_por_producto_mediana=(
                "segundos_por_producto",
                "median",
            ),
            productos_por_minuto_mediana=(
                "productos_por_minuto",
                "median",
            ),
            sobrecarga_api_mediana_s=(
                "sobrecarga_api_segundos",
                "median",
            ),
            error_aditividad_maximo=(
                "error_aditividad_maximo",
                "max",
            ),
        )
    )

    summary[
        "latencia_total_desviacion_s"
    ] = summary[
        "latencia_total_desviacion_s"
    ].fillna(0.0)

    return summary

def generar_figura_rendimiento(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Genera una figura con los principales indicadores
    de latencia y rendimiento del proceso Kernel SHAP.
    """
    data = (
        summary
        .sort_values(
            "tamano_solicitado"
        )
        .reset_index(drop=True)
    )

    sizes = (
        data["tamano_solicitado"]
        .astype(int)
        .to_numpy()
    )

    positions = np.arange(
        len(sizes)
    )

    latency_median = data[
        "latencia_total_mediana_s"
    ].to_numpy(dtype=float)

    latency_minimum = data[
        "latencia_total_minima_s"
    ].to_numpy(dtype=float)

    latency_maximum = data[
        "latencia_total_maxima_s"
    ].to_numpy(dtype=float)

    throughput = data[
        "productos_por_minuto_mediana"
    ].to_numpy(dtype=float)

    seconds_per_product = data[
        "segundos_por_producto_mediana"
    ].to_numpy(dtype=float)

    shap_seconds = data[
        "tiempo_shap_mediano_s"
    ].to_numpy(dtype=float)

    overhead_seconds = data[
        "sobrecarga_api_mediana_s"
    ].to_numpy(dtype=float)

    measured_total = (
        shap_seconds
        + overhead_seconds
    )

    shap_percentage = np.divide(
        shap_seconds * 100,
        measured_total,
        out=np.zeros_like(
            shap_seconds
        ),
        where=measured_total > 0,
    )

    overhead_percentage = np.divide(
        overhead_seconds * 100,
        measured_total,
        out=np.zeros_like(
            overhead_seconds
        ),
        where=measured_total > 0,
    )

    lower_error = np.maximum(
        0,
        latency_median
        - latency_minimum,
    )

    upper_error = np.maximum(
        0,
        latency_maximum
        - latency_median,
    )

    latency_error = np.vstack(
        [
            lower_error,
            upper_error,
        ]
    )

    dark_blue = "#1F4E79"
    medium_blue = "#5B84B1"
    light_blue = "#AABBCD"
    dark_gray = "#26313D"
    grid_color = "#D8DEE5"

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 10),
        dpi=180,
    )

    figure.suptitle(
        (
            "Latencia y rendimiento del proceso "
            "de explicabilidad con Kernel SHAP"
        ),
        fontsize=22,
        fontweight="bold",
        color=dark_gray,
    )

    # -------------------------------------------------
    # 1. Latencia total
    # -------------------------------------------------
    latency_axis = axes[0, 0]

    latency_axis.errorbar(
        positions,
        latency_median,
        yerr=latency_error,
        fmt="o-",
        color=dark_blue,
        linewidth=2.5,
        markersize=8,
        capsize=6,
        capthick=1.5,
    )

    latency_axis.set_title(
        "Latencia total por tamaño del lote",
        fontweight="bold",
    )

    latency_axis.set_xlabel(
        "Cantidad de publicaciones"
    )

    latency_axis.set_ylabel(
        "Latencia total (segundos, escala logarítmica)"
    )

    latency_axis.set_xticks(
        positions,
        labels=sizes,
    )

    latency_axis.set_yscale("log")

    latency_axis.grid(
        axis="y",
        color=grid_color,
        linestyle="--",
        alpha=0.8,
    )

    for position, value in zip(
        positions,
        latency_median,
    ):
        if value >= 60:
            label = f"{value / 60:.1f} min"
        else:
            label = f"{value:.2f} s"

        latency_axis.annotate(
            label,
            xy=(position, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            color=dark_gray,
        )

    # -------------------------------------------------
    # 2. Rendimiento
    # -------------------------------------------------
    throughput_axis = axes[0, 1]

    throughput_bars = (
        throughput_axis.bar(
            positions,
            throughput,
            color=medium_blue,
            edgecolor=dark_gray,
            linewidth=0.8,
        )
    )

    throughput_axis.set_title(
        "Rendimiento del endpoint",
        fontweight="bold",
    )

    throughput_axis.set_xlabel(
        "Cantidad de publicaciones"
    )

    throughput_axis.set_ylabel(
        "Publicaciones explicadas por minuto"
    )

    throughput_axis.set_xticks(
        positions,
        labels=sizes,
    )

    throughput_axis.grid(
        axis="y",
        color=grid_color,
        linestyle="--",
        alpha=0.8,
    )

    throughput_axis.bar_label(
        throughput_bars,
        labels=[
            f"{value:.2f}"
            for value in throughput
        ],
        padding=4,
        fontweight="bold",
    )

    # -------------------------------------------------
    # 3. Tiempo amortizado
    # -------------------------------------------------
    amortized_axis = axes[1, 0]

    amortized_bars = (
        amortized_axis.bar(
            positions,
            seconds_per_product,
            color=dark_blue,
            edgecolor=dark_gray,
            linewidth=0.8,
        )
    )

    amortized_axis.set_title(
        "Tiempo promedio amortizado",
        fontweight="bold",
    )

    amortized_axis.set_xlabel(
        "Cantidad de publicaciones"
    )

    amortized_axis.set_ylabel(
        "Segundos por publicación"
    )

    amortized_axis.set_xticks(
        positions,
        labels=sizes,
    )

    amortized_axis.grid(
        axis="y",
        color=grid_color,
        linestyle="--",
        alpha=0.8,
    )

    amortized_axis.bar_label(
        amortized_bars,
        labels=[
            f"{value:.3f}"
            for value in seconds_per_product
        ],
        padding=4,
        fontweight="bold",
    )

    # -------------------------------------------------
    # 4. Distribución del tiempo
    # -------------------------------------------------
    composition_axis = axes[1, 1]

    composition_axis.bar(
        positions,
        shap_percentage,
        color=dark_blue,
        edgecolor=dark_gray,
        linewidth=0.8,
        label="Kernel SHAP",
    )

    composition_axis.bar(
        positions,
        overhead_percentage,
        bottom=shap_percentage,
        color=light_blue,
        edgecolor=dark_gray,
        linewidth=0.8,
        label="API, serialización y comunicación",
    )

    composition_axis.set_title(
        "Distribución del tiempo total",
        fontweight="bold",
    )

    composition_axis.set_xlabel(
        "Cantidad de publicaciones"
    )

    composition_axis.set_ylabel(
        "Participación en el tiempo total (%)"
    )

    composition_axis.set_xticks(
        positions,
        labels=sizes,
    )

    composition_axis.set_ylim(
        0,
        105,
    )

    composition_axis.grid(
        axis="y",
        color=grid_color,
        linestyle="--",
        alpha=0.8,
    )

    composition_axis.legend(
        loc="lower right"
    )

    for position in positions:
        shap_value = (
            shap_percentage[position]
        )

        overhead_value = (
            overhead_percentage[position]
        )

        if shap_value >= 4:
            composition_axis.text(
                position,
                shap_value / 2,
                f"{shap_value:.1f} %",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        if overhead_value >= 4:
            composition_axis.text(
                position,
                (
                    shap_value
                    + overhead_value / 2
                ),
                f"{overhead_value:.1f} %",
                ha="center",
                va="center",
                color=dark_gray,
                fontweight="bold",
            )

    figure.text(
        0.5,
        0.015,
        (
            "Los valores corresponden a la mediana de las "
            "repeticiones. Las barras de error representan "
            "los tiempos mínimo y máximo observados."
        ),
        ha="center",
        fontsize=12,
        color="#66727E",
    )

    figure.tight_layout(
        rect=[
            0,
            0.045,
            1,
            0.94,
        ]
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Tipo no serializable: {type(value)}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa latencia y rendimiento del "
            "endpoint real de Kernel SHAP."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--explain-url",
        default=DEFAULT_EXPLAIN_URL,
    )

    parser.add_argument(
        "--health-url",
        default=DEFAULT_HEALTH_URL,
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help=(
            "Repeticiones para tamaños menores "
            "al dataset completo."
        ),
    )

    parser.add_argument(
        "--full-repetitions",
        type=int,
        default=1,
        help=(
            "Repeticiones cuando se utilizan "
            "todas las publicaciones."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        choices=[1, 2, 3],
    )

    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.repetitions < 1:
        raise ValueError(
            "repetitions debe ser mayor o "
            "igual a 1."
        )

    if arguments.full_repetitions < 1:
        raise ValueError(
            "full-repetitions debe ser mayor "
            "o igual a 1."
        )

    products = cargar_productos(
        arguments.input
    )

    total_products = len(products)

    print(
        "Dataset cargado:",
        total_products,
        "publicaciones.",
    )

    samples = preparar_muestras(
        productos=products,
        sizes=arguments.sizes,
        random_state=arguments.random_state,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        arguments.output_directory
        / "shap_performance_metrics.csv"
    )

    summary_path = (
        arguments.output_directory
        / "shap_performance_summary.csv"
    )

    report_path = (
        arguments.output_directory
        / "shap_performance_summary.json"
    )

    session = requests.Session()

    health = verificar_servicios(
        session=session,
        health_url=arguments.health_url,
    )

    if not arguments.skip_warmup:
        print()
        print(
            "Ejecutando calentamiento con "
            "una publicación..."
        )

        ejecutar_peticion(
            session=session,
            explain_url=arguments.explain_url,
            productos=samples[
                min(samples)
            ][:1],
            top_n=arguments.top_n,
        )

        print("Calentamiento completado.")

    records: list[dict[str, Any]] = []

    planned_runs = sum(
        (
            arguments.full_repetitions
            if size == total_products
            else arguments.repetitions
        )
        for size in sorted(samples)
    )

    current_run = 0

    for size in sorted(samples):
        repetitions = (
            arguments.full_repetitions
            if size == total_products
            else arguments.repetitions
        )

        for repetition in range(
            1,
            repetitions + 1,
        ):
            current_run += 1

            print()
            print(
                f"[{current_run:02d}/"
                f"{planned_runs:02d}] "
                f"Tamaño={size} | "
                f"Repetición={repetition}"
            )

            response_data, elapsed, status = (
                ejecutar_peticion(
                    session=session,
                    explain_url=(
                        arguments.explain_url
                    ),
                    productos=samples[size],
                    top_n=arguments.top_n,
                )
            )

            record = construir_registro(
                size=size,
                repetition=repetition,
                response_data=response_data,
                elapsed_seconds=elapsed,
                status_code=status,
            )

            records.append(record)

            guardar_metricas(
                records=records,
                output_path=metrics_path,
            )

            print(
                "  Latencia total:",
                f"{record['latencia_total_segundos']:.3f} s",
            )
            print(
                "  Tiempo Kernel SHAP:",
                f"{record['tiempo_shap_segundos']:.3f} s",
            )
            print(
                "  Promedio amortizado:",
                f"{record['segundos_por_producto']:.4f} "
                "s/publicación",
            )
            print(
                "  Rendimiento:",
                f"{record['productos_por_minuto']:.2f} "
                "publicaciones/minuto",
            )
            print(
                "  Aditividad:",
                (
                    "CUMPLE"
                    if record[
                        "aditividad_cumplida"
                    ]
                    else "NO CUMPLE"
                ),
            )

            if (
                current_run < planned_runs
                and arguments.pause_seconds > 0
            ):
                time.sleep(
                    arguments.pause_seconds
                )

    summary = generar_resumen(records)

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    figure_path = (
        arguments.output_directory
        / "figura_rendimiento_shap.png"
    )

    generar_figura_rendimiento(
        summary=summary,
        output_path=figure_path,
    )

    report = {
        "dataset": {
            "input_path": str(
                arguments.input
            ),
            "total_productos": (
                total_products
            ),
            "sizes": sorted(samples),
            "repetitions": (
                arguments.repetitions
            ),
            "full_repetitions": (
                arguments.full_repetitions
            ),
            "random_state": (
                arguments.random_state
            ),
            "figura": str(figure_path),
        },
        "worker": health,
        "resultados": summary.to_dict(
            orient="records"
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("EVALUACIÓN FINALIZADA")
    print("=" * 70)
    print()
    print(summary.to_string(index=False))
    print()
    print(f"  {figure_path}")
    print("Archivos generados:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()