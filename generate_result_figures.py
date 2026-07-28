from __future__ import annotations

"""
Genera las figuras finales del capítulo de resultados del ClasificadorABC.

El script puede:

1. Ejecutar la clasificación real de la aplicación desde un request JSON.
2. Reutilizar una respuesta JSON ya generada por POST /api/clasificar.
3. Incorporar los resultados de Optuna y de estabilidad entre random_state.
4. Incorporar, cuando exista, un CSV con resultados de bootstrap.

Ejemplo recomendado:

    python generate_result_figures.py \
        --input-request data/input_request.json \
        --optuna-result data/optuna_result.json \
        --stability-metrics data/stability_results/random_state_metrics.csv \
        --output-dir artifacts/result_figures

También puede ejecutarse sin volver a clasificar:

    python generate_result_figures.py \
        --classification-response data/classification_response.json \
        --optuna-result data/optuna_result.json

Dependencias adicionales:

    pip install matplotlib
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


CATEGORIAS = ("A", "B", "C")

COLUMNAS_ESCALABLES = (
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
)

COLUMNAS_MODELO = (*COLUMNAS_ESCALABLES, "en_promocion")

NOMBRES_VARIABLES = {
    "ventas_30d": "Ventas (30 días)",
    "visitas_30d": "Visitas (30 días)",
    "precio_actual": "Precio actual",
    "stock_actual": "Stock actual",
    "en_promocion": "En promoción",
}

# Paleta sobria y consistente para todas las figuras.
COLORES_CATEGORIA = {
    "A": "#163A5F",
    "B": "#4F78A8",
    "C": "#A7B4C2",
}

COLOR_TEXTO = "#1F2933"
COLOR_EJES = "#66727D"
COLOR_REJILLA = "#D9DEE3"
COLOR_DESTACADO = "#9D3C3C"


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta la clasificación ABC o reutiliza su respuesta y genera "
            "las figuras finales del capítulo de resultados."
        )
    )

    fuente = parser.add_mutually_exclusive_group(required=True)
    fuente.add_argument(
        "--input-request",
        type=Path,
        help=(
            "Request JSON con la clave 'productos'. La clasificación se "
            "ejecutará mediante los servicios reales de la aplicación."
        ),
    )
    fuente.add_argument(
        "--classification-response",
        type=Path,
        help="Respuesta JSON previamente generada por POST /api/clasificar.",
    )

    parser.add_argument(
        "--optuna-result",
        type=Path,
        help="JSON devuelto por POST /optimization/optuna.",
    )
    parser.add_argument(
        "--stability-metrics",
        type=Path,
        help=(
            "CSV generado por evaluate_random_state_stability.py. Debe "
            "contener adjusted_rand_index y silhouette."
        ),
    )
    parser.add_argument(
        "--bootstrap-metrics",
        type=Path,
        help=(
            "CSV opcional de bootstrap. Debe contener adjusted_rand_index "
            "y puede incluir jaccard_A, jaccard_B y jaccard_C."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/result_figures"),
        help="Directorio de salida. Valor predeterminado: artifacts/result_figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolución de las imágenes PNG. Valor predeterminado: 300.",
    )
    parser.add_argument(
        "--include-appendix",
        action="store_true",
        help="Genera además el gráfico precio-stock propuesto para anexos.",
    )

    return parser.parse_args()


def aplicar_estilo() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": COLOR_EJES,
            "axes.labelcolor": COLOR_TEXTO,
            "xtick.color": COLOR_TEXTO,
            "ytick.color": COLOR_TEXTO,
            "text.color": COLOR_TEXTO,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def leer_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"El archivo no contiene JSON válido: {path}. "
            f"Detalle: {error}"
        ) from error


def guardar_json(path: Path, contenido: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            convertir_json_serializable(contenido),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def convertir_json_serializable(valor: Any) -> Any:
    if isinstance(valor, Mapping):
        return {
            str(clave): convertir_json_serializable(contenido)
            for clave, contenido in valor.items()
        }

    if isinstance(valor, (list, tuple)):
        return [convertir_json_serializable(item) for item in valor]

    if isinstance(valor, Path):
        return str(valor)

    if isinstance(valor, np.generic):
        return valor.item()

    if pd.isna(valor):
        return None

    return valor


def ejecutar_clasificacion_desde_request(input_path: Path) -> dict[str, Any]:
    """
    Ejecuta los servicios reales de ClasificadorABC.

    El script debe ubicarse en la raíz del repositorio o ejecutarse con la
    raíz del repositorio disponible en PYTHONPATH.
    """
    try:
        from app.schemas.input_schema import RequestInput
        from app.services.clasificacion_service import ejecutar_clasificacion
    except ImportError as error:
        raise RuntimeError(
            "No fue posible importar la aplicación. Coloca este archivo en "
            "la raíz de ClasificadorABC y ejecútalo desde esa carpeta."
        ) from error

    request_payload = leer_json(input_path)
    request_data = RequestInput.model_validate(request_payload)
    response = ejecutar_clasificacion(request_data)

    if not isinstance(response, dict):
        raise TypeError(
            "ejecutar_clasificacion() no devolvió un diccionario."
        )

    return response


def obtener_respuesta_clasificacion(
    input_request: Optional[Path],
    classification_response: Optional[Path],
) -> dict[str, Any]:
    if input_request is not None:
        return ejecutar_clasificacion_desde_request(input_request)

    if classification_response is None:
        raise ValueError("Debe indicarse una fuente de clasificación.")

    contenido = leer_json(classification_response)

    if not isinstance(contenido, dict):
        raise TypeError(
            "La respuesta de clasificación debe ser un objeto JSON."
        )

    return contenido


def construir_dataframe(
    response: Mapping[str, Any],
) -> pd.DataFrame:
    resultados = response.get("resultados")

    if not isinstance(resultados, list) or not resultados:
        raise ValueError(
            "La respuesta no contiene una lista no vacía en 'resultados'."
        )

    df = pd.DataFrame(resultados)
    columnas_requeridas = {
        "publication_id",
        *COLUMNAS_MODELO,
        "categoria",
    }
    faltantes = sorted(columnas_requeridas.difference(df.columns))

    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias en la respuesta de clasificación: "
            f"{faltantes}"
        )

    for columna in COLUMNAS_MODELO:
        df[columna] = pd.to_numeric(df[columna], errors="raise")

    df["categoria"] = df["categoria"].astype(str).str.upper()
    categorias_invalidas = sorted(
        set(df["categoria"]).difference(CATEGORIAS)
    )

    if categorias_invalidas:
        raise ValueError(
            f"Se encontraron categorías inválidas: {categorias_invalidas}"
        )

    return df.reset_index(drop=True)


def construir_matriz_modelo(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Reproduce la transformación actual de app/preprocessing/scaler.py:
    StandardScaler para ventas, visitas, precio y stock; en_promocion se
    conserva como 0/1.
    """
    scaler = StandardScaler()
    X = df.loc[:, COLUMNAS_MODELO].astype(float).copy()
    X.loc[:, COLUMNAS_ESCALABLES] = scaler.fit_transform(
        X.loc[:, COLUMNAS_ESCALABLES]
    )

    return X, scaler


def categorias_presentes(df: pd.DataFrame) -> list[str]:
    return [
        categoria
        for categoria in CATEGORIAS
        if categoria in set(df["categoria"])
    ]


def estilizar_eje(ax: Axes, usar_rejilla: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if usar_rejilla:
        ax.grid(
            True,
            color=COLOR_REJILLA,
            linewidth=0.7,
            alpha=0.75,
        )
        ax.set_axisbelow(True)


def guardar_figura(
    fig: Figure,
    output_path: Path,
    dpi: int,
) -> Path:
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def leyenda_categorias(categorias: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=COLORES_CATEGORIA[categoria],
            markeredgecolor="white",
            label=f"Categoría {categoria}",
        )
        for categoria in categorias
    ]


def generar_mapa_calor_perfil(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, list[str]]:
    categorias = categorias_presentes(df)
    variables_con_variacion = [
        columna
        for columna in COLUMNAS_MODELO
        if df[columna].nunique(dropna=False) > 1
    ]
    variables_omitidas = [
        columna
        for columna in COLUMNAS_MODELO
        if columna not in variables_con_variacion
    ]

    if not variables_con_variacion:
        raise ValueError(
            "Ninguna variable presenta variación; no se puede crear el mapa "
            "de calor."
        )

    scaler = StandardScaler()
    valores_z = pd.DataFrame(
        scaler.fit_transform(df[variables_con_variacion].astype(float)),
        columns=variables_con_variacion,
        index=df.index,
    )
    valores_z["categoria"] = df["categoria"].values
    perfil = (
        valores_z.groupby("categoria")[variables_con_variacion]
        .mean()
        .reindex(categorias)
    )

    limite = float(np.nanmax(np.abs(perfil.to_numpy())))
    limite = max(limite, 0.01)

    fig, ax = plt.subplots(
        figsize=(max(7.5, 1.8 * len(variables_con_variacion)), 4.2)
    )
    imagen = ax.imshow(
        perfil.to_numpy(),
        cmap="RdBu_r",
        vmin=-limite,
        vmax=limite,
        aspect="auto",
    )

    ax.set_title(
        "Perfil estandarizado de las categorías ABC",
        pad=16,
    )
    ax.set_xlabel("Variable del modelo")
    ax.set_ylabel("Categoría")
    ax.set_xticks(range(len(variables_con_variacion)))
    ax.set_xticklabels(
        [NOMBRES_VARIABLES[col] for col in variables_con_variacion],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(len(categorias)))
    ax.set_yticklabels(categorias)

    for fila in range(len(categorias)):
        for columna in range(len(variables_con_variacion)):
            valor = float(perfil.iloc[fila, columna])
            color = "white" if abs(valor) >= limite * 0.5 else COLOR_TEXTO
            ax.text(
                columna,
                fila,
                f"{valor:.2f}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

    colorbar = fig.colorbar(imagen, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label("Media estandarizada (z)")

    nota = (
        "Valores positivos indican un nivel medio superior al conjunto; "
        "valores negativos, un nivel inferior."
    )
    if variables_omitidas:
        nombres = ", ".join(
            NOMBRES_VARIABLES.get(col, col) for col in variables_omitidas
        )
        nota += f" Se omite por varianza cero: {nombres}."

    fig.text(
        0.5,
        -0.02,
        nota,
        ha="center",
        fontsize=8.5,
        color=COLOR_EJES,
    )
    fig.tight_layout()

    path = output_dir / "figura_01_perfil_multicriterio_abc.png"
    return guardar_figura(fig, path, dpi), variables_omitidas


def generar_dispersion_ventas_visitas(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    categorias = categorias_presentes(df)
    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    for categoria in categorias:
        grupo = df[df["categoria"] == categoria]
        ax.scatter(
            np.log1p(grupo["ventas_30d"]),
            np.log1p(grupo["visitas_30d"]),
            s=24,
            alpha=0.58,
            color=COLORES_CATEGORIA[categoria],
            edgecolors="none",
            label=f"Categoría {categoria} (n={len(grupo)})",
        )

        centro_x = float(np.log1p(grupo["ventas_30d"]).mean())
        centro_y = float(np.log1p(grupo["visitas_30d"]).mean())
        ax.scatter(
            centro_x,
            centro_y,
            marker="X",
            s=150,
            color=COLORES_CATEGORIA[categoria],
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"Centro {categoria}",
            (centro_x, centro_y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_title("Distribución comercial de las publicaciones")
    ax.set_xlabel("ln(1 + ventas en 30 días)")
    ax.set_ylabel("ln(1 + visitas en 30 días)")
    ax.legend(frameon=False, loc="best")
    estilizar_eje(ax)
    fig.text(
        0.5,
        0.01,
        (
            "La transformación logarítmica mejora la lectura sin alterar el "
            "orden de las observaciones."
        ),
        ha="center",
        fontsize=8.5,
        color=COLOR_EJES,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    path = output_dir / "figura_02_distribucion_ventas_visitas.png"
    return guardar_figura(fig, path, dpi)


def generar_grafico_silueta(
    X: pd.DataFrame,
    labels: pd.Series,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, float, dict[str, float]]:
    categorias = [
        categoria
        for categoria in CATEGORIAS
        if categoria in set(labels)
    ]

    if len(categorias) < 2:
        raise ValueError(
            "La silueta requiere al menos dos categorías."
        )

    valores = silhouette_samples(X, labels)
    promedio = float(silhouette_score(X, labels))
    medias_categoria = {
        categoria: float(
            valores[labels.to_numpy() == categoria].mean()
        )
        for categoria in categorias
    }

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    y_inferior = 10
    posiciones_texto: list[tuple[str, float]] = []

    for categoria in categorias:
        mascara = labels.to_numpy() == categoria
        valores_categoria = np.sort(valores[mascara])
        cantidad = len(valores_categoria)
        y_superior = y_inferior + cantidad

        ax.fill_betweenx(
            np.arange(y_inferior, y_superior),
            0,
            valores_categoria,
            facecolor=COLORES_CATEGORIA[categoria],
            edgecolor=COLORES_CATEGORIA[categoria],
            alpha=0.78,
        )
        posiciones_texto.append(
            (categoria, y_inferior + 0.5 * cantidad)
        )
        y_inferior = y_superior + 10

    ax.axvline(
        promedio,
        color=COLOR_DESTACADO,
        linestyle="--",
        linewidth=1.8,
        label=f"Silueta media = {promedio:.3f}",
    )
    ax.axvline(
        0,
        color=COLOR_EJES,
        linestyle=":",
        linewidth=1,
    )
    ax.set_title("Coeficiente de silueta por categoría")
    ax.set_xlabel("Coeficiente de silueta")
    ax.set_ylabel("Publicaciones agrupadas por categoría")
    ax.set_yticks([posicion for _, posicion in posiciones_texto])
    ax.set_yticklabels(
        [
            f"{categoria} (media={medias_categoria[categoria]:.3f})"
            for categoria, _ in posiciones_texto
        ]
    )
    ax.set_xlim(
        min(-0.15, float(np.nanmin(valores)) - 0.03),
        min(1.0, float(np.nanmax(valores)) + 0.08),
    )
    ax.legend(frameon=False, loc="lower right")
    estilizar_eje(ax)
    fig.tight_layout()

    path = output_dir / "figura_03_silueta_por_categoria.png"
    return (
        guardar_figura(fig, path, dpi),
        promedio,
        medias_categoria,
    )


def dibujar_pca_panel(
    ax: Axes,
    coordenadas: pd.DataFrame,
    categorias: Sequence[str],
    titulo: str,
    aplicar_zoom: bool,
) -> None:
    for categoria in categorias:
        grupo = coordenadas[coordenadas["categoria"] == categoria]
        ax.scatter(
            grupo["CP1"],
            grupo["CP2"],
            s=21,
            alpha=0.55,
            color=COLORES_CATEGORIA[categoria],
            edgecolors="none",
        )

        centro_x = float(grupo["CP1"].mean())
        centro_y = float(grupo["CP2"].mean())
        ax.scatter(
            centro_x,
            centro_y,
            marker="X",
            s=155,
            color=COLORES_CATEGORIA[categoria],
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            categoria,
            (centro_x, centro_y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    if aplicar_zoom:
        x_min, x_max = coordenadas["CP1"].quantile([0.01, 0.99])
        y_min, y_max = coordenadas["CP2"].quantile([0.01, 0.99])

        if x_min < x_max:
            margen_x = (x_max - x_min) * 0.08
            ax.set_xlim(x_min - margen_x, x_max + margen_x)

        if y_min < y_max:
            margen_y = (y_max - y_min) * 0.08
            ax.set_ylim(y_min - margen_y, y_max + margen_y)

    ax.set_title(titulo, fontsize=11)
    estilizar_eje(ax)


def generar_pca_centroides(
    X: pd.DataFrame,
    labels: pd.Series,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, list[float]]:
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X)
    explicada = [float(valor) for valor in pca.explained_variance_ratio_]

    coordenadas = pd.DataFrame(
        {
            "CP1": componentes[:, 0],
            "CP2": componentes[:, 1],
            "categoria": labels.to_numpy(),
        }
    )
    categorias = [
        categoria
        for categoria in CATEGORIAS
        if categoria in set(labels)
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.25),
        sharex=False,
        sharey=False,
    )
    dibujar_pca_panel(
        axes[0],
        coordenadas,
        categorias,
        "Vista completa",
        aplicar_zoom=False,
    )
    dibujar_pca_panel(
        axes[1],
        coordenadas,
        categorias,
        "Región central (percentiles 1–99)",
        aplicar_zoom=True,
    )

    for ax in axes:
        ax.set_xlabel(f"CP1 ({explicada[0] * 100:.1f}% de varianza)")
        ax.set_ylabel(f"CP2 ({explicada[1] * 100:.1f}% de varianza)")

    fig.suptitle(
        "Proyección PCA de las publicaciones y centroides por categoría",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )
    fig.legend(
        handles=leyenda_categorias(categorias),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(categorias),
        frameon=False,
    )
    fig.text(
        0.5,
        -0.01,
        (
            "La vista ampliada evita que las observaciones extremas oculten "
            "la estructura de la zona central."
        ),
        ha="center",
        fontsize=8.5,
        color=COLOR_EJES,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))

    path = output_dir / "figura_04_pca_categorias_centroides.png"
    return guardar_figura(fig, path, dpi), explicada


def normalizar_trials_optuna(
    contenido: Mapping[str, Any],
) -> pd.DataFrame:
    trials = contenido.get("trials")

    if not isinstance(trials, list) or not trials:
        raise ValueError(
            "El resultado de Optuna no contiene una lista no vacía en "
            "'trials'."
        )

    filas: list[dict[str, Any]] = []

    for trial in trials:
        if not isinstance(trial, Mapping):
            continue

        state = str(trial.get("state", "")).upper()
        value = trial.get("value")
        params = trial.get("params") or {}
        attrs = trial.get("attrs") or {}

        if state != "COMPLETE" or value is None:
            continue

        filas.append(
            {
                "number": int(trial.get("number", len(filas))),
                "value": float(value),
                "n_init": int(params["n_init"]),
                "tol": float(params["tol"]),
                "silhouette_std": float(
                    attrs.get("silhouette_std", 0.0)
                ),
                "pairwise_ari_mean": attrs.get("pairwise_ari_mean"),
            }
        )

    if not filas:
        raise ValueError(
            "Optuna no contiene trials COMPLETE con valor objetivo."
        )

    return pd.DataFrame(filas).sort_values("number").reset_index(drop=True)


def generar_historial_optuna(
    optuna_path: Path,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, dict[str, Any]]:
    contenido = leer_json(optuna_path)

    if not isinstance(contenido, Mapping):
        raise TypeError(
            "El resultado de Optuna debe ser un objeto JSON."
        )

    trials = normalizar_trials_optuna(contenido)
    best_index = int(trials["value"].idxmax())
    best_row = trials.loc[best_index]
    running_best = trials["value"].cummax()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.25))

    axes[0].errorbar(
        trials["number"],
        trials["value"],
        yerr=trials["silhouette_std"],
        fmt="o",
        color=COLORES_CATEGORIA["B"],
        ecolor=COLOR_REJILLA,
        elinewidth=1,
        capsize=2,
        markersize=4.8,
        alpha=0.88,
        label="Media por trial ± desviación",
    )
    axes[0].plot(
        trials["number"],
        running_best,
        color=COLOR_DESTACADO,
        linewidth=1.8,
        label="Mejor valor acumulado",
    )
    axes[0].scatter(
        [best_row["number"]],
        [best_row["value"]],
        marker="*",
        s=180,
        color=COLOR_DESTACADO,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    axes[0].set_title("Evolución de la optimización")
    axes[0].set_xlabel("Número de trial")
    axes[0].set_ylabel("Silueta media")
    axes[0].legend(frameon=False, fontsize=8.5)
    estilizar_eje(axes[0])

    n_init_values = sorted(trials["n_init"].unique())
    for n_init in n_init_values:
        grupo = trials[trials["n_init"] == n_init]
        axes[1].scatter(
            grupo["tol"],
            grupo["value"],
            s=46,
            alpha=0.78,
            label=f"n_init={n_init}",
        )

    axes[1].scatter(
        [best_row["tol"]],
        [best_row["value"]],
        marker="*",
        s=180,
        color=COLOR_DESTACADO,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    axes[1].annotate(
        (
            f"Mejor trial {int(best_row['number'])}\n"
            f"n_init={int(best_row['n_init'])}\n"
            f"tol={best_row['tol']:.2e}"
        ),
        (best_row["tol"], best_row["value"]),
        xytext=(9, -4),
        textcoords="offset points",
        fontsize=8.2,
        va="top",
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Relación entre parámetros y objetivo")
    axes[1].set_xlabel("Tolerancia (escala logarítmica)")
    axes[1].set_ylabel("Silueta media")
    axes[1].legend(frameon=False, fontsize=8.2)
    estilizar_eje(axes[1])

    fig.suptitle(
        "Optimización de hiperparámetros con Optuna",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = output_dir / "figura_05_optimizacion_optuna.png"
    best_params = contenido.get("best_params") or {
        "n_init": int(best_row["n_init"]),
        "tol": float(best_row["tol"]),
    }
    resumen = {
        "best_trial": int(contenido.get("best_trial", best_row["number"])),
        "best_value": float(
            contenido.get("best_value", best_row["value"])
        ),
        "best_params": best_params,
        "trials_complete": int(len(trials)),
    }

    return guardar_figura(fig, path, dpi), resumen


def seleccionar_corridas_semillas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "corrida" not in df.columns:
        raise ValueError(
            "El CSV de estabilidad no contiene la columna 'corrida'."
        )

    patron = re.compile(r"^seed_(\d+)$")
    filas: list[pd.Series] = []

    for _, row in df.iterrows():
        match = patron.match(str(row["corrida"]))
        if match:
            copia = row.copy()
            copia["random_state"] = int(match.group(1))
            filas.append(copia)

    if not filas:
        raise ValueError(
            "No se encontraron corridas con nombres seed_0, seed_1, etc."
        )

    return (
        pd.DataFrame(filas)
        .sort_values("random_state")
        .reset_index(drop=True)
    )


def generar_estabilidad_semillas(
    stability_path: Path,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, dict[str, Any]]:
    metricas = pd.read_csv(stability_path, encoding="utf-8-sig")
    corridas = seleccionar_corridas_semillas(metricas)

    columnas_requeridas = {
        "adjusted_rand_index",
        "silhouette",
    }
    faltantes = sorted(columnas_requeridas.difference(corridas.columns))

    if faltantes:
        raise ValueError(
            "Faltan columnas en el CSV de estabilidad: "
            f"{faltantes}"
        )

    for columna in (
        "adjusted_rand_index",
        "silhouette",
        "random_state",
    ):
        corridas[columna] = pd.to_numeric(
            corridas[columna],
            errors="raise",
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4.85))
    configuraciones = (
        (
            "adjusted_rand_index",
            "Índice Rand ajustado (ARI)",
            "Estabilidad de las asignaciones",
            (0.0, 1.05),
        ),
        (
            "silhouette",
            "Coeficiente de silueta",
            "Calidad interna por semilla",
            None,
        ),
    )

    for ax, (columna, ylabel, titulo, limites) in zip(
        axes,
        configuraciones,
    ):
        media = float(corridas[columna].mean())
        ax.plot(
            corridas["random_state"],
            corridas[columna],
            color=COLORES_CATEGORIA["B"],
            linewidth=1.2,
            marker="o",
            markersize=5,
        )
        ax.axhline(
            media,
            color=COLOR_DESTACADO,
            linestyle="--",
            linewidth=1.5,
            label=f"Media = {media:.3f}",
        )
        ax.set_title(titulo)
        ax.set_xlabel("random_state")
        ax.set_ylabel(ylabel)
        ax.set_xticks(corridas["random_state"].astype(int))
        if limites is not None:
            ax.set_ylim(*limites)
        ax.legend(frameon=False)
        estilizar_eje(ax)

    fig.suptitle(
        "Estabilidad de la clasificación frente a la inicialización",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = output_dir / "figura_06_estabilidad_random_state.png"
    resumen: dict[str, Any] = {
        "corridas": int(len(corridas)),
        "ari_mean": float(corridas["adjusted_rand_index"].mean()),
        "ari_std": float(corridas["adjusted_rand_index"].std(ddof=1)),
        "silhouette_mean": float(corridas["silhouette"].mean()),
        "silhouette_std": float(corridas["silhouette"].std(ddof=1)),
    }

    if "corridas_convergentes" in corridas.columns:
        valores = pd.to_numeric(
            corridas["corridas_convergentes"],
            errors="coerce",
        ).dropna()
        if not valores.empty:
            resumen["n_init_inferido"] = int(valores.mode().iloc[0])

    return guardar_figura(fig, path, dpi), resumen


def buscar_columna(
    columnas: Iterable[str],
    candidatos: Sequence[str],
) -> Optional[str]:
    disponibles = {str(columna).lower(): str(columna) for columna in columnas}

    for candidato in candidatos:
        if candidato.lower() in disponibles:
            return disponibles[candidato.lower()]

    return None


def generar_estabilidad_bootstrap(
    bootstrap_path: Path,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, dict[str, Any]]:
    bootstrap = pd.read_csv(bootstrap_path, encoding="utf-8-sig")
    ari_col = buscar_columna(
        bootstrap.columns,
        ("adjusted_rand_index", "ari"),
    )

    if ari_col is None:
        raise ValueError(
            "El CSV de bootstrap debe contener adjusted_rand_index o ari."
        )

    jaccard_cols = {
        categoria: buscar_columna(
            bootstrap.columns,
            (
                f"jaccard_{categoria}",
                f"jaccard_{categoria.lower()}",
                f"cluster_{categoria}_jaccard",
            ),
        )
        for categoria in CATEGORIAS
    }
    jaccard_cols = {
        categoria: columna
        for categoria, columna in jaccard_cols.items()
        if columna is not None
    }

    ari = pd.to_numeric(bootstrap[ari_col], errors="raise").dropna()

    if ari.empty:
        raise ValueError(
            "La columna ARI del bootstrap no contiene valores válidos."
        )

    columnas_figura = 2 if jaccard_cols else 1
    fig, axes_obj = plt.subplots(
        1,
        columnas_figura,
        figsize=(11.7 if columnas_figura == 2 else 6.4, 4.9),
    )
    axes = (
        list(np.atleast_1d(axes_obj))
        if columnas_figura > 1
        else [axes_obj]
    )

    axes[0].boxplot(
        [ari.to_numpy()],
        vert=True,
        patch_artist=True,
        tick_labels=["ARI"],
        boxprops={"facecolor": COLORES_CATEGORIA["B"], "alpha": 0.75},
        medianprops={"color": COLOR_DESTACADO, "linewidth": 1.8},
    )
    jitter = np.random.default_rng(42).normal(1.0, 0.025, len(ari))
    axes[0].scatter(
        jitter,
        ari,
        s=15,
        alpha=0.4,
        color=COLOR_TEXTO,
        edgecolors="none",
    )
    axes[0].set_title("Estabilidad global")
    axes[0].set_ylabel("Índice Rand ajustado")
    axes[0].set_ylim(0, 1.05)
    estilizar_eje(axes[0])

    resumen = {
        "replicas": int(len(ari)),
        "ari_mean": float(ari.mean()),
        "ari_std": float(ari.std(ddof=1)),
        "ari_median": float(ari.median()),
    }

    if jaccard_cols:
        datos = []
        etiquetas = []
        colores = []

        for categoria in CATEGORIAS:
            columna = jaccard_cols.get(categoria)
            if columna is None:
                continue
            serie = pd.to_numeric(
                bootstrap[columna],
                errors="raise",
            ).dropna()
            datos.append(serie.to_numpy())
            etiquetas.append(categoria)
            colores.append(COLORES_CATEGORIA[categoria])
            resumen[f"jaccard_{categoria}_mean"] = float(serie.mean())

        boxplot = axes[1].boxplot(
            datos,
            vert=True,
            patch_artist=True,
            tick_labels=etiquetas,
            medianprops={
                "color": COLOR_DESTACADO,
                "linewidth": 1.8,
            },
        )
        for caja, color in zip(boxplot["boxes"], colores):
            caja.set_facecolor(color)
            caja.set_alpha(0.78)

        axes[1].set_title("Estabilidad por categoría")
        axes[1].set_xlabel("Categoría")
        axes[1].set_ylabel("Índice de Jaccard")
        axes[1].set_ylim(0, 1.05)
        estilizar_eje(axes[1])

    fig.suptitle(
        "Validación de estabilidad mediante bootstrap",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = output_dir / "figura_07_estabilidad_bootstrap.png"
    return guardar_figura(fig, path, dpi), resumen


def generar_anexo_precio_stock(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    categorias = categorias_presentes(df)
    fig, ax = plt.subplots(figsize=(8.3, 5.45))

    for categoria in categorias:
        grupo = df[df["categoria"] == categoria]
        ax.scatter(
            np.log1p(grupo["precio_actual"]),
            np.log1p(grupo["stock_actual"]),
            s=23,
            alpha=0.55,
            color=COLORES_CATEGORIA[categoria],
            edgecolors="none",
            label=f"Categoría {categoria}",
        )

    ax.set_title("Distribución económica y de disponibilidad")
    ax.set_xlabel("ln(1 + precio actual)")
    ax.set_ylabel("ln(1 + stock actual)")
    ax.legend(frameon=False)
    estilizar_eje(ax)
    fig.text(
        0.5,
        0.01,
        (
            "Figura complementaria: su capacidad explicativa disminuye "
            "cuando el stock presenta varianza baja o nula."
        ),
        ha="center",
        fontsize=8.5,
        color=COLOR_EJES,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    path = output_dir / "anexo_01_distribucion_precio_stock.png"
    return guardar_figura(fig, path, dpi)


def crear_resumen_categorias(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    categorias = categorias_presentes(df)
    total = len(df)
    filas: list[dict[str, Any]] = []

    for categoria in categorias:
        grupo = df[df["categoria"] == categoria]
        fila: dict[str, Any] = {
            "categoria": categoria,
            "cantidad": int(len(grupo)),
            "porcentaje": float(len(grupo) / total * 100),
        }

        for columna in COLUMNAS_MODELO:
            fila[f"{columna}_media"] = float(grupo[columna].mean())
            fila[f"{columna}_mediana"] = float(grupo[columna].median())
            fila[f"{columna}_total"] = float(grupo[columna].sum())

        filas.append(fila)

    resumen = pd.DataFrame(filas)
    path = output_dir / "resumen_por_categoria.csv"
    resumen.to_csv(path, index=False, encoding="utf-8-sig")

    return path, resumen


def diagnosticar_coherencia_parametros(
    optuna_summary: Optional[Mapping[str, Any]],
    stability_summary: Optional[Mapping[str, Any]],
) -> list[str]:
    advertencias: list[str] = []

    if optuna_summary is None:
        return advertencias

    best_params = optuna_summary.get("best_params")

    if isinstance(best_params, Mapping):
        if best_params.get("shuffle_unlabeled") is False:
            advertencias.append(
                "El resultado de Optuna fue obtenido fijando "
                "shuffle_unlabeled=False. Si la configuración definitiva ya "
                "no fija ese valor, repite Optuna antes de citar sus "
                "parámetros como definitivos."
            )

    if stability_summary is None:
        return advertencias

    n_init_estabilidad = stability_summary.get("n_init_inferido")

    if isinstance(best_params, Mapping) and n_init_estabilidad is not None:
        n_init_optuna = best_params.get("n_init")
        if (
            n_init_optuna is not None
            and int(n_init_optuna) != int(n_init_estabilidad)
        ):
            advertencias.append(
                "El CSV de estabilidad parece haberse generado con "
                f"n_init={n_init_estabilidad}, mientras que Optuna recomienda "
                f"n_init={n_init_optuna}. Repite la estabilidad antes de usar "
                "la figura como resultado definitivo."
            )

    return advertencias


def imprimir_archivos(
    titulo: str,
    archivos: Sequence[Path],
) -> None:
    print(f"\n{titulo}")
    for archivo in archivos:
        print(f"  - {archivo}")


def main() -> None:
    args = parsear_argumentos()
    aplicar_estilo()

    if args.dpi < 72:
        raise ValueError("--dpi debe ser mayor o igual a 72.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    response = obtener_respuesta_clasificacion(
        input_request=args.input_request,
        classification_response=args.classification_response,
    )
    response_path = output_dir / "classification_response.json"
    guardar_json(response_path, response)

    df = construir_dataframe(response)
    X, _ = construir_matriz_modelo(df)
    labels = df["categoria"].copy()

    figuras_principales: list[Path] = []
    figuras_condicionales: list[Path] = []
    figuras_anexo: list[Path] = []

    heatmap_path, variables_omitidas = generar_mapa_calor_perfil(
        df,
        output_dir,
        args.dpi,
    )
    figuras_principales.append(heatmap_path)

    figuras_principales.append(
        generar_dispersion_ventas_visitas(
            df,
            output_dir,
            args.dpi,
        )
    )

    silhouette_path, silhouette_mean, silhouette_by_category = (
        generar_grafico_silueta(
            X,
            labels,
            output_dir,
            args.dpi,
        )
    )
    figuras_principales.append(silhouette_path)

    pca_path, explained_variance = generar_pca_centroides(
        X,
        labels,
        output_dir,
        args.dpi,
    )
    figuras_principales.append(pca_path)

    optuna_summary: Optional[dict[str, Any]] = None
    if args.optuna_result is not None:
        optuna_path, optuna_summary = generar_historial_optuna(
            args.optuna_result,
            output_dir,
            args.dpi,
        )
        figuras_principales.append(optuna_path)

    stability_summary: Optional[dict[str, Any]] = None
    if args.stability_metrics is not None:
        stability_path, stability_summary = generar_estabilidad_semillas(
            args.stability_metrics,
            output_dir,
            args.dpi,
        )
        figuras_condicionales.append(stability_path)

    bootstrap_summary: Optional[dict[str, Any]] = None
    if args.bootstrap_metrics is not None:
        bootstrap_path, bootstrap_summary = generar_estabilidad_bootstrap(
            args.bootstrap_metrics,
            output_dir,
            args.dpi,
        )
        figuras_condicionales.append(bootstrap_path)

    if args.include_appendix:
        figuras_anexo.append(
            generar_anexo_precio_stock(
                df,
                output_dir,
                args.dpi,
            )
        )

    resumen_path, resumen_categorias = crear_resumen_categorias(
        df,
        output_dir,
    )

    diagnostico = response.get("diagnostico") or {}
    advertencias = diagnosticar_coherencia_parametros(
        optuna_summary,
        stability_summary,
    )

    categorias_silueta_negativa = [
        categoria
        for categoria, valor in silhouette_by_category.items()
        if valor < 0
    ]
    if categorias_silueta_negativa:
        detalle = ", ".join(
            (
                f"{categoria}="
                f"{silhouette_by_category[categoria]:.3f}"
            )
            for categoria in categorias_silueta_negativa
        )
        advertencias.append(
            "La silueta media por categoría es negativa en "
            f"{detalle}. Esto indica solapamiento o asignaciones más próximas "
            "a otro grupo y debe discutirse como limitación del resultado."
        )

    if args.bootstrap_metrics is None:
        advertencias.append(
            "No se proporcionó --bootstrap-metrics. La figura de bootstrap "
            "queda pendiente hasta ejecutar esa validación."
        )

    manifest = {
        "dataset": {
            "publicaciones": int(len(df)),
            "categorias": {
                str(row["categoria"]): {
                    "cantidad": int(row["cantidad"]),
                    "porcentaje": float(row["porcentaje"]),
                }
                for _, row in resumen_categorias.iterrows()
            },
            "variables_modelo": list(COLUMNAS_MODELO),
            "variables_omitidas_heatmap_varianza_cero": variables_omitidas,
        },
        "clasificacion": {
            "diagnostico_api": diagnostico,
            "silhouette_recalculada": silhouette_mean,
            "silhouette_por_categoria": silhouette_by_category,
            "pca_varianza_explicada": {
                "CP1": explained_variance[0],
                "CP2": explained_variance[1],
                "total_2_componentes": float(sum(explained_variance)),
            },
        },
        "optuna": optuna_summary,
        "estabilidad_random_state": stability_summary,
        "estabilidad_bootstrap": bootstrap_summary,
        "seleccion_figuras": {
            "principales": [path.name for path in figuras_principales],
            "condicionales_validacion": [
                path.name for path in figuras_condicionales
            ],
            "anexos": [path.name for path in figuras_anexo],
            "no_recomendado_como_figura": (
                "Conteo A/B/C: las proporciones están impuestas por diseño; "
                "se conserva en resumen_por_categoria.csv."
            ),
        },
        "advertencias": advertencias,
    }
    manifest_path = output_dir / "manifest_resultados.json"
    guardar_json(manifest_path, manifest)

    imprimir_archivos(
        "Figuras principales generadas:",
        figuras_principales,
    )

    if figuras_condicionales:
        imprimir_archivos(
            "Figuras de validación generadas:",
            figuras_condicionales,
        )

    if figuras_anexo:
        imprimir_archivos(
            "Figuras de anexo generadas:",
            figuras_anexo,
        )

    print("\nArchivos auxiliares:")
    print(f"  - {response_path}")
    print(f"  - {resumen_path}")
    print(f"  - {manifest_path}")

    if advertencias:
        print("\nAdvertencias antes de usar las figuras como definitivas:")
        for advertencia in advertencias:
            print(f"  - {advertencia}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
