from __future__ import annotations

"""
Genera las figuras principales de resultados del ClasificadorABC.

Preprocesamiento reproducido:

1. Selección de ventas_30d, visitas_30d y precio_actual.
2. Transformación log1p de ventas_30d y visitas_30d.
3. Estandarización mediante StandardScaler.
4. Cálculo de silueta y PCA sobre el mismo espacio del modelo.

Ejemplo ejecutando nuevamente la clasificación:

    python generate_result_figures.py ^
        --input-request data/input_request.json ^
        --output-dir artifacts/result_figures_final

Ejemplo utilizando una respuesta existente:

    python generate_result_figures.py ^
        --classification-response data/classification_response.json ^
        --output-dir artifacts/result_figures_final
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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

# Únicamente estas variables ingresan al clustering.
COLUMNAS_MODELO = (
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
)

# Estas variables reciben log1p antes de StandardScaler.
COLUMNAS_LOGARITMICAS = (
    "ventas_30d",
    "visitas_30d",
)

# Se conservan para el resumen descriptivo, pero no entran al modelo.
COLUMNAS_CONTEXTO = (
    "stock_actual",
    "en_promocion",
)

COLUMNAS_RESPUESTA = (
    *COLUMNAS_MODELO,
    *COLUMNAS_CONTEXTO,
)

NOMBRES_VARIABLES = {
    "ventas_30d": "Ventas (log1p)",
    "visitas_30d": "Visitas (log1p)",
    "precio_actual": "Precio actual",
    "stock_actual": "Stock actual",
    "en_promocion": "En promoción",
}

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
            "Ejecuta o reutiliza la clasificación ABC y genera "
            "las figuras principales del capítulo de resultados."
        )
    )

    fuente = parser.add_mutually_exclusive_group(required=True)

    fuente.add_argument(
        "--input-request",
        type=Path,
        help=(
            "Request JSON con la clave productos. La clasificación se "
            "ejecutará utilizando los servicios reales de la aplicación."
        ),
    )

    fuente.add_argument(
        "--classification-response",
        type=Path,
        help=(
            "Respuesta JSON generada previamente mediante "
            "POST /api/clasificar."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/result_figures"),
        help="Directorio donde se guardarán los resultados.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolución de las figuras PNG.",
    )

    return parser.parse_args()


def aplicar_estilo() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.edgecolor": COLOR_EJES,
            "axes.labelcolor": COLOR_TEXTO,
            "xtick.color": COLOR_TEXTO,
            "ytick.color": COLOR_TEXTO,
            "text.color": COLOR_TEXTO,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


def leer_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"El archivo no contiene JSON válido: {path}. "
            f"Detalle: {error}"
        ) from error


def convertir_json_serializable(valor: Any) -> Any:
    if isinstance(valor, Mapping):
        return {
            str(clave): convertir_json_serializable(contenido)
            for clave, contenido in valor.items()
        }

    if isinstance(valor, (list, tuple)):
        return [
            convertir_json_serializable(item)
            for item in valor
        ]

    if isinstance(valor, Path):
        return str(valor)

    if isinstance(valor, np.generic):
        return valor.item()

    if isinstance(valor, float) and np.isnan(valor):
        return None

    return valor


def guardar_json(
    path: Path,
    contenido: Mapping[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            convertir_json_serializable(contenido),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def ejecutar_clasificacion_desde_request(
    input_path: Path,
) -> dict[str, Any]:
    try:
        from app.schemas.input_schema import RequestInput
        from app.services.clasificacion_service import (
            ejecutar_clasificacion,
        )
    except ImportError as error:
        raise RuntimeError(
            "No fue posible importar la aplicación. "
            "Coloca este archivo en la raíz de ClasificadorABC "
            "y ejecútalo desde esa carpeta."
        ) from error

    request_payload = leer_json(input_path)
    request_data = RequestInput.model_validate(
        request_payload
    )

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
        return ejecutar_clasificacion_desde_request(
            input_request
        )

    if classification_response is None:
        raise ValueError(
            "Debe indicarse una fuente de clasificación."
        )

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
            "La respuesta no contiene una lista no vacía "
            "en la clave resultados."
        )

    df = pd.DataFrame(resultados)

    columnas_requeridas = {
        "publication_id",
        *COLUMNAS_MODELO,
        "categoria",
    }

    faltantes = sorted(
        columnas_requeridas.difference(df.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias en la respuesta: "
            f"{faltantes}"
        )

    for columna in COLUMNAS_RESPUESTA:
        if columna in df.columns:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="raise",
            )

    df["categoria"] = (
        df["categoria"]
        .astype(str)
        .str.upper()
    )

    categorias_invalidas = sorted(
        set(df["categoria"]).difference(CATEGORIAS)
    )

    if categorias_invalidas:
        raise ValueError(
            "Se encontraron categorías inválidas: "
            f"{categorias_invalidas}"
        )

    return df.reset_index(drop=True)


def preparar_variables_modelo(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selecciona las tres variables actuales y aplica log1p a los
    dos conteos utilizados por el modelo.
    """
    X_preparado = (
        df.loc[:, COLUMNAS_MODELO]
        .astype(float)
        .copy()
    )

    if (
        X_preparado.loc[:, COLUMNAS_LOGARITMICAS] < 0
    ).any().any():
        raise ValueError(
            "Ventas y visitas no pueden contener valores negativos."
        )

    X_preparado.loc[:, COLUMNAS_LOGARITMICAS] = np.log1p(
        X_preparado.loc[:, COLUMNAS_LOGARITMICAS]
    )

    return X_preparado


def construir_matriz_modelo(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Reproduce exactamente el preprocesamiento del modelo:

    - log1p para ventas y visitas;
    - StandardScaler para ventas, visitas y precio.
    """
    X_preparado = preparar_variables_modelo(df)

    scaler = StandardScaler()

    X_escalado = pd.DataFrame(
        scaler.fit_transform(X_preparado),
        columns=COLUMNAS_MODELO,
        index=df.index,
    )

    return X_escalado, scaler


def categorias_presentes(
    df: pd.DataFrame,
) -> list[str]:
    categorias_encontradas = set(df["categoria"])

    return [
        categoria
        for categoria in CATEGORIAS
        if categoria in categorias_encontradas
    ]


def estilizar_eje(
    ax: Axes,
    usar_rejilla: bool = True,
) -> None:
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
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def leyenda_categorias(
    categorias: Sequence[str],
) -> list[Line2D]:
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

    X_preparado = preparar_variables_modelo(df)

    desviaciones = X_preparado.std(ddof=0)

    variables_omitidas = [
        columna
        for columna in COLUMNAS_MODELO
        if np.isclose(
            desviaciones[columna],
            0.0,
        )
    ]

    variables_utilizadas = [
        columna
        for columna in COLUMNAS_MODELO
        if columna not in variables_omitidas
    ]

    if not variables_utilizadas:
        raise ValueError(
            "Ninguna variable posee varianza suficiente "
            "para generar el mapa de calor."
        )

    scaler = StandardScaler()

    X_estandarizado = pd.DataFrame(
        scaler.fit_transform(
            X_preparado[variables_utilizadas]
        ),
        columns=variables_utilizadas,
        index=df.index,
    )

    perfil = (
        X_estandarizado
        .assign(
            categoria=df["categoria"].to_numpy()
        )
        .groupby("categoria")[variables_utilizadas]
        .mean()
        .reindex(categorias)
    )

    valores = perfil.to_numpy(dtype=float)

    max_abs = float(
        np.nanmax(np.abs(valores))
    )

    if np.isclose(max_abs, 0.0):
        max_abs = 1.0

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    imagen = ax.imshow(
        valores,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_title(
        "Perfil estandarizado de las categorías ABC",
        pad=18,
    )

    ax.set_xlabel("Variable del modelo")
    ax.set_ylabel("Categoría")

    ax.set_xticks(
        np.arange(len(variables_utilizadas))
    )

    ax.set_xticklabels(
        [
            NOMBRES_VARIABLES[columna]
            for columna in variables_utilizadas
        ],
        rotation=22,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(categorias))
    )
    ax.set_yticklabels(categorias)

    for fila in range(len(categorias)):
        for columna in range(
            len(variables_utilizadas)
        ):
            valor = valores[fila, columna]

            color_texto = (
                "white"
                if abs(valor) > max_abs * 0.55
                else COLOR_TEXTO
            )

            ax.text(
                columna,
                fila,
                f"{valor:.2f}",
                ha="center",
                va="center",
                color=color_texto,
                fontweight="bold",
                fontsize=11,
            )

    colorbar = fig.colorbar(
        imagen,
        ax=ax,
        fraction=0.035,
        pad=0.04,
    )

    colorbar.set_label(
        "Media estandarizada (z)"
    )

    fig.text(
        0.5,
        0.01,
        (
            "Valores positivos indican un nivel medio superior al conjunto; "
            "valores negativos indican un nivel inferior."
        ),
        ha="center",
        fontsize=9.5,
        color=COLOR_EJES,
    )

    fig.tight_layout(
        rect=(0, 0.05, 1, 1)
    )

    output_path = (
        output_dir
        / "figura_01_perfil_multicriterio_abc.png"
    )

    return (
        guardar_figura(
            fig,
            output_path,
            dpi,
        ),
        variables_omitidas,
    )


def generar_dispersion_ventas_visitas(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    categorias = categorias_presentes(df)

    fig, ax = plt.subplots(
        figsize=(7.2, 5.5)
    )

    ventas_log = np.log1p(
        df["ventas_30d"].astype(float)
    )

    visitas_log = np.log1p(
        df["visitas_30d"].astype(float)
    )

    for categoria in categorias:
        mascara = df["categoria"] == categoria

        ax.scatter(
            ventas_log[mascara],
            visitas_log[mascara],
            s=26,
            alpha=0.58,
            color=COLORES_CATEGORIA[categoria],
            edgecolors="none",
            label=(
                f"Categoría {categoria} "
                f"(n={int(mascara.sum())})"
            ),
        )

        centro_x = float(
            ventas_log[mascara].mean()
        )
        centro_y = float(
            visitas_log[mascara].mean()
        )

        ax.scatter(
            centro_x,
            centro_y,
            marker="X",
            s=145,
            color=COLORES_CATEGORIA[categoria],
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )

        ax.annotate(
            f"Centro {categoria}",
            (centro_x, centro_y),
            xytext=(7, 6),
            textcoords="offset points",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title(
        "Distribución comercial de las publicaciones"
    )
    ax.set_xlabel(
        "ln(1 + ventas en 30 días)"
    )
    ax.set_ylabel(
        "ln(1 + visitas en 30 días)"
    )

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    estilizar_eje(ax)

    fig.text(
        0.5,
        0.01,
        (
            "La transformación logarítmica mejora la lectura "
            "sin alterar el orden de las observaciones."
        ),
        ha="center",
        fontsize=9.5,
        color=COLOR_EJES,
    )

    fig.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    output_path = (
        output_dir
        / "figura_02_distribucion_ventas_visitas.png"
    )

    return guardar_figura(
        fig,
        output_path,
        dpi,
    )


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

    valores_silueta = silhouette_samples(
        X,
        labels,
    )

    media_global = float(
        silhouette_score(
            X,
            labels,
        )
    )

    media_por_categoria: dict[str, float] = {}

    cantidades = {
        categoria: int(
            (labels == categoria).sum()
        )
        for categoria in categorias
    }

    espacio = 14
    altura_total = (
        sum(cantidades.values())
        + espacio * (len(categorias) + 1)
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.8)
    )

    posicion_inicial = espacio
    centros_y: list[float] = []
    etiquetas_y: list[str] = []

    for categoria in categorias:
        mascara = (
            labels.to_numpy() == categoria
        )

        valores_categoria = np.sort(
            valores_silueta[mascara]
        )

        cantidad = len(valores_categoria)
        posicion_final = (
            posicion_inicial + cantidad
        )

        posiciones = np.arange(
            posicion_inicial,
            posicion_final,
        )

        ax.fill_betweenx(
            posiciones,
            0,
            valores_categoria,
            facecolor=COLORES_CATEGORIA[categoria],
            edgecolor=COLORES_CATEGORIA[categoria],
            alpha=0.78,
        )

        media_categoria = float(
            valores_categoria.mean()
        )

        media_por_categoria[categoria] = (
            media_categoria
        )

        centro = (
            posicion_inicial
            + posicion_final
        ) / 2

        centros_y.append(centro)
        etiquetas_y.append(
            f"{categoria} (media={media_categoria:.3f})"
        )

        posicion_inicial = (
            posicion_final + espacio
        )

    ax.axvline(
        0,
        color=COLOR_EJES,
        linestyle=":",
        linewidth=1.2,
    )

    ax.axvline(
        media_global,
        color=COLOR_DESTACADO,
        linestyle="--",
        linewidth=1.8,
        label=(
            f"Silueta media = {media_global:.3f}"
        ),
    )

    ax.set_title(
        "Coeficiente de silueta por categoría"
    )
    ax.set_xlabel(
        "Coeficiente de silueta"
    )
    ax.set_ylabel(
        "Publicaciones agrupadas por categoría"
    )

    ax.set_yticks(centros_y)
    ax.set_yticklabels(etiquetas_y)
    ax.set_ylim(0, altura_total)
    ax.set_xlim(-1.0, 1.0)

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    estilizar_eje(ax)

    fig.tight_layout()

    output_path = (
        output_dir
        / "figura_03_silueta_por_categoria.png"
    )

    return (
        guardar_figura(
            fig,
            output_path,
            dpi,
        ),
        media_global,
        media_por_categoria,
    )


def generar_pca_centroides(
    X: pd.DataFrame,
    labels: pd.Series,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, list[float]]:
    categorias = [
        categoria
        for categoria in CATEGORIAS
        if categoria in set(labels)
    ]

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    componentes = pca.fit_transform(X)

    df_pca = pd.DataFrame(
        componentes,
        columns=["CP1", "CP2"],
        index=X.index,
    )

    df_pca["categoria"] = (
        labels.to_numpy()
    )

    centroides_originales = (
        X.assign(
            categoria=labels.to_numpy()
        )
        .groupby("categoria")
        .mean()
        .reindex(categorias)
    )

    centroides_pca = pca.transform(
        centroides_originales
    )

    varianza = (
        pca.explained_variance_ratio_
        .astype(float)
        .tolist()
    )

    limite_central_x = np.percentile(
        df_pca["CP1"],
        [1, 99],
    )

    limite_central_y = np.percentile(
        df_pca["CP2"],
        [1, 99],
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.8)
    )

    for indice, categoria in enumerate(
        categorias
    ):
        grupo = df_pca[
            df_pca["categoria"] == categoria
        ]

        ax.scatter(
            grupo["CP1"],
            grupo["CP2"],
            s=30,
            alpha=0.62,
            color=COLORES_CATEGORIA[categoria],
            edgecolors="white",
            linewidths=0.25,
        )

        centro_x = float(
            centroides_pca[indice, 0]
        )

        centro_y = float(
            centroides_pca[indice, 1]
        )

        ax.scatter(
            centro_x,
            centro_y,
            marker="X",
            s=220,
            color=COLORES_CATEGORIA[categoria],
            edgecolors="white",
            linewidths=1.6,
            zorder=5,
        )

        ax.annotate(
            f"Centro {categoria}",
            (centro_x, centro_y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(
        "Proyección PCA de las publicaciones y centroides",
        pad=54,
    )

    ax.set_xlabel(
        f"CP1 ({varianza[0] * 100:.1f}% de varianza)"
    )

    ax.set_ylabel(
        f"CP2 ({varianza[1] * 100:.1f}% de varianza)"
    )

    ax.set_xlim(
        float(limite_central_x[0]),
        float(limite_central_x[1]),
    )

    ax.set_ylim(
        float(limite_central_y[0]),
        float(limite_central_y[1]),
    )

    ax.legend(
        handles=leyenda_categorias(categorias),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(categorias),
        frameon=False,
    )

    estilizar_eje(ax)

    fig.tight_layout()

    output_path = (
        output_dir
        / "figura_04_pca_categorias_centroides.png"
    )

    return (
        guardar_figura(
            fig,
            output_path,
            dpi,
        ),
        varianza,
    )


def crear_resumen_categorias(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    categorias = categorias_presentes(df)
    total = len(df)

    filas: list[dict[str, Any]] = []

    for categoria in categorias:
        grupo = df[
            df["categoria"] == categoria
        ]

        fila: dict[str, Any] = {
            "categoria": categoria,
            "cantidad": int(len(grupo)),
            "porcentaje": float(
                len(grupo) / total * 100
            ),
        }

        for columna in COLUMNAS_RESPUESTA:
            if columna not in grupo.columns:
                continue

            fila[f"{columna}_media"] = float(
                grupo[columna].mean()
            )

            fila[f"{columna}_mediana"] = float(
                grupo[columna].median()
            )

            fila[f"{columna}_total"] = float(
                grupo[columna].sum()
            )

        filas.append(fila)

    resumen = pd.DataFrame(filas)

    output_path = (
        output_dir
        / "resumen_por_categoria.csv"
    )

    resumen.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path, resumen


def obtener_diagnostico(
    response: Mapping[str, Any],
) -> Mapping[str, Any]:
    diagnostico = response.get("diagnostico")

    if isinstance(diagnostico, Mapping):
        return diagnostico

    return {}


def verificar_silueta_api(
    response: Mapping[str, Any],
    silhouette_recalculada: float,
) -> Optional[float]:
    diagnostico = obtener_diagnostico(response)
    metricas = diagnostico.get("metricas")

    if not isinstance(metricas, Mapping):
        return None

    silhouette_api = metricas.get("silhouette")

    if silhouette_api is None:
        return None

    silhouette_api = float(silhouette_api)

    if not np.isclose(
        silhouette_api,
        silhouette_recalculada,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "La silueta recalculada no coincide con la API. "
            f"API={silhouette_api:.6f}; "
            f"gráficos={silhouette_recalculada:.6f}. "
            "El generador no está reproduciendo el mismo "
            "preprocesamiento utilizado por el modelo."
        )

    return silhouette_api


def main() -> None:
    args = parsear_argumentos()
    aplicar_estilo()

    if args.dpi < 72:
        raise ValueError(
            "--dpi debe ser mayor o igual a 72."
        )

    output_dir: Path = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    response = obtener_respuesta_clasificacion(
        input_request=args.input_request,
        classification_response=(
            args.classification_response
        ),
    )

    response_path = (
        output_dir
        / "classification_response.json"
    )

    guardar_json(
        response_path,
        response,
    )

    df = construir_dataframe(response)

    X, _ = construir_matriz_modelo(df)

    labels = (
        df["categoria"]
        .astype(str)
        .reset_index(drop=True)
    )

    X = X.reset_index(drop=True)

    figuras: list[Path] = []

    heatmap_path, variables_omitidas = (
        generar_mapa_calor_perfil(
            df=df,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    figuras.append(heatmap_path)

    figuras.append(
        generar_dispersion_ventas_visitas(
            df=df,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    (
        silhouette_path,
        silhouette_media,
        silhouette_por_categoria,
    ) = generar_grafico_silueta(
        X=X,
        labels=labels,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    figuras.append(silhouette_path)

    (
        pca_path,
        varianza_explicada,
    ) = generar_pca_centroides(
        X=X,
        labels=labels,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    figuras.append(pca_path)

    silhouette_api = verificar_silueta_api(
        response=response,
        silhouette_recalculada=silhouette_media,
    )

    resumen_path, resumen_categorias = (
        crear_resumen_categorias(
            df=df,
            output_dir=output_dir,
        )
    )

    diagnostico = obtener_diagnostico(
        response
    )

    advertencias: list[str] = []

    categorias_silueta_negativa = {
        categoria: valor
        for categoria, valor
        in silhouette_por_categoria.items()
        if valor < 0
    }

    if categorias_silueta_negativa:
        detalle = ", ".join(
            f"{categoria}={valor:.3f}"
            for categoria, valor
            in categorias_silueta_negativa.items()
        )

        advertencias.append(
            "La silueta media es negativa en "
            f"{detalle}. Estas categorías presentan "
            "solapamiento parcial con otros grupos."
        )

    manifest = {
        "dataset": {
            "publicaciones": int(len(df)),
            "categorias": {
                str(fila["categoria"]): {
                    "cantidad": int(
                        fila["cantidad"]
                    ),
                    "porcentaje": float(
                        fila["porcentaje"]
                    ),
                }
                for _, fila
                in resumen_categorias.iterrows()
            },
            "variables_modelo": list(
                COLUMNAS_MODELO
            ),
            "transformaciones": {
                "ventas_30d": (
                    "log1p + StandardScaler"
                ),
                "visitas_30d": (
                    "log1p + StandardScaler"
                ),
                "precio_actual": (
                    "StandardScaler"
                ),
            },
            "variables_contexto_excluidas_modelo": [
                columna
                for columna in COLUMNAS_CONTEXTO
                if columna in df.columns
            ],
            "variables_omitidas_heatmap_varianza_cero": (
                variables_omitidas
            ),
        },
        "clasificacion": {
            "diagnostico_api": diagnostico,
            "silhouette_api": silhouette_api,
            "silhouette_recalculada": (
                silhouette_media
            ),
            "silhouette_por_categoria": (
                silhouette_por_categoria
            ),
            "pca_varianza_explicada": {
                "CP1": float(
                    varianza_explicada[0]
                ),
                "CP2": float(
                    varianza_explicada[1]
                ),
                "total_2_componentes": float(
                    sum(varianza_explicada)
                ),
            },
        },
        "figuras_generadas": [
            path.name
            for path in figuras
        ],
        "archivos_auxiliares": [
            response_path.name,
            resumen_path.name,
        ],
        "advertencias": advertencias,
    }

    manifest_path = (
        output_dir
        / "manifest_resultados.json"
    )

    guardar_json(
        manifest_path,
        manifest,
    )

    print(
        "\nFiguras generadas correctamente:"
    )

    for figura in figuras:
        print(f"  - {figura}")

    print(
        "\nArchivos auxiliares:"
    )
    print(f"  - {response_path}")
    print(f"  - {resumen_path}")
    print(f"  - {manifest_path}")

    print(
        "\nVariables utilizadas por el modelo:"
    )

    for columna in COLUMNAS_MODELO:
        print(f"  - {columna}")

    print(
        "\nSilueta global:"
    )
    print(f"  - API: {silhouette_api}")
    print(
        f"  - Recalculada: "
        f"{silhouette_media:.6f}"
    )

    if advertencias:
        print(
            "\nAdvertencias:"
        )

        for advertencia in advertencias:
            print(f"  - {advertencia}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        raise