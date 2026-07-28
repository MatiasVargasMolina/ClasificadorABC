from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from app.schemas.input_schema import RequestInput
from app.services.clasificacion_service import ejecutar_clasificacion


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "input_request.json"
DEFAULT_OUTPUT = ROOT / "data" / "bootstrap_results"

LABELS = ("A", "B", "C")
N_BOOTSTRAP = 100
BOOTSTRAP_SEED = 2026
CONFIDENCE_LEVEL = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validación bootstrap del clasificador ABC usando "
            "el pipeline vigente del proyecto."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    if args.n_bootstrap < 2:
        parser.error("--n-bootstrap debe ser mayor o igual a 2.")

    return args


def read_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"El archivo está vacío: {path}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Permite usar un cuerpo JSON copiado desde un comando cURL.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No se encontró un objeto JSON en {path}.")

        try:
            payload = json.loads(raw[start:end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(
                "JSON inválido: "
                f"línea {error.lineno}, columna {error.colno}: {error.msg}"
            ) from error

    if not isinstance(payload, dict):
        raise ValueError("La raíz del JSON debe ser un objeto.")

    return payload


def run_pipeline(request: RequestInput) -> dict[str, Any]:
    """
    Ejecuta exactamente el mismo servicio usado por la API.

    Este archivo no implementa validación, transformación,
    escalamiento, SSEKMeans ni métricas internas.
    """
    response = ejecutar_clasificacion(request)

    if not response.get("resultados"):
        raise RuntimeError(
            response.get("mensaje", "La clasificación no generó resultados.")
        )
    if "diagnostico" not in response:
        raise RuntimeError("La respuesta no contiene diagnóstico.")

    return response


def prepare_reference(
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    request = RequestInput.model_validate(payload)
    response = run_pipeline(request)

    reference = pd.DataFrame(response["resultados"])
    required = {"publication_id", "categoria"}
    missing = required - set(reference.columns)
    if missing:
        raise RuntimeError(
            f"Faltan columnas en la clasificación: {sorted(missing)}"
        )

    reference = reference[["publication_id", "categoria"]].copy()
    reference["publication_id"] = reference["publication_id"].astype(str)
    reference["categoria"] = reference["categoria"].astype(str)

    if reference["publication_id"].duplicated().any():
        raise RuntimeError(
            "La clasificación de referencia contiene IDs duplicados."
        )

    original_products = {
        str(product.publication_id): product.model_dump()
        for product in request.productos
    }
    valid_products = {}

    for publication_id in reference["publication_id"]:
        if publication_id not in original_products:
            raise RuntimeError(
                f"No se encontró el producto original {publication_id}."
            )
        valid_products[publication_id] = original_products[publication_id]

    return reference, valid_products, response["diagnostico"]


def make_bootstrap_request(
    bootstrap_id: int,
    sampled_ids: np.ndarray,
    valid_products: dict[str, dict[str, Any]],
) -> tuple[RequestInput, dict[str, str]]:
    products = []
    synthetic_to_original = {}

    for position, original_id in enumerate(sampled_ids):
        original_id = str(original_id)
        product = dict(valid_products[original_id])

        # Bootstrap produce copias. RequestInput exige IDs únicos,
        # por eso se usa un ID temporal que no entra al modelo.
        synthetic_id = f"bs_{bootstrap_id:04d}_{position:06d}"
        product["publication_id"] = synthetic_id

        products.append(product)
        synthetic_to_original[synthetic_id] = original_id

    request = RequestInput.model_validate({"productos": products})
    return request, synthetic_to_original


def unique_mode(values: pd.Series) -> Optional[str]:
    counts = values.value_counts()
    if counts.empty:
        return None

    maximum = int(counts.iloc[0])
    winners = counts[counts == maximum].index.tolist()
    return str(winners[0]) if len(winners) == 1 else None


def consolidate_duplicates(
    results: list[dict[str, Any]],
    synthetic_to_original: dict[str, str],
) -> tuple[pd.DataFrame, int]:
    frame = pd.DataFrame(results)
    frame["publication_id"] = frame["publication_id"].astype(str)
    frame["categoria"] = frame["categoria"].astype(str)
    frame["original_id"] = frame["publication_id"].map(
        synthetic_to_original
    )

    if frame["original_id"].isna().any():
        raise RuntimeError(
            "No fue posible relacionar todos los resultados con sus IDs."
        )

    consolidated = (
        frame.groupby("original_id", sort=False)["categoria"]
        .apply(unique_mode)
        .rename("categoria_bootstrap")
        .reset_index()
        .rename(columns={"original_id": "publication_id"})
    )

    ambiguous = int(consolidated["categoria_bootstrap"].isna().sum())
    consolidated = consolidated.dropna(
        subset=["categoria_bootstrap"]
    ).copy()
    consolidated["categoria_bootstrap"] = (
        consolidated["categoria_bootstrap"].astype(str)
    )

    return consolidated, ambiguous


def jaccard(
    reference: pd.Series,
    candidate: pd.Series,
    label: str,
) -> Optional[float]:
    reference_mask = reference.to_numpy() == label
    candidate_mask = candidate.to_numpy() == label
    union = np.logical_or(reference_mask, candidate_mask).sum()

    if union == 0:
        return None

    intersection = np.logical_and(
        reference_mask, candidate_mask
    ).sum()
    return float(intersection / union)


def metric(diagnostic: dict[str, Any], name: str) -> Optional[float]:
    value = diagnostic.get("metricas", {}).get(name)
    return None if value is None else float(value)


def run_iteration(
    bootstrap_id: int,
    sample_seed: int,
    reference: pd.DataFrame,
    valid_products: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, set[str]]:
    all_ids = reference["publication_id"].to_numpy(dtype=str)
    n = len(all_ids)
    rng = np.random.default_rng(sample_seed)
    sampled_ids = rng.choice(all_ids, size=n, replace=True)
    included_ids = set(sampled_ids.tolist())

    request, id_map = make_bootstrap_request(
        bootstrap_id, sampled_ids, valid_products
    )

    started = time.perf_counter()
    response = run_pipeline(request)
    elapsed = time.perf_counter() - started

    consolidated, ambiguous = consolidate_duplicates(
        response["resultados"], id_map
    )
    comparison = reference.merge(
        consolidated,
        on="publication_id",
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) < 2:
        raise RuntimeError(
            "La repetición no produjo suficientes observaciones comparables."
        )

    reference_labels = comparison["categoria"].astype(str)
    bootstrap_labels = comparison["categoria_bootstrap"].astype(str)
    diagnostic = response["diagnostico"]
    counts = diagnostic.get("conteos_finales", {})

    row = {
        "bootstrap_id": bootstrap_id,
        "sample_seed": sample_seed,
        "n_total": n,
        "n_unicos_muestra": len(included_ids),
        "cobertura_muestra": len(included_ids) / n,
        "n_fuera_muestra": n - len(included_ids),
        "proporcion_fuera_muestra": 1.0 - len(included_ids) / n,
        "n_asignaciones_ambiguas": ambiguous,
        "n_comparables": len(comparison),
        "proporcion_comparable": len(comparison) / n,
        "iteraciones": diagnostic.get("iteraciones"),
        "inercia": diagnostic.get("inertia"),
        "silhouette": metric(diagnostic, "silhouette"),
        "davies_bouldin": metric(diagnostic, "davies_bouldin"),
        "calinski_harabasz": metric(
            diagnostic, "calinski_harabasz"
        ),
        "ari_referencia": float(
            adjusted_rand_score(reference_labels, bootstrap_labels)
        ),
        "coincidencia_referencia": float(
            (
                reference_labels.to_numpy()
                == bootstrap_labels.to_numpy()
            ).mean()
        ),
        "jaccard_A": jaccard(
            reference_labels, bootstrap_labels, "A"
        ),
        "jaccard_B": jaccard(
            reference_labels, bootstrap_labels, "B"
        ),
        "jaccard_C": jaccard(
            reference_labels, bootstrap_labels, "C"
        ),
        "cantidad_A": int(counts.get("A", 0)),
        "cantidad_B": int(counts.get("B", 0)),
        "cantidad_C": int(counts.get("C", 0)),
        "tiempo_segundos": elapsed,
        "error": None,
    }

    assignments = comparison[
        ["publication_id", "categoria_bootstrap"]
    ].copy()
    return row, assignments, included_ids


def describe_metric(values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)

    if values.empty:
        return {
            "n_validos": 0,
            "media": None,
            "desviacion_estandar": None,
            "mediana": None,
            "ic_inferior": None,
            "ic_superior": None,
            "minimo": None,
            "maximo": None,
        }

    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)

    return {
        "n_validos": int(len(values)),
        "media": float(values.mean()),
        "desviacion_estandar": (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ),
        "mediana": float(values.median()),
        "ic_inferior": float(np.percentile(values, lower)),
        "ic_superior": float(np.percentile(values, upper)),
        "minimo": float(values.min()),
        "maximo": float(values.max()),
    }


def build_summary(
    runs: pd.DataFrame,
    reference_diagnostic: dict[str, Any],
) -> pd.DataFrame:
    reference_metrics = reference_diagnostic.get("metricas", {})
    metrics = {
        "silhouette": reference_metrics.get("silhouette"),
        "davies_bouldin": reference_metrics.get("davies_bouldin"),
        "calinski_harabasz": reference_metrics.get(
            "calinski_harabasz"
        ),
        "inercia": reference_diagnostic.get("inertia"),
        "ari_referencia": 1.0,
        "coincidencia_referencia": 1.0,
        "jaccard_A": 1.0,
        "jaccard_B": 1.0,
        "jaccard_C": 1.0,
        "cobertura_muestra": None,
        "proporcion_comparable": None,
        "iteraciones": reference_diagnostic.get("iteraciones"),
        "tiempo_segundos": None,
    }

    rows = []
    for name, reference_value in metrics.items():
        rows.append(
            {
                "metrica": name,
                "valor_referencia": reference_value,
                **describe_metric(runs[name]),
            }
        )

    return pd.DataFrame(rows)


def build_publication_stability(
    reference: pd.DataFrame,
    counters: pd.DataFrame,
) -> pd.DataFrame:
    stability = (
        reference.set_index("publication_id")
        .join(counters, how="left")
        .fillna(0)
    )

    for column in [
        "veces_incluida",
        "veces_comparable",
        "veces_A",
        "veces_B",
        "veces_C",
    ]:
        stability[column] = stability[column].astype(int)

    stability["veces_ambigua"] = (
        stability["veces_incluida"] - stability["veces_comparable"]
    )

    def publication_row(row: pd.Series) -> pd.Series:
        counts = {label: int(row[f"veces_{label}"]) for label in LABELS}
        comparable = int(row["veces_comparable"])
        reference_label = str(row["categoria"])

        if comparable == 0:
            return pd.Series(
                {
                    "categoria_modal_bootstrap": None,
                    "probabilidad_categoria_modal": None,
                    "estabilidad_categoria_referencia": None,
                }
            )

        modal = max(
            LABELS,
            key=lambda label: (counts[label], -LABELS.index(label)),
        )
        return pd.Series(
            {
                "categoria_modal_bootstrap": modal,
                "probabilidad_categoria_modal": (
                    counts[modal] / comparable
                ),
                "estabilidad_categoria_referencia": (
                    counts[reference_label] / comparable
                ),
            }
        )

    calculated = stability.apply(publication_row, axis=1)
    stability = stability.join(calculated)
    stability = stability.rename(
        columns={"categoria": "categoria_referencia"}
    )
    return stability.reset_index()


def generate_figure(
    runs: pd.DataFrame,
    publication_stability: pd.DataFrame,
    reference_diagnostic: dict[str, Any],
    output_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nNo se generó el gráfico porque matplotlib no está "
            "instalado. Instálalo con:\n"
            "python -m pip install matplotlib"
        )
        return False

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle(
        "Validación bootstrap de la clasificación ABC",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].hist(
        runs["silhouette"].dropna(),
        bins=15,
        color="0.65",
        edgecolor="black",
    )
    reference_silhouette = reference_diagnostic.get(
        "metricas", {}
    ).get("silhouette")
    if reference_silhouette is not None:
        axes[0, 0].axvline(
            float(reference_silhouette),
            color="black",
            linestyle="--",
            label="Referencia",
        )
        axes[0, 0].legend()
    axes[0, 0].set_title("Distribución de Silhouette")
    axes[0, 0].set_xlabel("Coeficiente Silhouette")
    axes[0, 0].set_ylabel("Frecuencia")

    axes[0, 1].hist(
        runs["ari_referencia"].dropna(),
        bins=15,
        color="0.45",
        edgecolor="black",
    )
    axes[0, 1].set_title("Estabilidad de la partición")
    axes[0, 1].set_xlabel("Índice Rand ajustado (ARI)")
    axes[0, 1].set_ylabel("Frecuencia")

    axes[1, 0].hist(
        runs["coincidencia_referencia"].dropna(),
        bins=15,
        color="0.75",
        edgecolor="black",
    )
    axes[1, 0].set_title("Coincidencia de categorías")
    axes[1, 0].set_xlabel("Proporción de coincidencia")
    axes[1, 0].set_ylabel("Frecuencia")

    distributions = [
        publication_stability.loc[
            publication_stability["categoria_referencia"] == label,
            "estabilidad_categoria_referencia",
        ].dropna()
        for label in LABELS
    ]
    axes[1, 1].boxplot(
        distributions,
        tick_labels=LABELS,
        patch_artist=True,
        boxprops={"facecolor": "0.80", "edgecolor": "black"},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    axes[1, 1].set_title("Estabilidad individual por categoría")
    axes[1, 1].set_xlabel("Categoría de referencia")
    axes[1, 1].set_ylabel("Proporción de coincidencia")
    axes[1, 1].set_ylim(0.0, 1.05)

    for axis in axes.flat:
        axis.grid(alpha=0.25, linestyle=":")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return True


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Ejecutando la clasificación de referencia...")
    payload = read_payload(input_path)
    reference, valid_products, reference_diagnostic = prepare_reference(
        payload
    )

    publication_ids = reference["publication_id"].tolist()
    n = len(publication_ids)
    counters = pd.DataFrame(
        0,
        index=pd.Index(publication_ids, name="publication_id"),
        columns=[
            "veces_incluida",
            "veces_comparable",
            "veces_A",
            "veces_B",
            "veces_C",
        ],
        dtype=int,
    )

    print(f"Publicaciones válidas: {n}")
    print(
        "Conteos de referencia:",
        reference_diagnostic.get("conteos_finales"),
    )
    print(
        "Silhouette de referencia:",
        reference_diagnostic.get("metricas", {}).get("silhouette"),
    )
    print(
        f"\nEjecutando {args.n_bootstrap} remuestreos con "
        "ejecutar_clasificacion()..."
    )

    rows = []
    started = time.perf_counter()

    for bootstrap_id in range(1, args.n_bootstrap + 1):
        sample_seed = args.bootstrap_seed + bootstrap_id - 1

        try:
            row, assignments, included_ids = run_iteration(
                bootstrap_id,
                sample_seed,
                reference,
                valid_products,
            )
            rows.append(row)

            counters.loc[
                list(included_ids), "veces_incluida"
            ] += 1

            for assignment in assignments.itertuples(index=False):
                publication_id = str(assignment.publication_id)
                category = str(assignment.categoria_bootstrap)
                counters.loc[
                    publication_id, "veces_comparable"
                ] += 1
                counters.loc[
                    publication_id, f"veces_{category}"
                ] += 1

            print(
                f"[{bootstrap_id:03d}/{args.n_bootstrap:03d}] "
                f"ARI={row['ari_referencia']:.4f} | "
                f"coincidencia={row['coincidencia_referencia']:.4f} | "
                f"Silhouette={row['silhouette']:.4f} | "
                f"comparables={row['n_comparables']}"
            )

        except (RuntimeError, ValueError) as error:
            rows.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "sample_seed": sample_seed,
                    "error": str(error),
                }
            )
            print(
                f"[{bootstrap_id:03d}/{args.n_bootstrap:03d}] "
                f"ERROR: {error}"
            )

    elapsed = time.perf_counter() - started
    runs = pd.DataFrame(rows)

    if "error" not in runs.columns:
        runs["error"] = None

    successful = runs[runs["error"].isna()].copy()
    if successful.empty:
        raise RuntimeError(
            "Ningún remuestreo bootstrap terminó correctamente."
        )

    summary = build_summary(successful, reference_diagnostic)
    publication_stability = build_publication_stability(
        reference, counters
    )

    metrics_path = output_dir / "bootstrap_metrics.csv"
    summary_path = output_dir / "bootstrap_summary.csv"
    publications_path = (
        output_dir / "bootstrap_publication_stability.csv"
    )
    json_path = output_dir / "bootstrap_summary.json"
    figure_path = output_dir / "bootstrap_validation.png"

    runs.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    publication_stability.to_csv(
        publications_path, index=False, encoding="utf-8-sig"
    )

    figure_generated = generate_figure(
        successful,
        publication_stability,
        reference_diagnostic,
        figure_path,
    )

    threshold = 0.80
    unstable = int(
        (
            publication_stability["estabilidad_categoria_referencia"]
            < threshold
        ).sum()
    )

    report = {
        "metodo": (
            "Bootstrap no paramétrico con remuestreo de publicaciones "
            "con reemplazo."
        ),
        "reutilizacion_pipeline": {
            "servicio": (
                "app.services.clasificacion_service."
                "ejecutar_clasificacion"
            ),
            "detalle": (
                "Cada repetición usa el mismo pipeline de validación, "
                "preprocesamiento, escalamiento, SSEKMeans y métricas "
                "empleado por la API."
            ),
        },
        "nota_interpretacion": (
            "Los intervalos describen estabilidad interna, no exactitud "
            "frente a etiquetas reales. Las observaciones no incluidas "
            "en una muestra no se clasifican porque el pipeline actual "
            "no expone predict para datos externos al ajuste."
        ),
        "configuracion": {
            "n_bootstrap_solicitados": args.n_bootstrap,
            "n_bootstrap_exitosos": len(successful),
            "n_bootstrap_fallidos": args.n_bootstrap - len(successful),
            "semilla_bootstrap": args.bootstrap_seed,
            "nivel_confianza": CONFIDENCE_LEVEL,
            "configuracion_modelo": (
                "Definida exclusivamente en "
                "app/services/kmeans_service.py."
            ),
        },
        "dataset": {"publicaciones_validas": n},
        "modelo_referencia": reference_diagnostic,
        "resumen_metricas": summary.to_dict(orient="records"),
        "estabilidad_publicaciones": {
            "umbral": threshold,
            "cantidad_bajo_umbral": unstable,
            "proporcion_bajo_umbral": unstable / n,
            "media": float(
                publication_stability[
                    "estabilidad_categoria_referencia"
                ].mean()
            ),
            "mediana": float(
                publication_stability[
                    "estabilidad_categoria_referencia"
                ].median()
            ),
        },
        "ejecucion": {
            "segundos_totales": elapsed,
            "segundos_promedio": elapsed / len(successful),
        },
        "archivos": {
            "metricas": metrics_path,
            "resumen": summary_path,
            "estabilidad_publicaciones": publications_path,
            "figura": figure_path if figure_generated else None,
        },
    }
    json_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nResumen")
    print(f"  Corridas correctas: {len(successful)}/{args.n_bootstrap}")
    print(
        f"  Cobertura media: "
        f"{successful['cobertura_muestra'].mean():.4f}"
    )
    print(f"  ARI medio: {successful['ari_referencia'].mean():.4f}")
    print(
        f"  Coincidencia media: "
        f"{successful['coincidencia_referencia'].mean():.4f}"
    )
    print(
        f"  Silhouette medio: {successful['silhouette'].mean():.4f}"
    )
    print(f"  Publicaciones con estabilidad < 0.80: {unstable}/{n}")
    print(f"  Tiempo total: {elapsed:.2f} segundos")

    print("\nArchivos generados:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {publications_path}")
    print(f"  {json_path}")
    if figure_generated:
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()
