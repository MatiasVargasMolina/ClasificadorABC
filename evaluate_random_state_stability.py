from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from app.ml.core.config import get_production_config
from app.ml.core.ss_kmeans import SSEKMeans
from app.ml.metrics.clustering_metrics import evaluate_internal_metrics
from app.schemas.input_schema import RequestInput
from app.services.preprocessing_service import ejecutar_preprocesamiento


INPUT_PATH = Path("data/input_request.json")
OUTPUT_DIRECTORY = Path("data/stability_results")

FIXED_SEED = 42
FIXED_SEED_REPETITIONS = 3
STABILITY_SEEDS = list(range(10))


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga el mismo dataset utilizado por la aplicación y ejecuta
    el preprocesamiento oficial del proyecto.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset de entrada: {INPUT_PATH}"
        )

    raw_text = INPUT_PATH.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not raw_text:
        raise ValueError(
            f"El archivo {INPUT_PATH} está vacío."
        )

    try:
        request_payload = json.loads(raw_text)

    except json.JSONDecodeError:
        # También permite procesar contenido copiado desde cURL,
        # por ejemplo: -d '{ ... }'
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")

        if first_brace == -1 or last_brace == -1:
            raise ValueError(
                "No se encontró un objeto JSON válido en "
                f"{INPUT_PATH}. El contenido debe comenzar con '{{' "
                "y terminar con '}}'."
            )

        json_text = raw_text[
            first_brace:last_brace + 1
        ]

        try:
            request_payload = json.loads(json_text)

        except json.JSONDecodeError as error:
            raise ValueError(
                "El archivo contiene un JSON inválido. "
                f"Error en la línea {error.lineno}, "
                f"columna {error.colno}: {error.msg}"
            ) from error

    request_data = RequestInput.model_validate(
        request_payload
    )

    preprocessing_result = ejecutar_preprocesamiento(
        request_data
    )

    if not preprocessing_result["hay_validos"]:
        raise RuntimeError(
            "El dataset no contiene publicaciones válidas "
            "para clasificar."
        )

    X_modelo = (
        preprocessing_result["X_modelo"]
        .copy()
        .reset_index(drop=True)
    )

    df_transformado = (
        preprocessing_result["df_transformado"]
        .copy()
        .reset_index(drop=True)
    )

    publication_ids = (
        df_transformado["publication_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    print(
        "Dataset preparado:",
        len(X_modelo),
        "publicaciones válidas.",
    )

    print(
        "Variables utilizadas:",
        list(X_modelo.columns),
    )

    return X_modelo, publication_ids


def mostrar_configuracion() -> None:
    """
    Muestra la configuración productiva que utilizará la prueba.
    """
    config = get_production_config(
        random_state=FIXED_SEED,
    )

    print("\nConfiguración oficial del modelo:")
    print(f"  Proporciones: {config.proportions}")
    print(f"  max_iter: {config.max_iter}")
    print(f"  tol: {config.tol}")
    print(f"  n_init: {config.n_init}")
    print(f"  random_state de referencia: {config.random_state}")
    print(f"  shuffle_unlabeled: {config.shuffle_unlabeled}")


def ejecutar_corrida(
    X: pd.DataFrame,
    seed: int,
    run_name: str,
) -> tuple[dict[str, Any], pd.Series]:
    """
    Ejecuta una clasificación utilizando la configuración oficial.

    Solamente reemplaza random_state para evaluar estabilidad.
    """
    config = get_production_config(
        random_state=seed,
    )

    model = SSEKMeans(
        config=config,
    )

    start_time = time.perf_counter()

    results = model.fit_predict(X)

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    labels = (
        results["categoria"]
        .astype(str)
        .reset_index(drop=True)
    )

    metrics = evaluate_internal_metrics(
        X,
        labels,
    )

    row = {
        "corrida": run_name,
        "random_state": seed,
        "max_iter": model.max_iter,
        "tolerancia": model.tol,
        "n_init": model.n_init,
        "shuffle_unlabeled": model.shuffle_unlabeled,
        "convergio": model.converged_,
        "motivo_termino": model.stop_reason_,
        "corridas_convergentes": model.converged_runs_,
        "corridas_descartadas": model.discarded_runs_,
        "iteraciones": model.n_iter_,
        "inercia": model.inertia_,
        "silhouette": metrics["silhouette"],
        "davies_bouldin": metrics["davies_bouldin"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "cantidad_A": int((labels == "A").sum()),
        "cantidad_B": int((labels == "B").sum()),
        "cantidad_C": int((labels == "C").sum()),
        "tiempo_segundos": elapsed_seconds,
    }

    return row, labels


def agregar_comparacion(
    row: dict[str, Any],
    labels: pd.Series,
    reference_labels: pd.Series,
) -> None:
    """
    Compara una clasificación contra la corrida de referencia.
    """
    row["coincidencia_referencia_pct"] = float(
        (labels == reference_labels).mean() * 100
    )

    row["publicaciones_diferentes"] = int(
        (labels != reference_labels).sum()
    )

    row["adjusted_rand_index"] = float(
        adjusted_rand_score(
            reference_labels,
            labels,
        )
    )


def main() -> None:
    X, publication_ids = cargar_datos()

    mostrar_configuracion()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_rows: list[dict[str, Any]] = []

    label_columns: dict[str, pd.Series] = {
        "publication_id": publication_ids,
    }

    print(
        "\n1. Prueba de reproducibilidad "
        "con random_state=42"
    )

    reference_row, reference_labels = ejecutar_corrida(
        X=X,
        seed=FIXED_SEED,
        run_name="seed_42_repeticion_1",
    )

    agregar_comparacion(
        row=reference_row,
        labels=reference_labels,
        reference_labels=reference_labels,
    )

    report_rows.append(reference_row)

    label_columns[
        "seed_42_repeticion_1"
    ] = reference_labels

    print(
        "  Repetición 1:",
        f"inercia={reference_row['inercia']:.4f}",
        f"iteraciones={reference_row['iteraciones']}",
        f"convergentes="
        f"{reference_row['corridas_convergentes']}/"
        f"{reference_row['n_init']}",
    )

    for repetition in range(
        2,
        FIXED_SEED_REPETITIONS + 1,
    ):
        run_name = (
            f"seed_42_repeticion_{repetition}"
        )

        row, labels = ejecutar_corrida(
            X=X,
            seed=FIXED_SEED,
            run_name=run_name,
        )

        agregar_comparacion(
            row=row,
            labels=labels,
            reference_labels=reference_labels,
        )

        report_rows.append(row)
        label_columns[run_name] = labels

        print(
            f"  Repetición {repetition}:",
            f"coincidencia="
            f"{row['coincidencia_referencia_pct']:.2f}%",
            f"ARI={row['adjusted_rand_index']:.4f}",
            f"diferentes="
            f"{row['publicaciones_diferentes']}",
        )

    print(
        "\n2. Prueba de estabilidad "
        "con semillas 0–9"
    )

    for seed in STABILITY_SEEDS:
        run_name = f"seed_{seed}"

        try:
            row, labels = ejecutar_corrida(
                X=X,
                seed=seed,
                run_name=run_name,
            )

            agregar_comparacion(
                row=row,
                labels=labels,
                reference_labels=reference_labels,
            )

            report_rows.append(row)
            label_columns[run_name] = labels

            print(
                f"  Semilla {seed}:",
                f"inercia={row['inercia']:.4f}",
                f"coincidencia="
                f"{row['coincidencia_referencia_pct']:.2f}%",
                f"ARI={row['adjusted_rand_index']:.4f}",
                f"diferentes="
                f"{row['publicaciones_diferentes']}",
            )

        except RuntimeError as error:
            report_rows.append(
                {
                    "corrida": run_name,
                    "random_state": seed,
                    "convergio": False,
                    "error": str(error),
                }
            )

            print(
                f"  Semilla {seed}: "
                f"ERROR - {error}"
            )

    report = pd.DataFrame(
        report_rows
    )

    labels_report = pd.DataFrame(
        label_columns
    )

    report_path = (
        OUTPUT_DIRECTORY
        / "random_state_metrics.csv"
    )

    labels_path = (
        OUTPUT_DIRECTORY
        / "random_state_labels.csv"
    )

    report.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    labels_report.to_csv(
        labels_path,
        index=False,
        encoding="utf-8-sig",
    )

    fixed_seed_rows = report[
        report["corrida"].str.startswith(
            "seed_42_repeticion_",
            na=False,
        )
    ]

    reproducible = bool(
        (
            fixed_seed_rows[
                "coincidencia_referencia_pct"
            ] == 100.0
        ).all()
        and (
            fixed_seed_rows[
                "adjusted_rand_index"
            ] == 1.0
        ).all()
    )

    expected_stability_runs = [
        f"seed_{seed}"
        for seed in STABILITY_SEEDS
    ]

    stability_rows = report[
        report["corrida"].isin(
            expected_stability_runs
        )
        & report["convergio"].eq(True)
    ]

    print("\n3. Resumen")

    print(
        "  Reproducibilidad con semilla 42:",
        (
            "CONFIRMADA"
            if reproducible
            else "NO CONFIRMADA"
        ),
    )

    if not stability_rows.empty:
        coincidence_mean = stability_rows[
            "coincidencia_referencia_pct"
        ].mean()

        coincidence_std = stability_rows[
            "coincidencia_referencia_pct"
        ].std(ddof=1)

        ari_mean = stability_rows[
            "adjusted_rand_index"
        ].mean()

        ari_std = stability_rows[
            "adjusted_rand_index"
        ].std(ddof=1)

        inertia_mean = stability_rows[
            "inercia"
        ].mean()

        inertia_std = stability_rows[
            "inercia"
        ].std(ddof=1)

        silhouette_mean = stability_rows[
            "silhouette"
        ].mean()

        silhouette_std = stability_rows[
            "silhouette"
        ].std(ddof=1)

        different_mean = stability_rows[
            "publicaciones_diferentes"
        ].mean()

        print(
            "  Corridas convergentes:",
            f"{len(stability_rows)}/"
            f"{len(STABILITY_SEEDS)}",
        )

        print(
            "  Coincidencia promedio:",
            f"{coincidence_mean:.2f}%",
        )

        print(
            "  Desviación de coincidencia:",
            f"{coincidence_std:.4f}",
        )

        print(
            "  ARI promedio:",
            f"{ari_mean:.4f}",
        )

        print(
            "  Desviación del ARI:",
            f"{ari_std:.4f}",
        )

        print(
            "  Publicaciones diferentes promedio:",
            f"{different_mean:.2f}",
        )

        print(
            "  Inercia promedio:",
            f"{inertia_mean:.4f}",
        )

        print(
            "  Desviación de inercia:",
            f"{inertia_std:.4f}",
        )

        print(
            "  Silhouette promedio:",
            f"{silhouette_mean:.4f}",
        )

        print(
            "  Desviación de Silhouette:",
            f"{silhouette_std:.4f}",
        )

    print("\nArchivos generados:")
    print(f"  {report_path}")
    print(f"  {labels_path}")


if __name__ == "__main__":
    main()