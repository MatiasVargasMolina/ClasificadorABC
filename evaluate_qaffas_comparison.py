from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

import app.ml.core.ss_kmeans as ss_kmeans_module

from app.ml.assignment.constrained_assignment import (
    assign_with_capacity as assign_with_capacity_current,
)
from app.ml.assignment.qaffas_assignment import (
    assign_with_capacity_qaffas,
)
from app.ml.core.config import get_production_config
from app.ml.core.qaffas_ss_kmeans import (
    QaffasSSEKMeans,
)
from app.ml.core.ss_kmeans import SSEKMeans
from app.ml.metrics.clustering_metrics import (
    compute_inertia,
    evaluate_internal_metrics,
)
from app.schemas.input_schema import RequestInput
from app.services.preprocessing_service import (
    ejecutar_preprocesamiento,
)


INPUT_PATH = Path(
    "data/input_request.json"
)

OUTPUT_DIRECTORY = Path(
    "data/qaffas_comparison"
)

LABELS_ABC = (
    "A",
    "B",
    "C",
)

AssignmentFunction = Callable[
    ...,
    pd.Series,
]


def cargar_request() -> RequestInput:
    """
    Carga el mismo archivo JSON utilizado por la aplicación.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {INPUT_PATH}"
        )

    raw_text = INPUT_PATH.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not raw_text:
        raise ValueError(
            f"El archivo {INPUT_PATH} está vacío."
        )

    try:
        payload = json.loads(
            raw_text
        )

    except json.JSONDecodeError:
        first_brace = raw_text.find(
            "{"
        )

        last_brace = raw_text.rfind(
            "}"
        )

        if (
            first_brace == -1
            or last_brace == -1
        ):
            raise ValueError(
                "No se encontró un objeto JSON válido."
            )

        json_text = raw_text[
            first_brace:last_brace + 1
        ]

        try:
            payload = json.loads(
                json_text
            )

        except json.JSONDecodeError as error:
            raise ValueError(
                "El archivo contiene un JSON inválido. "
                f"Línea {error.lineno}, "
                f"columna {error.colno}: "
                f"{error.msg}"
            ) from error

    return RequestInput.model_validate(
        payload
    )


def preparar_datos() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    """
    Reutiliza la validación y transformación del proyecto.

    Retorna:

    X_original:
        Ventas, visitas y precio sin escalar.

    X_standard:
        Preprocesamiento productivo actual.

    X_minmax:
        Min-Max aplicado directamente sobre las variables
        originales, sin transformación log1p.

    publication_ids:
        Identificadores de las publicaciones.
    """
    request_data = cargar_request()

    preprocessing_result = (
        ejecutar_preprocesamiento(
            request_data
        )
    )

    if not preprocessing_result[
        "hay_validos"
    ]:
        raise RuntimeError(
            "No existen publicaciones válidas."
        )

    X_original = (
        preprocessing_result["X"]
        .copy()
        .astype(float)
        .reset_index(drop=True)
    )

    X_standard = (
        preprocessing_result["X_modelo"]
        .copy()
        .astype(float)
        .reset_index(drop=True)
    )

    df_transformado = (
        preprocessing_result[
            "df_transformado"
        ]
        .copy()
        .reset_index(drop=True)
    )

    publication_ids = (
        df_transformado[
            "publication_id"
        ]
        .astype(str)
        .reset_index(drop=True)
    )

    minmax_scaler = MinMaxScaler()

    X_minmax = pd.DataFrame(
        minmax_scaler.fit_transform(
            X_original
        ),
        columns=X_original.columns,
        index=X_original.index,
    )

    # Corrige posibles desviaciones de punto flotante,
    # por ejemplo 1.0000000000000002.
    X_minmax = X_minmax.clip(
        lower=0.0,
        upper=1.0,
    )

    return (
        X_original,
        X_standard,
        X_minmax,
        publication_ids,
    )


def ejecutar_ssekmeans_actual(
    nombre: str,
    X: pd.DataFrame,
    tipo_preprocesamiento: str,
    tipo_asignacion: str,
    preprocessing_name: str,
    assignment_name: str,
    assignment_function: AssignmentFunction,
) -> tuple[
    dict[str, Any],
    pd.Series,
]:
    """
    Ejecuta la implementación actual de SSEKMeans.

    Permite reemplazar temporalmente solo la función de
    asignación, conservando:

    - la inicialización actual;
    - las proporciones;
    - max_iter;
    - tolerancia;
    - n_init;
    - random_state.
    """
    config = get_production_config()

    original_assignment = (
        ss_kmeans_module
        .assign_with_capacity
    )

    try:
        ss_kmeans_module.assign_with_capacity = (
            assignment_function
        )

        model = SSEKMeans(
            config=config
        )

        start_time = time.perf_counter()

        results = model.fit_predict(
            X
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

    finally:
        ss_kmeans_module.assign_with_capacity = (
            original_assignment
        )

    labels = (
        results["categoria"]
        .astype(str)
        .reset_index(drop=True)
    )

    diagnostics = {
        "variante": nombre,
        "tipo_evaluacion": (
            "diseno_factorial_2x2"
        ),
        "factor_preprocesamiento": (
            tipo_preprocesamiento
        ),
        "factor_asignacion": (
            tipo_asignacion
        ),
        "preprocesamiento": (
            preprocessing_name
        ),
        "inicializacion": (
            "inicialización actual por zonas "
            "alta, media y baja"
        ),
        "asignacion": assignment_name,
        "n_init": model.n_init,
        "random_state": model.random_state,
        "max_iter": model.max_iter,
        "tol": model.tol,
        "convergio": model.converged_,
        "motivo_termino": (
            model.stop_reason_
        ),
        "iteraciones": model.n_iter_,
        "mejor_iteracion": (
            model.n_iter_
        ),
        "inercia_nativa": (
            model.inertia_
        ),
        "corridas_convergentes": (
            model.converged_runs_
        ),
        "corridas_descartadas": (
            model.discarded_runs_
        ),
        "tiempo_segundos": (
            elapsed_seconds
        ),
        "cantidad_A": int(
            (labels == "A").sum()
        ),
        "cantidad_B": int(
            (labels == "B").sum()
        ),
        "cantidad_C": int(
            (labels == "C").sum()
        ),
    }

    return (
        diagnostics,
        labels,
    )


def ejecutar_qaffas_completo(
    X_minmax: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    pd.Series,
]:
    """
    Ejecuta el método completo basado en Qaffas:

    - normalización Min-Max;
    - score Ω;
    - A = máximo Ω;
    - B = mediana de Ω;
    - C = mínimo Ω;
    - asignación por preferencias de distancia;
    - una inicialización determinista.

    Esta ejecución no pertenece al diseño factorial 2x2,
    porque también cambia la inicialización y n_init.
    """
    config = get_production_config()

    model = QaffasSSEKMeans(
        proportions=config.proportions,
        max_iter=config.max_iter,
        tol=config.tol,
    )

    start_time = time.perf_counter()

    results = model.fit_predict(
        X_minmax
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    labels = (
        results["categoria"]
        .astype(str)
        .reset_index(drop=True)
    )

    diagnostics = {
        "variante": (
            "qaffas_completo"
        ),
        "tipo_evaluacion": (
            "comparacion_metodos_completos"
        ),
        "factor_preprocesamiento": (
            "minmax"
        ),
        "factor_asignacion": (
            "qaffas"
        ),
        "preprocesamiento": (
            "MinMaxScaler [0,1] sin log1p"
        ),
        "inicializacion": (
            "A=máximo Ω, "
            "B=mediana de Ω, "
            "C=mínimo Ω"
        ),
        "asignacion": (
            "preferencias por distancia "
            "con capacidad máxima"
        ),
        "n_init": 1,
        "random_state": None,
        "max_iter": model.max_iter,
        "tol": model.tol,
        "convergio": (
            model.converged_
        ),
        "motivo_termino": (
            model.stop_reason_
        ),
        "iteraciones": (
            model.n_iter_
        ),
        "mejor_iteracion": (
            model.best_iteration_
        ),
        "inercia_nativa": (
            model.inertia_
        ),
        "corridas_convergentes": (
            1
            if model.converged_
            else 0
        ),
        "corridas_descartadas": 0,
        "tiempo_segundos": (
            elapsed_seconds
        ),
        "cantidad_A": int(
            (labels == "A").sum()
        ),
        "cantidad_B": int(
            (labels == "B").sum()
        ),
        "cantidad_C": int(
            (labels == "C").sum()
        ),
    }

    return (
        diagnostics,
        labels,
    )


def calcular_centros(
    X: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """
    Recalcula los centroides de una clasificación.

    Se utiliza para calcular la inercia de todas las variantes
    sobre el mismo espacio Min-Max.
    """
    centers: dict[
        str,
        Any,
    ] = {}

    for label in LABELS_ABC:
        members = X.loc[
            labels == label
        ]

        if members.empty:
            raise RuntimeError(
                f"La categoría {label} está vacía."
            )

        centers[label] = (
            members
            .mean(axis=0)
            .to_numpy(dtype=float)
        )

    return pd.DataFrame.from_dict(
        centers,
        orient="index",
        columns=X.columns,
    ).loc[
        list(LABELS_ABC)
    ]


def calcular_concentracion(
    X_original: pd.DataFrame,
    labels: pd.Series,
    column: str,
) -> dict[str, float]:
    """
    Calcula el porcentaje total de una variable que queda
    contenido en cada categoría.
    """
    total = float(
        X_original[column].sum()
    )

    if total == 0:
        return {
            label: 0.0
            for label in LABELS_ABC
        }

    return {
        label: float(
            X_original.loc[
                labels == label,
                column,
            ].sum()
            / total
            * 100
        )
        for label in LABELS_ABC
    }


def construir_resumen(
    diagnostics: dict[str, Any],
    X_native: pd.DataFrame,
    X_common: pd.DataFrame,
    X_original: pd.DataFrame,
    labels: pd.Series,
    reference_labels: pd.Series,
) -> dict[str, Any]:
    """
    Calcula las métricas de cada variante.

    Métricas nativas:
        Calculadas sobre el espacio usado para entrenar.

    Métricas comunes:
        Calculadas sobre X_minmax para comparar las
        clasificaciones utilizando una misma representación.
    """
    native_metrics = (
        evaluate_internal_metrics(
            X_native,
            labels,
        )
    )

    common_metrics = (
        evaluate_internal_metrics(
            X_common,
            labels,
        )
    )

    common_centers = calcular_centros(
        X=X_common,
        labels=labels,
    )

    common_inertia = compute_inertia(
        X=X_common,
        labels=labels,
        centers=common_centers,
    )

    sales_concentration = (
        calcular_concentracion(
            X_original=X_original,
            labels=labels,
            column="ventas_30d",
        )
    )

    visits_concentration = (
        calcular_concentracion(
            X_original=X_original,
            labels=labels,
            column="visitas_30d",
        )
    )

    exact_agreement = float(
        (
            labels
            == reference_labels
        ).mean()
        * 100
    )

    ari = float(
        adjusted_rand_score(
            reference_labels,
            labels,
        )
    )

    changed_publications = int(
        (
            labels
            != reference_labels
        ).sum()
    )

    return {
        **diagnostics,
        "silhouette_nativo": (
            native_metrics[
                "silhouette"
            ]
        ),
        "davies_bouldin_nativo": (
            native_metrics[
                "davies_bouldin"
            ]
        ),
        "calinski_harabasz_nativo": (
            native_metrics[
                "calinski_harabasz"
            ]
        ),
        "inercia_espacio_comun_minmax": (
            common_inertia
        ),
        "silhouette_espacio_comun_minmax": (
            common_metrics[
                "silhouette"
            ]
        ),
        "davies_bouldin_espacio_comun_minmax": (
            common_metrics[
                "davies_bouldin"
            ]
        ),
        "calinski_harabasz_espacio_comun_minmax": (
            common_metrics[
                "calinski_harabasz"
            ]
        ),
        "coincidencia_vs_actual_pct": (
            exact_agreement
        ),
        "adjusted_rand_index_vs_actual": (
            ari
        ),
        "publicaciones_diferentes_vs_actual": (
            changed_publications
        ),
        "ventas_clase_A_pct": (
            sales_concentration["A"]
        ),
        "ventas_clase_B_pct": (
            sales_concentration["B"]
        ),
        "ventas_clase_C_pct": (
            sales_concentration["C"]
        ),
        "visitas_clase_A_pct": (
            visits_concentration["A"]
        ),
        "visitas_clase_B_pct": (
            visits_concentration["B"]
        ),
        "visitas_clase_C_pct": (
            visits_concentration["C"]
        ),
    }


def construir_perfiles_clase(
    X_original: pd.DataFrame,
    labels: pd.Series,
    variant_name: str,
) -> pd.DataFrame:
    """
    Construye estadísticas descriptivas para A, B y C.
    """
    data = X_original.copy()

    data["categoria"] = (
        labels.to_numpy()
    )

    rows: list[
        dict[str, Any]
    ] = []

    for label in LABELS_ABC:
        subset = data.loc[
            data["categoria"] == label
        ]

        row: dict[
            str,
            Any,
        ] = {
            "variante": variant_name,
            "categoria": label,
            "cantidad": len(subset),
        }

        for column in X_original.columns:
            values = (
                subset[column]
                .astype(float)
            )

            row[
                f"{column}_suma"
            ] = float(
                values.sum()
            )

            row[
                f"{column}_media"
            ] = float(
                values.mean()
            )

            row[
                f"{column}_mediana"
            ] = float(
                values.median()
            )

            row[
                f"{column}_minimo"
            ] = float(
                values.min()
            )

            row[
                f"{column}_maximo"
            ] = float(
                values.max()
            )

        row[
            "ventas_cero_pct"
        ] = float(
            (
                subset[
                    "ventas_30d"
                ]
                == 0
            ).mean()
            * 100
        )

        row[
            "visitas_cero_pct"
        ] = float(
            (
                subset[
                    "visitas_30d"
                ]
                == 0
            ).mean()
            * 100
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def construir_transiciones(
    reference_labels: pd.Series,
    variant_labels: pd.Series,
    variant_name: str,
) -> pd.DataFrame:
    """
    Cuenta los cambios de categoría respecto del modelo
    StandardScaler + asignación global.
    """
    rows: list[
        dict[str, Any]
    ] = []

    for current_label in LABELS_ABC:
        for variant_label in LABELS_ABC:
            count = int(
                (
                    (
                        reference_labels
                        == current_label
                    )
                    & (
                        variant_labels
                        == variant_label
                    )
                ).sum()
            )

            rows.append(
                {
                    "variante": (
                        variant_name
                    ),
                    "categoria_actual": (
                        current_label
                    ),
                    "categoria_variante": (
                        variant_label
                    ),
                    "cantidad": count,
                }
            )

    return pd.DataFrame(
        rows
    )


def obtener_valor(
    row: pd.Series,
    column: str,
) -> Optional[float]:
    """
    Obtiene un valor numérico desde una fila del resumen.
    """
    value = row.get(
        column
    )

    if pd.isna(
        value
    ):
        return None

    return float(
        value
    )


def calcular_delta(
    base_value: Optional[float],
    compared_value: Optional[float],
) -> Optional[float]:
    """
    Calcula variante comparada menos variante base.
    """
    if (
        base_value is None
        or compared_value is None
    ):
        return None

    return float(
        compared_value
        - base_value
    )


def construir_comparacion_factorial(
    summary_report: pd.DataFrame,
    labels_by_variant: dict[
        str,
        pd.Series,
    ],
    comparison_name: str,
    factor_evaluado: str,
    factor_fijo: str,
    base_variant: str,
    compared_variant: str,
) -> dict[str, Any]:
    """
    Construye una comparación de dos variantes donde solamente
    cambia uno de los factores.
    """
    indexed_summary = (
        summary_report
        .set_index("variante")
    )

    if (
        base_variant
        not in indexed_summary.index
    ):
        raise ValueError(
            f"No existe la variante base: {base_variant}"
        )

    if (
        compared_variant
        not in indexed_summary.index
    ):
        raise ValueError(
            "No existe la variante comparada: "
            f"{compared_variant}"
        )

    base_row = indexed_summary.loc[
        base_variant
    ]

    compared_row = indexed_summary.loc[
        compared_variant
    ]

    base_labels = labels_by_variant[
        base_variant
    ]

    compared_labels = labels_by_variant[
        compared_variant
    ]

    exact_agreement = float(
        (
            base_labels
            == compared_labels
        ).mean()
        * 100
    )

    ari = float(
        adjusted_rand_score(
            base_labels,
            compared_labels,
        )
    )

    changed_publications = int(
        (
            base_labels
            != compared_labels
        ).sum()
    )

    metric_columns = [
        "tiempo_segundos",
        "inercia_espacio_comun_minmax",
        "silhouette_espacio_comun_minmax",
        "davies_bouldin_espacio_comun_minmax",
        "calinski_harabasz_espacio_comun_minmax",
        "ventas_clase_A_pct",
        "ventas_clase_B_pct",
        "ventas_clase_C_pct",
        "visitas_clase_A_pct",
        "visitas_clase_B_pct",
        "visitas_clase_C_pct",
    ]

    result: dict[
        str,
        Any,
    ] = {
        "comparacion": comparison_name,
        "factor_evaluado": factor_evaluado,
        "factor_fijo": factor_fijo,
        "variante_base": base_variant,
        "variante_comparada": (
            compared_variant
        ),
        "coincidencia_entre_variantes_pct": (
            exact_agreement
        ),
        "adjusted_rand_index": ari,
        "publicaciones_diferentes": (
            changed_publications
        ),
    }

    for column in metric_columns:
        base_value = obtener_valor(
            base_row,
            column,
        )

        compared_value = obtener_valor(
            compared_row,
            column,
        )

        result[
            f"{column}_base"
        ] = base_value

        result[
            f"{column}_comparada"
        ] = compared_value

        result[
            f"delta_{column}"
        ] = calcular_delta(
            base_value,
            compared_value,
        )

    return result


def main() -> None:
    (
        X_original,
        X_standard,
        X_minmax,
        publication_ids,
    ) = preparar_datos()

    print(
        "Dataset preparado:",
        len(X_original),
        "publicaciones válidas.",
    )

    print(
        "Variables utilizadas:",
        list(X_original.columns),
    )

    factorial_specs = [
        {
            "nombre": (
                "standard_hungarian"
            ),
            "X": X_standard,
            "tipo_preprocesamiento": (
                "standard"
            ),
            "tipo_asignacion": (
                "hungarian"
            ),
            "preprocesamiento": (
                "log1p + StandardScaler"
            ),
            "asignacion": (
                "linear_sum_assignment"
            ),
            "assignment_function": (
                assign_with_capacity_current
            ),
        },
        {
            "nombre": (
                "standard_qaffas_assignment"
            ),
            "X": X_standard,
            "tipo_preprocesamiento": (
                "standard"
            ),
            "tipo_asignacion": (
                "qaffas"
            ),
            "preprocesamiento": (
                "log1p + StandardScaler"
            ),
            "asignacion": (
                "preferencias por distancia "
                "con capacidad máxima"
            ),
            "assignment_function": (
                assign_with_capacity_qaffas
            ),
        },
        {
            "nombre": (
                "minmax_hungarian"
            ),
            "X": X_minmax,
            "tipo_preprocesamiento": (
                "minmax"
            ),
            "tipo_asignacion": (
                "hungarian"
            ),
            "preprocesamiento": (
                "MinMaxScaler [0,1] sin log1p"
            ),
            "asignacion": (
                "linear_sum_assignment"
            ),
            "assignment_function": (
                assign_with_capacity_current
            ),
        },
        {
            "nombre": (
                "minmax_qaffas_assignment"
            ),
            "X": X_minmax,
            "tipo_preprocesamiento": (
                "minmax"
            ),
            "tipo_asignacion": (
                "qaffas"
            ),
            "preprocesamiento": (
                "MinMaxScaler [0,1] sin log1p"
            ),
            "asignacion": (
                "preferencias por distancia "
                "con capacidad máxima"
            ),
            "assignment_function": (
                assign_with_capacity_qaffas
            ),
        },
    ]

    execution_results: list[
        dict[str, Any]
    ] = []

    print(
        "\n1. Diseño factorial 2x2"
    )

    for spec in factorial_specs:
        variant_name = spec[
            "nombre"
        ]

        print(
            f"\nEjecutando: {variant_name}"
        )

        try:
            diagnostics, labels = (
                ejecutar_ssekmeans_actual(
                    nombre=variant_name,
                    X=spec["X"],
                    tipo_preprocesamiento=(
                        spec[
                            "tipo_preprocesamiento"
                        ]
                    ),
                    tipo_asignacion=(
                        spec[
                            "tipo_asignacion"
                        ]
                    ),
                    preprocessing_name=(
                        spec[
                            "preprocesamiento"
                        ]
                    ),
                    assignment_name=(
                        spec[
                            "asignacion"
                        ]
                    ),
                    assignment_function=(
                        spec[
                            "assignment_function"
                        ]
                    ),
                )
            )

            execution_results.append(
                {
                    "diagnostics": (
                        diagnostics
                    ),
                    "labels": labels,
                    "X_native": spec["X"],
                }
            )

            print(
                "  Convergió:",
                diagnostics[
                    "convergio"
                ],
            )

            print(
                "  Iteraciones:",
                diagnostics[
                    "iteraciones"
                ],
            )

            print(
                "  Tiempo:",
                f"{diagnostics['tiempo_segundos']:.4f} s",
            )

        except Exception as error:
            print(
                f"  ERROR: {error}"
            )

            raise

    print(
        "\n2. Comparación del método completo"
    )

    qaffas_diagnostics, qaffas_labels = (
        ejecutar_qaffas_completo(
            X_minmax=X_minmax,
        )
    )

    execution_results.append(
        {
            "diagnostics": (
                qaffas_diagnostics
            ),
            "labels": (
                qaffas_labels
            ),
            "X_native": (
                X_minmax
            ),
        }
    )

    print(
        "  Qaffas completo convergió:",
        qaffas_diagnostics[
            "convergio"
        ],
    )

    print(
        "  Iteraciones:",
        qaffas_diagnostics[
            "iteraciones"
        ],
    )

    reference_name = (
        "standard_hungarian"
    )

    reference_result = next(
        result
        for result in execution_results
        if result[
            "diagnostics"
        ][
            "variante"
        ] == reference_name
    )

    reference_labels = (
        reference_result[
            "labels"
        ]
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    labels_report = pd.DataFrame(
        {
            "publication_id": (
                publication_ids
            ),
        }
    )

    profile_frames: list[
        pd.DataFrame
    ] = []

    transition_frames: list[
        pd.DataFrame
    ] = []

    labels_by_variant: dict[
        str,
        pd.Series,
    ] = {}

    for result in execution_results:
        diagnostics = result[
            "diagnostics"
        ]

        labels = result[
            "labels"
        ]

        variant_name = diagnostics[
            "variante"
        ]

        labels_by_variant[
            variant_name
        ] = labels

        labels_report[
            f"categoria_{variant_name}"
        ] = labels

        summary_rows.append(
            construir_resumen(
                diagnostics=diagnostics,
                X_native=result[
                    "X_native"
                ],
                X_common=X_minmax,
                X_original=X_original,
                labels=labels,
                reference_labels=(
                    reference_labels
                ),
            )
        )

        profile_frames.append(
            construir_perfiles_clase(
                X_original=X_original,
                labels=labels,
                variant_name=(
                    variant_name
                ),
            )
        )

        transition_frames.append(
            construir_transiciones(
                reference_labels=(
                    reference_labels
                ),
                variant_labels=labels,
                variant_name=(
                    variant_name
                ),
            )
        )

    summary_report = pd.DataFrame(
        summary_rows
    )

    profiles_report = pd.concat(
        profile_frames,
        ignore_index=True,
    )

    transitions_report = pd.concat(
        transition_frames,
        ignore_index=True,
    )

    factorial_summary = (
        summary_report.loc[
            summary_report[
                "tipo_evaluacion"
            ]
            == "diseno_factorial_2x2"
        ]
        .copy()
        .reset_index(drop=True)
    )

    complete_methods_summary = (
        summary_report.loc[
            summary_report[
                "variante"
            ].isin(
                [
                    "standard_hungarian",
                    "qaffas_completo",
                ]
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    factor_comparisons = pd.DataFrame(
        [
            # Efecto del preprocesamiento con
            # asignación global fija.
            construir_comparacion_factorial(
                summary_report=(
                    factorial_summary
                ),
                labels_by_variant=(
                    labels_by_variant
                ),
                comparison_name=(
                    "preprocesamiento_con_hungarian"
                ),
                factor_evaluado=(
                    "preprocesamiento"
                ),
                factor_fijo=(
                    "asignacion=hungarian"
                ),
                base_variant=(
                    "standard_hungarian"
                ),
                compared_variant=(
                    "minmax_hungarian"
                ),
            ),

            # Efecto del preprocesamiento con
            # asignación Qaffas fija.
            construir_comparacion_factorial(
                summary_report=(
                    factorial_summary
                ),
                labels_by_variant=(
                    labels_by_variant
                ),
                comparison_name=(
                    "preprocesamiento_con_qaffas"
                ),
                factor_evaluado=(
                    "preprocesamiento"
                ),
                factor_fijo=(
                    "asignacion=qaffas"
                ),
                base_variant=(
                    "standard_qaffas_assignment"
                ),
                compared_variant=(
                    "minmax_qaffas_assignment"
                ),
            ),

            # Efecto de la asignación con
            # StandardScaler fijo.
            construir_comparacion_factorial(
                summary_report=(
                    factorial_summary
                ),
                labels_by_variant=(
                    labels_by_variant
                ),
                comparison_name=(
                    "asignacion_con_standard"
                ),
                factor_evaluado=(
                    "asignacion"
                ),
                factor_fijo=(
                    "preprocesamiento=standard"
                ),
                base_variant=(
                    "standard_hungarian"
                ),
                compared_variant=(
                    "standard_qaffas_assignment"
                ),
            ),

            # Efecto de la asignación con
            # Min-Max fijo.
            construir_comparacion_factorial(
                summary_report=(
                    factorial_summary
                ),
                labels_by_variant=(
                    labels_by_variant
                ),
                comparison_name=(
                    "asignacion_con_minmax"
                ),
                factor_evaluado=(
                    "asignacion"
                ),
                factor_fijo=(
                    "preprocesamiento=minmax"
                ),
                base_variant=(
                    "minmax_hungarian"
                ),
                compared_variant=(
                    "minmax_qaffas_assignment"
                ),
            ),
        ]
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / "comparison_summary.csv"
    )

    factorial_summary_path = (
        OUTPUT_DIRECTORY
        / "factorial_summary.csv"
    )

    factor_comparisons_path = (
        OUTPUT_DIRECTORY
        / "factor_comparisons.csv"
    )

    complete_methods_path = (
        OUTPUT_DIRECTORY
        / "complete_methods_summary.csv"
    )

    labels_path = (
        OUTPUT_DIRECTORY
        / "comparison_labels.csv"
    )

    profiles_path = (
        OUTPUT_DIRECTORY
        / "comparison_class_profiles.csv"
    )

    transitions_path = (
        OUTPUT_DIRECTORY
        / "comparison_transitions.csv"
    )

    summary_report.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    factorial_summary.to_csv(
        factorial_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    factor_comparisons.to_csv(
        factor_comparisons_path,
        index=False,
        encoding="utf-8-sig",
    )

    complete_methods_summary.to_csv(
        complete_methods_path,
        index=False,
        encoding="utf-8-sig",
    )

    labels_report.to_csv(
        labels_path,
        index=False,
        encoding="utf-8-sig",
    )

    profiles_report.to_csv(
        profiles_path,
        index=False,
        encoding="utf-8-sig",
    )

    transitions_report.to_csv(
        transitions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n3. Resumen factorial"
    )

    factorial_columns = [
        "variante",
        "factor_preprocesamiento",
        "factor_asignacion",
        "tiempo_segundos",
        "silhouette_espacio_comun_minmax",
        "davies_bouldin_espacio_comun_minmax",
        "calinski_harabasz_espacio_comun_minmax",
        "ventas_clase_A_pct",
        "visitas_clase_A_pct",
    ]

    print(
        factorial_summary[
            factorial_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n4. Comparaciones por factor"
    )

    factor_columns = [
        "comparacion",
        "factor_evaluado",
        "factor_fijo",
        "variante_base",
        "variante_comparada",
        "delta_tiempo_segundos",
        "delta_silhouette_espacio_comun_minmax",
        "delta_davies_bouldin_espacio_comun_minmax",
        "delta_calinski_harabasz_espacio_comun_minmax",
        "delta_ventas_clase_A_pct",
        "delta_visitas_clase_A_pct",
        "publicaciones_diferentes",
    ]

    print(
        factor_comparisons[
            factor_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nArchivos generados:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {factorial_summary_path}"
    )

    print(
        f"  {factor_comparisons_path}"
    )

    print(
        f"  {complete_methods_path}"
    )

    print(
        f"  {labels_path}"
    )

    print(
        f"  {profiles_path}"
    )

    print(
        f"  {transitions_path}"
    )


if __name__ == "__main__":
    main()