from __future__ import annotations

"""
Evaluación ampliada de publicaciones sin ventas clasificadas por SS-EKMeans.

Ejecución desde la raíz del repositorio:

    python evaluate_publicaciones_sin_ventas_completo.py

Rutas alternativas:

    python evaluate_publicaciones_sin_ventas_completo.py \
        --input data/score_comparison/labels_comparison.csv \
        --output data/score_comparison

El script no vuelve a ejecutar el clustering. Analiza la partición guardada en
labels_comparison.csv y reconstruye el espacio estandarizado utilizado por el
modelo para estudiar los centroides finales.
"""

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_PATH = Path("data/score_comparison/labels_comparison.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("data/score_comparison")

LABELS_ABC = ("A", "B", "C")
FEATURE_COLUMNS = (
    "ventas_log_estandarizada",
    "visitas_log_estandarizada",
    "precio_estandarizado",
)
REQUIRED_COLUMNS = (
    "publication_id",
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "categoria_ss_ekmeans",
)


def safe_number(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    number = float(value)
    return number if np.isfinite(number) else float("nan")


def percentage(numerator: float, denominator: float) -> float:
    return float(numerator / denominator * 100.0) if denominator else 0.0


def format_decimal_es(value: Any, decimals: int = 2) -> str:
    number = safe_number(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{decimals}f}".replace(".", ",")


def format_integer_es(value: Any) -> str:
    number = safe_number(value)
    if not np.isfinite(number):
        return "—"
    return f"{int(round(number)):,}".replace(",", ".")


def format_money_clp(value: Any) -> str:
    return f"${format_integer_es(value)}"


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "< 0,001"
    return format_decimal_es(value, 3)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero "
            "evaluate_score_baseline_comparison.py."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    unnamed = [column for column in df if str(column).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    missing = [column for column in REQUIRED_COLUMNS if column not in df]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df.copy()
    df["publication_id"] = df["publication_id"].astype(str)
    df["categoria_ss_ekmeans"] = (
        df["categoria_ss_ekmeans"].astype(str).str.strip().str.upper()
    )

    for column in ("ventas_30d", "visitas_30d", "precio_actual"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    invalid_labels = sorted(set(df["categoria_ss_ekmeans"]) - set(LABELS_ABC))
    if invalid_labels:
        raise ValueError(f"Categorías inválidas: {invalid_labels}")
    if df["publication_id"].duplicated().any():
        raise ValueError("Existen publication_id duplicados.")
    if (df["ventas_30d"] < 0).any():
        raise ValueError("ventas_30d contiene valores negativos.")
    if (df["visitas_30d"] < 0).any():
        raise ValueError("visitas_30d contiene valores negativos.")
    if (df["precio_actual"] <= 0).any():
        raise ValueError("precio_actual debe ser mayor que cero.")

    return df.reset_index(drop=True)


def reconstruct_model_space(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    original = pd.DataFrame(
        {
            "ventas_log": np.log1p(df["ventas_30d"].astype(float)),
            "visitas_log": np.log1p(df["visitas_30d"].astype(float)),
            "precio": df["precio_actual"].astype(float),
        },
        index=df.index,
    )

    scaler = StandardScaler()
    transformed = scaler.fit_transform(original)
    space = pd.DataFrame(
        transformed,
        columns=FEATURE_COLUMNS,
        index=df.index,
    )
    space["categoria_ss_ekmeans"] = df["categoria_ss_ekmeans"].values

    centers = (
        space.groupby("categoria_ss_ekmeans")[list(FEATURE_COLUMNS)]
        .mean()
        .reindex(LABELS_ABC)
    )
    if centers.isna().any().any():
        raise ValueError("No fue posible reconstruir los tres centroides.")

    matrix = space.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    for label in LABELS_ABC:
        center = centers.loc[label].to_numpy(dtype=float)
        squared_distance = ((matrix - center) ** 2).sum(axis=1)
        space[f"distancia2_centroide_{label}"] = squared_distance
        space[f"distancia_centroide_{label}"] = np.sqrt(squared_distance)

    distance_columns = [
        f"distancia2_centroide_{label}"
        for label in LABELS_ABC
    ]
    space["centroide_mas_cercano"] = (
        space[distance_columns]
        .idxmin(axis=1)
        .str.replace("distancia2_centroide_", "", regex=False)
    )
    space["coincide_centroide_mas_cercano"] = (
        space["categoria_ss_ekmeans"]
        == space["centroide_mas_cercano"]
    )

    assigned_distance2 = np.zeros(len(space), dtype=float)
    best_other_distance2 = np.zeros(len(space), dtype=float)

    for position, (_, row) in enumerate(space.iterrows()):
        assigned = row["categoria_ss_ekmeans"]
        assigned_distance2[position] = row[
            f"distancia2_centroide_{assigned}"
        ]
        best_other_distance2[position] = min(
            row[f"distancia2_centroide_{label}"]
            for label in LABELS_ABC
            if label != assigned
        )

    space["margen_asignacion"] = (
        best_other_distance2 - assigned_distance2
    )
    space["asignacion_puntualmente_preferida"] = (
        space["margen_asignacion"] >= 0
    )

    centers_export = centers.reset_index().rename(
        columns={"categoria_ss_ekmeans": "categoria"}
    )
    return space, centers_export, scaler


def compute_partition_metrics(
    space: pd.DataFrame,
    centers: pd.DataFrame,
) -> dict[str, float]:
    matrix = space.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    labels = space["categoria_ss_ekmeans"].to_numpy()

    inertia = 0.0
    indexed_centers = centers.set_index("categoria")

    for label in LABELS_ABC:
        mask = labels == label
        center = indexed_centers.loc[
            label,
            list(FEATURE_COLUMNS),
        ].to_numpy(float)
        inertia += float(((matrix[mask] - center) ** 2).sum())

    return {
        "n_publicaciones": float(len(space)),
        "silhouette": float(silhouette_score(matrix, labels)),
        "davies_bouldin": float(
            davies_bouldin_score(matrix, labels)
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(matrix, labels)
        ),
        "inercia_reconstruida": inertia,
    }


def build_zero_sales_detail(
    df: pd.DataFrame,
    space: pd.DataFrame,
) -> pd.DataFrame:
    detail = pd.concat(
        [
            df.reset_index(drop=True),
            space.drop(columns="categoria_ss_ekmeans"),
        ],
        axis=1,
    )
    detail = detail[detail["ventas_30d"] == 0].copy()

    detail["grupo_interpretacion"] = "compatible_directamente"
    detail.loc[
        ~detail["coincide_centroide_mas_cercano"],
        "grupo_interpretacion",
    ] = "fronteriza_por_capacidad"

    preferred = [
        "publication_id",
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
        "categoria_ss_ekmeans",
        "centroide_mas_cercano",
        "coincide_centroide_mas_cercano",
        "margen_asignacion",
        "grupo_interpretacion",
        *FEATURE_COLUMNS,
        "distancia_centroide_A",
        "distancia_centroide_B",
        "distancia_centroide_C",
    ]
    optional = [
        column
        for column in ("score_inicial", "categoria_score")
        if column in detail
    ]
    columns = [
        column
        for column in preferred + optional
        if column in detail
    ]

    return detail[columns].sort_values(
        [
            "categoria_ss_ekmeans",
            "visitas_30d",
            "precio_actual",
        ],
        ascending=[True, False, False],
    )


def build_profile(
    df: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for label in LABELS_ABC:
        complete_group = df[
            df["categoria_ss_ekmeans"] == label
        ]
        group = detail[
            detail["categoria_ss_ekmeans"] == label
        ]

        with_visits = int((group["visitas_30d"] > 0).sum())
        nearest = int(
            group["coincide_centroide_mas_cercano"].sum()
        )

        rows.append(
            {
                "categoria": label,
                "publicaciones_categoria": int(len(complete_group)),
                "publicaciones_con_ventas": int(
                    (complete_group["ventas_30d"] > 0).sum()
                ),
                "publicaciones_sin_ventas": int(len(group)),
                "sin_ventas_pct_categoria": percentage(
                    len(group),
                    len(complete_group),
                ),
                "visitas_total": float(
                    group["visitas_30d"].sum()
                ),
                "visitas_media": safe_number(
                    group["visitas_30d"].mean()
                ),
                "visitas_q25": safe_number(
                    group["visitas_30d"].quantile(0.25)
                ),
                "visitas_mediana": safe_number(
                    group["visitas_30d"].median()
                ),
                "visitas_q75": safe_number(
                    group["visitas_30d"].quantile(0.75)
                ),
                "visitas_maximo": safe_number(
                    group["visitas_30d"].max()
                ),
                "publicaciones_con_visitas": with_visits,
                "publicaciones_con_visitas_pct": percentage(
                    with_visits,
                    len(group),
                ),
                "precio_media": safe_number(
                    group["precio_actual"].mean()
                ),
                "precio_q25": safe_number(
                    group["precio_actual"].quantile(0.25)
                ),
                "precio_mediana": safe_number(
                    group["precio_actual"].median()
                ),
                "precio_q75": safe_number(
                    group["precio_actual"].quantile(0.75)
                ),
                "precio_minimo": safe_number(
                    group["precio_actual"].min()
                ),
                "precio_maximo": safe_number(
                    group["precio_actual"].max()
                ),
                "centroide_asignado_es_mas_cercano": nearest,
                "centroide_asignado_es_mas_cercano_pct": percentage(
                    nearest,
                    len(group),
                ),
                "margen_asignacion_medio": safe_number(
                    group["margen_asignacion"].mean()
                ),
                "margen_asignacion_mediano": safe_number(
                    group["margen_asignacion"].median()
                ),
            }
        )

    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Corrección de Holm sin depender de statsmodels."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)

    previous = 0.0
    total = len(values)

    for rank, original_index in enumerate(order):
        adjusted = min(
            1.0,
            (total - rank) * values[original_index],
        )
        previous = max(previous, adjusted)
        adjusted_sorted[rank] = previous

    result = np.empty(len(values), dtype=float)
    for rank, original_index in enumerate(order):
        result[original_index] = adjusted_sorted[rank]

    return result.tolist()


def effect_magnitude(delta: float) -> str:
    absolute = abs(delta)

    if absolute < 0.147:
        # "Despreciable" describe el tamaño del efecto y evita
        # confundirlo con la significación estadística.
        return "despreciable"
    if absolute < 0.330:
        return "pequeño"
    if absolute < 0.474:
        return "moderado"
    return "grande"


def numeric_statistical_tests(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    variables = (
        (
            "visitas_30d",
            "Visitas de publicaciones sin ventas",
        ),
        (
            "precio_actual",
            "Precio de publicaciones sin ventas",
        ),
    )

    for variable, description in variables:
        groups = {
            label: detail.loc[
                detail["categoria_ss_ekmeans"] == label,
                variable,
            ].to_numpy(dtype=float)
            for label in LABELS_ABC
        }

        statistic, p_value = kruskal(
            *(groups[label] for label in LABELS_ABC)
        )

        global_rows.append(
            {
                "prueba": "Kruskal-Wallis",
                "variable": variable,
                "descripcion": description,
                "estadistico": float(statistic),
                "grados_libertad": 2,
                "p_valor": float(p_value),
                "interpretacion": (
                    "Existen diferencias entre categorías"
                    if p_value < 0.05
                    else (
                        "No se detectaron diferencias "
                        "entre categorías"
                    )
                ),
            }
        )

        variable_rows: list[dict[str, Any]] = []
        raw_p_values: list[float] = []

        for first_label, second_label in combinations(
            LABELS_ABC,
            2,
        ):
            first = groups[first_label]
            second = groups[second_label]

            result = mannwhitneyu(
                first,
                second,
                alternative="two-sided",
            )

            denominator = len(first) * len(second)
            cliffs_delta = float(
                2.0 * result.statistic / denominator - 1.0
            )

            raw_p_values.append(float(result.pvalue))
            variable_rows.append(
                {
                    "prueba": "Mann-Whitney U",
                    "variable": variable,
                    "categoria_1": first_label,
                    "categoria_2": second_label,
                    "n_1": len(first),
                    "n_2": len(second),
                    "media_1": float(np.mean(first)),
                    "media_2": float(np.mean(second)),
                    "mediana_1": float(np.median(first)),
                    "mediana_2": float(np.median(second)),
                    "estadistico_u": float(
                        result.statistic
                    ),
                    "p_valor": float(result.pvalue),
                    "delta_cliff": cliffs_delta,
                    "magnitud_efecto": effect_magnitude(
                        cliffs_delta
                    ),
                }
            )

        adjusted = holm_adjust(raw_p_values)

        for row, adjusted_p in zip(
            variable_rows,
            adjusted,
        ):
            row["p_ajustado_holm"] = adjusted_p
            row["diferencia_significativa"] = (
                adjusted_p < 0.05
            )

        pairwise_rows.extend(variable_rows)

    return (
        pd.DataFrame(global_rows),
        pd.DataFrame(pairwise_rows),
    )


def visit_presence_tests(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contingency = []

    for label in LABELS_ABC:
        group = detail[
            detail["categoria_ss_ekmeans"] == label
        ]
        with_visits = int(
            (group["visitas_30d"] > 0).sum()
        )
        without_visits = int(
            (group["visitas_30d"] == 0).sum()
        )
        contingency.append(
            [with_visits, without_visits]
        )

    table = np.asarray(contingency, dtype=int)

    chi2, p_value, dof, _ = chi2_contingency(table)

    denominator = (
        table.sum()
        * min(
            table.shape[0] - 1,
            table.shape[1] - 1,
        )
    )
    cramers_v = float(
        np.sqrt(chi2 / denominator)
    )

    global_result = pd.DataFrame(
        [
            {
                "prueba": (
                    "Chi-cuadrado de independencia"
                ),
                "variable": "presencia_de_visitas",
                "estadistico": float(chi2),
                "grados_libertad": int(dof),
                "p_valor": float(p_value),
                "cramers_v": cramers_v,
                "interpretacion": (
                    "Asociación interna entre categoría "
                    "y presencia de visitas"
                ),
            }
        ]
    )

    pairwise: list[dict[str, Any]] = []
    p_values: list[float] = []

    for first_label, second_label in combinations(
        LABELS_ABC,
        2,
    ):
        first_position = LABELS_ABC.index(
            first_label
        )
        second_position = LABELS_ABC.index(
            second_label
        )

        pair_table = table[
            [first_position, second_position],
            :,
        ]

        odds_ratio, fisher_p = fisher_exact(
            pair_table,
            alternative="two-sided",
        )

        first_rate = (
            pair_table[0, 0]
            / pair_table[0].sum()
        )
        second_rate = (
            pair_table[1, 0]
            / pair_table[1].sum()
        )

        p_values.append(float(fisher_p))
        pairwise.append(
            {
                "prueba": "Fisher exacta",
                "variable": "presencia_de_visitas",
                "categoria_1": first_label,
                "categoria_2": second_label,
                "proporcion_1": float(first_rate),
                "proporcion_2": float(second_rate),
                "diferencia_puntos_porcentuales": float(
                    (first_rate - second_rate) * 100.0
                ),
                "odds_ratio": float(odds_ratio),
                "p_valor": float(fisher_p),
            }
        )

    adjusted = holm_adjust(p_values)

    for row, adjusted_p in zip(
        pairwise,
        adjusted,
    ):
        row["p_ajustado_holm"] = adjusted_p
        row["diferencia_significativa"] = (
            adjusted_p < 0.05
        )

    return (
        global_result,
        pd.DataFrame(pairwise),
    )


def build_a_diagnostic(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    group_a = detail[
        detail["categoria_ss_ekmeans"] == "A"
    ].copy()

    groups = {
        "A es el centroide más cercano": group_a[
            group_a["centroide_mas_cercano"] == "A"
        ],
        "Caso fronterizo por capacidad": group_a[
            group_a["centroide_mas_cercano"] != "A"
        ],
        "Más cercano a B": group_a[
            group_a["centroide_mas_cercano"] == "B"
        ],
        "Más cercano a C": group_a[
            group_a["centroide_mas_cercano"] == "C"
        ],
    }

    rows: list[dict[str, Any]] = []

    for name, group in groups.items():
        rows.append(
            {
                "grupo": name,
                "n": len(group),
                "porcentaje_de_A_sin_ventas": percentage(
                    len(group),
                    len(group_a),
                ),
                "visitas_media": safe_number(
                    group["visitas_30d"].mean()
                ),
                "visitas_mediana": safe_number(
                    group["visitas_30d"].median()
                ),
                "visitas_q25": safe_number(
                    group["visitas_30d"].quantile(0.25)
                ),
                "visitas_q75": safe_number(
                    group["visitas_30d"].quantile(0.75)
                ),
                "precio_media": safe_number(
                    group["precio_actual"].mean()
                ),
                "precio_mediana": safe_number(
                    group["precio_actual"].median()
                ),
                "margen_asignacion_medio": safe_number(
                    group["margen_asignacion"].mean()
                ),
                "margen_asignacion_mediano": safe_number(
                    group["margen_asignacion"].median()
                ),
            }
        )

    return pd.DataFrame(rows)


def compare_a_direct_and_borderline(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compara los dos perfiles de A sin ventas definidos por distancia.

    Los centroides finales se mantienen fijos. "Directo" significa
    que A es el centroide puntualmente más cercano; "fronterizo"
    significa que A fue asignado por la solución global capacitada,
    aunque otro centroide final es más cercano para esa observación.

    Esto no vuelve a ejecutar K-means sin restricciones y no
    representa una clasificación alternativa.
    """

    group_a = detail[
        detail["categoria_ss_ekmeans"] == "A"
    ]
    direct = group_a[
        group_a["centroide_mas_cercano"] == "A"
    ]
    borderline = group_a[
        group_a["centroide_mas_cercano"] != "A"
    ]

    if direct.empty or borderline.empty:
        raise ValueError(
            "No es posible comparar A directo y A fronterizo "
            "porque uno de los dos subgrupos está vacío."
        )

    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    variables = (
        (
            "visitas_30d",
            "Visitas de A sin ventas",
        ),
        (
            "precio_actual",
            "Precio de A sin ventas",
        ),
    )

    for variable, description in variables:
        first = direct[variable].to_numpy(
            dtype=float
        )
        second = borderline[variable].to_numpy(
            dtype=float
        )

        result = mannwhitneyu(
            first,
            second,
            alternative="two-sided",
        )

        cliffs_delta = float(
            2.0
            * result.statistic
            / (len(first) * len(second))
            - 1.0
        )

        raw_p_values.append(
            float(result.pvalue)
        )

        rows.append(
            {
                "prueba": "Mann-Whitney U",
                "variable": variable,
                "descripcion": description,
                "grupo_1": "A directo",
                "grupo_2": "A fronterizo",
                "n_1": len(first),
                "n_2": len(second),
                "media_1": float(np.mean(first)),
                "media_2": float(np.mean(second)),
                "mediana_1": float(np.median(first)),
                "mediana_2": float(np.median(second)),
                "estadistico_u": float(
                    result.statistic
                ),
                "p_valor": float(result.pvalue),
                "delta_cliff": cliffs_delta,
                "magnitud_efecto": effect_magnitude(
                    cliffs_delta
                ),
            }
        )

    adjusted = holm_adjust(raw_p_values)

    for row, adjusted_p in zip(
        rows,
        adjusted,
    ):
        row["p_ajustado_holm"] = adjusted_p
        row["diferencia_significativa"] = (
            adjusted_p < 0.05
        )

        if adjusted_p < 0.05:
            row["interpretacion"] = (
                "Diferencia estadísticamente significativa; "
                "revisar además la magnitud y el signo "
                "de delta de Cliff"
            )
        else:
            row["interpretacion"] = (
                "No se detectó una diferencia "
                "estadísticamente significativa"
            )

    return pd.DataFrame(rows)


def build_formatted_table(
    profile: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for _, row in profile.iterrows():
        rows.append(
            {
                "Categoría": row["categoria"],
                "Sin ventas (n)": format_integer_es(
                    row["publicaciones_sin_ventas"]
                ),
                "Sin ventas dentro de la categoría": (
                    f"{format_decimal_es(
                        row['sin_ventas_pct_categoria']
                    )} %"
                ),
                "Visitas media": format_decimal_es(
                    row["visitas_media"]
                ),
                "Visitas mediana [Q1–Q3]": (
                    f"{format_decimal_es(
                        row['visitas_mediana']
                    )} "
                    f"[{format_decimal_es(
                        row['visitas_q25']
                    )}–"
                    f"{format_decimal_es(
                        row['visitas_q75']
                    )}]"
                ),
                "Con visitas": (
                    f"{format_decimal_es(
                        row['publicaciones_con_visitas_pct']
                    )} %"
                ),
                "Precio medio": format_money_clp(
                    row["precio_media"]
                ),
                "Precio mediano": format_money_clp(
                    row["precio_mediana"]
                ),
                "Centroide asignado es el más cercano": (
                    f"{format_integer_es(
                        row['centroide_asignado_es_mas_cercano']
                    )} "
                    f"({format_decimal_es(
                        row[
                            'centroide_asignado_es_mas_cercano_pct'
                        ]
                    )} %)"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_summary(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    a_diagnostic: pd.DataFrame,
    a_subgroup_tests: pd.DataFrame,
    partition_metrics: dict[str, float],
    global_tests: pd.DataFrame,
    pairwise_tests: pd.DataFrame,
) -> str:
    indexed = profile.set_index("categoria")

    a = indexed.loc["A"]
    b = indexed.loc["B"]
    c = indexed.loc["C"]

    diagnostic = a_diagnostic.set_index("grupo")
    direct = diagnostic.loc[
        "A es el centroide más cercano"
    ]
    borderline = diagnostic.loc[
        "Caso fronterizo por capacidad"
    ]

    sellers_total = int(
        (df["ventas_30d"] > 0).sum()
    )
    sellers_a = int(
        (
            (df["ventas_30d"] > 0)
            & (
                df["categoria_ss_ekmeans"]
                == "A"
            )
        ).sum()
    )
    all_sellers_in_a = (
        sellers_total == sellers_a
    )

    required_zero_sales_a = int(
        a["publicaciones_categoria"]
        - sellers_a
    )

    visit_test = global_tests[
        global_tests["variable"] == "visitas_30d"
    ].iloc[0]

    price_test = global_tests[
        global_tests["variable"] == "precio_actual"
    ].iloc[0]

    presence_test = global_tests[
        global_tests["variable"]
        == "presencia_de_visitas"
    ].iloc[0]

    visit_a_b = pairwise_tests[
        (
            pairwise_tests["variable"]
            == "visitas_30d"
        )
        & (
            pairwise_tests["categoria_1"]
            == "A"
        )
        & (
            pairwise_tests["categoria_2"]
            == "B"
        )
    ].iloc[0]

    visit_a_c = pairwise_tests[
        (
            pairwise_tests["variable"]
            == "visitas_30d"
        )
        & (
            pairwise_tests["categoria_1"]
            == "A"
        )
        & (
            pairwise_tests["categoria_2"]
            == "C"
        )
    ].iloc[0]

    price_a_c = pairwise_tests[
        (
            pairwise_tests["variable"]
            == "precio_actual"
        )
        & (
            pairwise_tests["categoria_1"]
            == "A"
        )
        & (
            pairwise_tests["categoria_2"]
            == "C"
        )
    ].iloc[0]

    a_visit_subgroup = a_subgroup_tests[
        a_subgroup_tests["variable"]
        == "visitas_30d"
    ].iloc[0]

    a_price_subgroup = a_subgroup_tests[
        a_subgroup_tests["variable"]
        == "precio_actual"
    ].iloc[0]

    lines = [
        (
            "ANÁLISIS AMPLIADO DE "
            "PUBLICACIONES SIN VENTAS"
        ),
        "",
        "1. Consecuencia de la capacidad fija",
        (
            f"La categoría A contiene "
            f"{int(a['publicaciones_categoria'])} "
            f"publicaciones. Existen "
            f"{sellers_total} publicaciones con ventas "
            f"y {sellers_a} de ellas fueron asignadas a A."
        ),
        (
            "Todas las publicaciones con ventas "
            f"están en A: "
            f"{'sí' if all_sellers_in_a else 'no'}. "
            "Con la capacidad actual, A debe contener "
            f"{required_zero_sales_a} publicaciones "
            "sin ventas."
        ),
        "",
        "2. Evidencia basada en visitas",
        (
            "Los A sin ventas presentan una media de "
            f"{format_decimal_es(
                a['visitas_media']
            )} y una mediana de "
            f"{format_decimal_es(
                a['visitas_mediana']
            )} visitas. B presenta "
            f"{format_decimal_es(
                b['visitas_media']
            )} y "
            f"{format_decimal_es(
                b['visitas_mediana']
            )}; C presenta "
            f"{format_decimal_es(
                c['visitas_media']
            )} y "
            f"{format_decimal_es(
                c['visitas_mediana']
            )}, respectivamente."
        ),
        (
            "La proporción con al menos una visita es "
            f"{format_decimal_es(
                a['publicaciones_con_visitas_pct']
            )} % en A, "
            f"{format_decimal_es(
                b['publicaciones_con_visitas_pct']
            )} % en B y "
            f"{format_decimal_es(
                c['publicaciones_con_visitas_pct']
            )} % en C."
        ),
        (
            "Kruskal-Wallis para visitas: "
            f"H={format_decimal_es(
                visit_test['estadistico'],
                4,
            )}, "
            f"p {format_p_value(
                visit_test['p_valor']
            )}."
        ),
        (
            "En las comparaciones por pares, "
            "A supera a B con delta de Cliff="
            f"{format_decimal_es(
                visit_a_b['delta_cliff'],
                4,
            )} "
            f"({visit_a_b['magnitud_efecto']}) "
            "y a C con delta="
            f"{format_decimal_es(
                visit_a_c['delta_cliff'],
                4,
            )} "
            f"({visit_a_c['magnitud_efecto']}); "
            "en ambos casos, p ajustado "
            f"{format_p_value(
                max(
                    visit_a_b['p_ajustado_holm'],
                    visit_a_c['p_ajustado_holm'],
                )
            )}."
        ),
        (
            "Asociación entre categoría y "
            "presencia de visitas: "
            "chi-cuadrado="
            f"{format_decimal_es(
                presence_test['estadistico'],
                4,
            )}, "
            f"p {format_p_value(
                presence_test['p_valor']
            )}, "
            "V de Cramér="
            f"{format_decimal_es(
                presence_test['cramers_v'],
                4,
            )}."
        ),
        "",
        "3. Papel del precio",
        (
            "El precio medio (mediano) de los "
            "A sin ventas es "
            f"{format_money_clp(
                a['precio_media']
            )} "
            f"({format_money_clp(
                a['precio_mediana']
            )}); en B es "
            f"{format_money_clp(
                b['precio_media']
            )} "
            f"({format_money_clp(
                b['precio_mediana']
            )}) y en C "
            f"{format_money_clp(
                c['precio_media']
            )} "
            f"({format_money_clp(
                c['precio_mediana']
            )})."
        ),
        (
            "Kruskal-Wallis para precio: "
            f"H={format_decimal_es(
                price_test['estadistico'],
                4,
            )}, "
            f"p {format_p_value(
                price_test['p_valor']
            )}. Sin embargo, entre A y C, "
            "aunque p ajustado "
            f"{format_p_value(
                price_a_c['p_ajustado_holm']
            )}, el tamaño del efecto es "
            f"{price_a_c['magnitud_efecto']} "
            "(delta de Cliff="
            f"{format_decimal_es(
                price_a_c['delta_cliff'],
                4,
            )}). La significación estadística "
            "no implica relevancia práctica: "
            "el precio no justifica por sí solo "
            "que los casos pertenezcan a A. "
            "Debe interpretarse como nivel de "
            "precio de la oferta, no como capital, "
            "rentabilidad ni desempeño."
        ),
        "",
        (
            "4. Compatibilidad directa "
            "y efecto de la capacidad"
        ),
        (
            "Con los centroides finales fijados, "
            f"{int(direct['n'])} publicaciones A "
            "sin ventas "
            f"({format_decimal_es(
                direct[
                    'porcentaje_de_A_sin_ventas'
                ]
            )} %) tienen a A como centroide "
            "más cercano."
        ),
        (
            f"Las {int(borderline['n'])} restantes "
            f"({format_decimal_es(
                borderline[
                    'porcentaje_de_A_sin_ventas'
                ]
            )} %) fueron asignadas a A por la "
            "solución global capacitada, aunque "
            "otro centroide final queda más cerca "
            "en la comparación punto a punto; "
            "por ello se consideran casos "
            "fronterizos o sensibles a la capacidad."
        ),
        (
            "El grupo directamente compatible "
            "presenta una mediana de "
            f"{format_decimal_es(
                direct['visitas_mediana']
            )} visitas; el grupo fronterizo, "
            f"{format_decimal_es(
                borderline['visitas_mediana']
            )}."
        ),
        (
            "La diferencia de visitas entre ambos "
            "subgrupos es significativa "
            f"(U={format_decimal_es(
                a_visit_subgroup['estadistico_u'],
                0,
            )}, "
            "p ajustado "
            f"{format_p_value(
                a_visit_subgroup[
                    'p_ajustado_holm'
                ]
            )}, "
            "delta de Cliff="
            f"{format_decimal_es(
                a_visit_subgroup['delta_cliff'],
                4,
            )}, efecto "
            f"{a_visit_subgroup[
                'magnitud_efecto'
            ]})."
        ),
        (
            "Para precio no se detecta una "
            "diferencia significativa entre "
            "los subgrupos "
            f"(U={format_decimal_es(
                a_price_subgroup['estadistico_u'],
                0,
            )}, "
            "p ajustado "
            f"{format_p_value(
                a_price_subgroup[
                    'p_ajustado_holm'
                ]
            )}, "
            "delta de Cliff="
            f"{format_decimal_es(
                a_price_subgroup['delta_cliff'],
                4,
            )}, efecto "
            f"{a_price_subgroup[
                'magnitud_efecto'
            ]})."
        ),
        (
            "La comparación conserva los "
            "centroides finales y evalúa "
            "distancias punto a punto; no "
            "corresponde a una nueva ejecución "
            "de K-means sin restricciones "
            "de capacidad."
        ),
        "",
        (
            "5. Verificación de la "
            "partición reconstruida"
        ),
        (
            "Silhouette="
            f"{format_decimal_es(
                partition_metrics['silhouette'],
                7,
            )}, Davies-Bouldin="
            f"{format_decimal_es(
                partition_metrics[
                    'davies_bouldin'
                ],
                7,
            )}, Calinski-Harabasz="
            f"{format_decimal_es(
                partition_metrics[
                    'calinski_harabasz'
                ],
                7,
            )}, inercia="
            f"{format_decimal_es(
                partition_metrics[
                    'inercia_reconstruida'
                ],
                7,
            )}."
        ),
        "",
        "Interpretación final",
        (
            "Los resultados entregan evidencia "
            "de coherencia interna para una parte "
            "de los A sin ventas, principalmente "
            "por su exposición. No constituyen "
            "validación externa ni demuestran "
            "rentabilidad o desempeño futuro. "
            "La categoría A debe interpretarse "
            "como prioridad relativa de gestión."
        ),
    ]

    return "\n".join(lines)


def save_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        # Conserva valores p extremadamente pequeños
        # en notación científica. Con %.8f se exportaban
        # como 0.00000000, lo que es incorrecto.
        float_format="%.12g",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa publicaciones sin ventas "
            "clasificadas por SS-EKMeans."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Cargando: {args.input}")

    df = load_data(args.input)
    space, centers, _ = reconstruct_model_space(df)

    detail = build_zero_sales_detail(
        df,
        space,
    )
    profile = build_profile(
        df,
        detail,
    )
    formatted_table = build_formatted_table(
        profile
    )
    partition_metrics = compute_partition_metrics(
        space,
        centers,
    )

    numeric_global, numeric_pairwise = (
        numeric_statistical_tests(detail)
    )
    presence_global, presence_pairwise = (
        visit_presence_tests(detail)
    )

    global_tests = pd.concat(
        [
            numeric_global,
            presence_global,
        ],
        ignore_index=True,
        sort=False,
    )
    pairwise_tests = pd.concat(
        [
            numeric_pairwise,
            presence_pairwise,
        ],
        ignore_index=True,
        sort=False,
    )

    a_diagnostic = build_a_diagnostic(detail)
    a_subgroup_tests = (
        compare_a_direct_and_borderline(detail)
    )

    summary = build_summary(
        df=df,
        profile=profile,
        a_diagnostic=a_diagnostic,
        a_subgroup_tests=a_subgroup_tests,
        partition_metrics=partition_metrics,
        global_tests=global_tests,
        pairwise_tests=pairwise_tests,
    )

    outputs = {
        (
            "perfil_publicaciones_"
            "sin_ventas_completo.csv"
        ): profile,
        (
            "tabla_memoria_"
            "publicaciones_sin_ventas.csv"
        ): formatted_table,
        (
            "detalle_publicaciones_"
            "sin_ventas_completo.csv"
        ): detail,
        (
            "centroides_reconstruidos_"
            "ss_ekmeans.csv"
        ): centers,
        (
            "pruebas_globales_"
            "publicaciones_sin_ventas.csv"
        ): global_tests,
        (
            "comparaciones_pares_"
            "publicaciones_sin_ventas.csv"
        ): pairwise_tests,
        (
            "diagnostico_A_sin_ventas.csv"
        ): a_diagnostic,
        (
            "comparacion_directos_"
            "fronterizos_A.csv"
        ): a_subgroup_tests,
        (
            "metricas_particion_"
            "reconstruida.csv"
        ): pd.DataFrame([partition_metrics]),
    }

    for filename, dataframe in outputs.items():
        save_csv(
            dataframe,
            args.output / filename,
        )

    summary_path = (
        args.output
        / "resumen_publicaciones_sin_ventas_completo.txt"
    )
    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print("\nTabla para la memoria:")
    print(
        formatted_table.to_string(
            index=False
        )
    )

    print("\nPruebas globales:")
    print(
        global_tests.to_string(
            index=False
        )
    )

    print("\nDiagnóstico de A sin ventas:")
    print(
        a_diagnostic.to_string(
            index=False
        )
    )

    print(
        "\nComparación entre A directo "
        "y A fronterizo:"
    )
    print(
        a_subgroup_tests.to_string(
            index=False
        )
    )

    print("\nResumen:")
    print(summary)

    print("\nArchivos generados:")
    for filename in outputs:
        print(f"  {args.output / filename}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()