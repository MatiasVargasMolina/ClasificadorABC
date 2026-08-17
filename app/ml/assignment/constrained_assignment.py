from typing import Dict, Literal

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


LABELS_ABC = ("A", "B", "C")

MetodoAsignacion = Literal["global", "secuencial"]


def compute_distance_matrix(
    X: pd.DataFrame,
    centers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula las distancias euclidianas al cuadrado entre cada
    observación y los centroides A, B y C.
    """
    x = X.to_numpy(dtype=float)
    c = centers.loc[list(LABELS_ABC)].to_numpy(dtype=float)

    squared_distances = (
        (x[:, None, :] - c[None, :, :]) ** 2
    ).sum(axis=2)

    return pd.DataFrame(
        squared_distances,
        index=X.index,
        columns=LABELS_ABC,
    )


def _validate_capacities(
    X: pd.DataFrame,
    capacities: Dict[str, int],
) -> None:
    missing_labels = [
        label
        for label in LABELS_ABC
        if label not in capacities
    ]

    if missing_labels:
        raise ValueError(
            "No se encontró capacidad para las categorías: "
            f"{missing_labels}."
        )

    negative_capacities = {
        label: capacities[label]
        for label in LABELS_ABC
        if capacities[label] < 0
    }

    if negative_capacities:
        raise ValueError(
            "Las capacidades no pueden ser negativas: "
            f"{negative_capacities}."
        )

    total_capacity = sum(
        capacities[label]
        for label in LABELS_ABC
    )

    if total_capacity != len(X):
        raise ValueError(
            "La suma de las capacidades debe coincidir con "
            f"la cantidad de observaciones. Capacidad={total_capacity}, "
            f"observaciones={len(X)}."
        )


def _validate_assignment_result(
    labels: pd.Series,
    capacities: Dict[str, int],
) -> None:
    if labels.isna().any():
        raise RuntimeError(
            "La asignación produjo observaciones sin categoría."
        )

    counts = labels.value_counts().to_dict()

    for label in LABELS_ABC:
        if counts.get(label, 0) != capacities[label]:
            raise RuntimeError(
                "La asignación no respetó la capacidad de "
                f"{label}: esperado={capacities[label]}, "
                f"obtenido={counts.get(label, 0)}."
            )


def assign_with_capacity_global(
    X: pd.DataFrame,
    centers: pd.DataFrame,
    capacities: Dict[str, int],
) -> pd.Series:
    """
    Asignación global con capacidades.

    Construye un problema de asignación global y minimiza
    conjuntamente el costo de distancia mediante
    scipy.optimize.linear_sum_assignment.
    """
    _validate_capacities(
        X=X,
        capacities=capacities,
    )

    distance_matrix = compute_distance_matrix(
        X=X,
        centers=centers,
    )

    slot_labels: list[str] = []

    for label in LABELS_ABC:
        capacity = capacities[label]
        slot_labels.extend([label] * capacity)

    slot_positions = np.array(
        [
            LABELS_ABC.index(label)
            for label in slot_labels
        ],
        dtype=int,
    )

    cost_matrix = (
        distance_matrix
        .loc[:, list(LABELS_ABC)]
        .to_numpy(dtype=float)[:, slot_positions]
    )

    row_indices, column_indices = linear_sum_assignment(
        cost_matrix
    )

    if len(row_indices) != len(X):
        raise RuntimeError(
            "No fue posible asignar todas las observaciones "
            "respetando las capacidades."
        )

    labels = pd.Series(
        index=X.index,
        dtype="object",
    )

    for row_position, column_position in zip(
        row_indices,
        column_indices,
    ):
        labels.iloc[row_position] = slot_labels[column_position]

    _validate_assignment_result(
        labels=labels,
        capacities=capacities,
    )

    return labels.astype(str)


def assign_with_capacity_sequential(
    X: pd.DataFrame,
    centers: pd.DataFrame,
    capacities: Dict[str, int],
    rng: np.random.Generator,
    shuffle_unlabeled: bool = True,
) -> pd.Series:
    """
    Asignación secuencial con capacidades.

    Prioriza las observaciones cuya asignación es más clara y
    las procesa individualmente, asignándolas al centroide
    disponible más cercano.
    """
    _validate_capacities(
        X=X,
        capacities=capacities,
    )

    # La implementación histórica secuencial trabajaba con
    # distancia euclidiana, no con distancia al cuadrado.
    distance_matrix = np.sqrt(
        compute_distance_matrix(
            X=X,
            centers=centers,
        )
    )

    if shuffle_unlabeled:
        shuffled_idx = np.array(
            distance_matrix.index.to_list(),
            dtype=object,
        )

        rng.shuffle(shuffled_idx)

        distance_matrix = distance_matrix.loc[
            list(shuffled_idx)
        ]

    distances = distance_matrix.to_numpy(dtype=float)

    ranked = np.sort(
        distances,
        axis=1,
    )

    best = ranked[:, 0]

    second = (
        ranked[:, 1]
        if ranked.shape[1] > 1
        else ranked[:, 0]
    )

    priority = pd.DataFrame(
        {
            "best": best,
            "confidence": second - best,
        },
        index=distance_matrix.index,
    ).sort_values(
        by=["confidence", "best"],
        ascending=[False, True],
        kind="mergesort",
    )

    labels = pd.Series(
        index=X.index,
        dtype="object",
    )

    counts = {
        label: 0
        for label in LABELS_ABC
    }

    for idx in priority.index:
        sorted_labels = (
            distance_matrix
            .loc[idx]
            .sort_values(kind="mergesort")
            .index
            .tolist()
        )

        assigned = False

        for label in sorted_labels:
            if counts[label] < capacities[label]:
                labels.at[idx] = label
                counts[label] += 1
                assigned = True
                break

        if not assigned:
            raise RuntimeError(
                "No fue posible asignar una observación "
                "respetando las capacidades."
            )

    _validate_assignment_result(
        labels=labels,
        capacities=capacities,
    )

    return labels.astype(str)


def assign_with_capacity(
    X: pd.DataFrame,
    centers: pd.DataFrame,
    capacities: Dict[str, int],
    rng: np.random.Generator,
    shuffle_unlabeled: bool = True,
    metodo_asignacion: MetodoAsignacion = "global",
) -> pd.Series:
    """
    Selecciona el método de asignación utilizado por SS-EKMeans.

    global:
        Minimización conjunta mediante linear_sum_assignment.

    secuencial:
        Asignación publicación por publicación respetando capacidades.
    """
    if metodo_asignacion == "global":
        return assign_with_capacity_global(
            X=X,
            centers=centers,
            capacities=capacities,
        )

    if metodo_asignacion == "secuencial":
        return assign_with_capacity_sequential(
            X=X,
            centers=centers,
            capacities=capacities,
            rng=rng,
            shuffle_unlabeled=shuffle_unlabeled,
        )

    raise ValueError(
        "metodo_asignacion debe ser 'global' o 'secuencial'. "
        f"Valor recibido: {metodo_asignacion!r}."
    )