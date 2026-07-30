from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


LABELS_ABC = ("A", "B", "C")


def compute_distance_matrix(
    X: pd.DataFrame,
    centers: pd.DataFrame,
) -> pd.DataFrame:
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


def assign_with_capacity(
    X: pd.DataFrame,
    centers: pd.DataFrame,
    capacities: Dict[str, int],
    rng: np.random.Generator,
    shuffle_unlabeled: bool = True,
) -> pd.Series:
    del rng
    del shuffle_unlabeled

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

    return labels.astype(str)