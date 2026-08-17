from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


LABELS_ABC = ("A", "B", "C")


def compute_distance_matrix_qaffas(
    X: pd.DataFrame,
    centers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la distancia euclidiana entre cada observación
    y los representantes de las clases A, B y C.
    """
    missing_centers = [
        label
        for label in LABELS_ABC
        if label not in centers.index
    ]

    if missing_centers:
        raise ValueError(
            "Faltan representantes para las categorías: "
            f"{missing_centers}."
        )

    if list(X.columns) != list(centers.columns):
        raise ValueError(
            "X y centers deben contener las mismas columnas "
            "en el mismo orden."
        )

    x_values = X.to_numpy(dtype=float)

    center_values = (
        centers
        .loc[list(LABELS_ABC)]
        .to_numpy(dtype=float)
    )

    distances = np.sqrt(
        (
            (
                x_values[:, None, :]
                - center_values[None, :, :]
            )
            ** 2
        ).sum(axis=2)
    )

    return pd.DataFrame(
        distances,
        index=X.index,
        columns=LABELS_ABC,
    )


def _validate_capacities(
    n_samples: int,
    capacities: Mapping[str, int],
) -> Dict[str, int]:
    missing_labels = [
        label
        for label in LABELS_ABC
        if label not in capacities
    ]

    if missing_labels:
        raise ValueError(
            "Faltan capacidades para las categorías: "
            f"{missing_labels}."
        )

    normalized = {
        label: int(capacities[label])
        for label in LABELS_ABC
    }

    negative = {
        label: value
        for label, value in normalized.items()
        if value < 0
    }

    if negative:
        raise ValueError(
            "Las capacidades no pueden ser negativas: "
            f"{negative}."
        )

    total_capacity = sum(
        normalized.values()
    )

    if total_capacity != n_samples:
        raise ValueError(
            "La suma de las capacidades debe coincidir con "
            "la cantidad de observaciones. "
            f"Capacidad={total_capacity}, "
            f"observaciones={n_samples}."
        )

    return normalized


def assign_with_capacity_qaffas(
    X: pd.DataFrame,
    centers: pd.DataFrame,
    capacities: Mapping[str, int],
    rng: Optional[np.random.Generator] = None,
    shuffle_unlabeled: bool = False,
) -> pd.Series:
    """
    Asignación restringida basada en el paso 4 del
    Algoritmo 1 de Qaffas et al.

    Cada observación intenta ingresar primero a su centro
    más cercano. Cada clase conserva, como máximo, las
    observaciones más cercanas a su representante.

    Las observaciones rechazadas por una clase completa
    intentan ingresar a su siguiente centro más cercano.

    rng y shuffle_unlabeled se aceptan únicamente para
    mantener compatibilidad con la interfaz del modelo actual.
    Esta implementación es determinista.
    """
    del rng
    del shuffle_unlabeled

    if X.empty:
        raise ValueError(
            "X no puede estar vacío."
        )

    validated_capacities = _validate_capacities(
        n_samples=len(X),
        capacities=capacities,
    )

    distance_matrix = (
        compute_distance_matrix_qaffas(
            X=X,
            centers=centers,
        )
    )

    distance_values = (
        distance_matrix
        .loc[:, list(LABELS_ABC)]
        .to_numpy(dtype=float)
    )

    # Para cada publicación, ordena las clases desde
    # el centro más cercano hasta el más lejano.
    preference_order = np.argsort(
        distance_values,
        axis=1,
        kind="stable",
    )

    n_samples = len(X)

    # Indica cuál preferencia debe intentar cada publicación.
    next_preference = np.zeros(
        n_samples,
        dtype=int,
    )

    # Posiciones aceptadas actualmente por cada clase.
    accepted: dict[str, list[int]] = {
        label: []
        for label in LABELS_ABC
    }

    # Todas las publicaciones comienzan sin asignación.
    pending = list(
        range(n_samples)
    )

    max_rounds = (
        n_samples
        * len(LABELS_ABC)
        + 1
    )

    round_number = 0

    while pending:
        round_number += 1

        if round_number > max_rounds:
            raise RuntimeError(
                "La asignación restringida no pudo finalizar."
            )

        proposals: dict[str, list[int]] = {
            label: []
            for label in LABELS_ABC
        }

        # Cada publicación propone ingresar a su clase
        # disponible más cercana.
        for row_position in pending:
            preference_position = int(
                next_preference[row_position]
            )

            if (
                preference_position
                >= len(LABELS_ABC)
            ):
                raise RuntimeError(
                    "Una observación agotó todas sus "
                    "preferencias de clase."
                )

            cluster_position = int(
                preference_order[
                    row_position,
                    preference_position,
                ]
            )

            label = LABELS_ABC[
                cluster_position
            ]

            proposals[label].append(
                row_position
            )

        rejected: list[int] = []

        # Cada clase conserva las publicaciones más cercanas
        # hasta completar su capacidad.
        for label_position, label in enumerate(
            LABELS_ABC
        ):
            candidates = (
                accepted[label]
                + proposals[label]
            )

            candidates.sort(
                key=lambda row_position: (
                    distance_values[
                        row_position,
                        label_position,
                    ],
                    row_position,
                )
            )

            capacity = (
                validated_capacities[label]
            )

            accepted[label] = (
                candidates[:capacity]
            )

            rejected_candidates = (
                candidates[capacity:]
            )

            # Las publicaciones rechazadas prueban su
            # siguiente clase más cercana.
            for row_position in rejected_candidates:
                next_preference[
                    row_position
                ] += 1

                rejected.append(
                    row_position
                )

        pending = sorted(
            set(rejected)
        )

    assigned_positions = [
        row_position
        for positions in accepted.values()
        for row_position in positions
    ]

    if len(assigned_positions) != n_samples:
        raise RuntimeError(
            "La asignación no cubrió todas "
            "las observaciones."
        )

    if (
        len(set(assigned_positions))
        != n_samples
    ):
        raise RuntimeError(
            "La asignación produjo observaciones duplicadas."
        )

    labels = pd.Series(
        index=X.index,
        dtype="object",
    )

    for label in LABELS_ABC:
        row_positions = accepted[label]

        labels.iloc[
            row_positions
        ] = label

        obtained = len(
            row_positions
        )

        expected = (
            validated_capacities[label]
        )

        if obtained != expected:
            raise RuntimeError(
                "La asignación no respetó la capacidad de "
                f"{label}: esperado={expected}, "
                f"obtenido={obtained}."
            )

    if labels.isna().any():
        raise RuntimeError(
            "La asignación dejó observaciones "
            "sin categoría."
        )

    return labels.astype(str)