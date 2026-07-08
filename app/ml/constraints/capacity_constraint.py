from typing import Dict, Mapping, Sequence

import numpy as np

LABELS_ABC = ("A", "B", "C")


def compute_capacities(
    n_samples: int,
    proportions: Mapping[str, float],
    labels: Sequence[str] = LABELS_ABC,
) -> Dict[str, int]:
    if n_samples <= 0:
        raise ValueError("n_samples debe ser mayor que 0.")

    missing = [label for label in labels if label not in proportions]
    if missing:
        raise ValueError(f"Faltan proporciones para: {missing}")

    total = sum(float(proportions[label]) for label in labels)

    if total <= 0:
        raise ValueError("La suma de proporciones debe ser positiva.")

    normalized = {
        label: float(proportions[label]) / total
        for label in labels
    }

    raw = {
        label: normalized[label] * n_samples
        for label in labels
    }

    capacities = {
        label: int(np.floor(raw[label]))
        for label in labels
    }

    remainder = n_samples - sum(capacities.values())

    if remainder > 0:
        ordered = sorted(
            labels,
            key=lambda label: raw[label] - capacities[label],
            reverse=True,
        )

        for label in ordered[:remainder]:
            capacities[label] += 1

    return capacities