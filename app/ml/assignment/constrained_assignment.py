from typing import Dict

import numpy as np
import pandas as pd

LABELS_ABC = ("A", "B", "C")


def compute_distance_matrix(
    X: pd.DataFrame,
    centers: pd.DataFrame,
) -> pd.DataFrame:
    x = X.to_numpy(dtype=float)
    c = centers.to_numpy(dtype=float)

    distances = np.sqrt(
        ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    )

    return pd.DataFrame(
        distances,
        index=X.index,
        columns=centers.index,
    )


def assign_with_capacity(
    X: pd.DataFrame,
    centers: pd.DataFrame,
    capacities: Dict[str, int],
    rng: np.random.Generator,
    shuffle_unlabeled: bool = True,
) -> pd.Series:
    labels = pd.Series(index=X.index, dtype="object")
    counts = {label: 0 for label in LABELS_ABC}

    # 1. Calcular distancias de todos los productos a cada centroide
    dist_df = compute_distance_matrix(
        X=X,
        centers=centers,
    )

    # 2. Barajar para que n_init pueda generar corridas distintas
    if shuffle_unlabeled:
        shuffled_idx = np.array(X.index.to_list(), dtype=object)
        rng.shuffle(shuffled_idx)
        dist_df = dist_df.loc[list(shuffled_idx)]

    # 3. Priorizar puntos con asignación más clara
    distances = dist_df.to_numpy(dtype=float)
    ranked = np.sort(distances, axis=1)

    best = ranked[:, 0]
    second = ranked[:, 1] if ranked.shape[1] > 1 else ranked[:, 0]

    priority = pd.DataFrame(
        {
            "best": best,
            "confidence": second - best,
        },
        index=dist_df.index,
    ).sort_values(
        by=["confidence", "best"],
        ascending=[False, True],
    )

    # 4. Asignar respetando capacidades ABC
    for idx in priority.index:
        sorted_labels = (
            dist_df
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
                "No fue posible asignar una observación respetando capacidades."
            )

    return labels.astype(str)