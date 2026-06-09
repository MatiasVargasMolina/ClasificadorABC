from typing import Dict, Tuple
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
    seed_labels: pd.Series,
    rng: np.random.Generator,
    shuffle_unlabeled: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    labels = pd.Series(index=X.index, dtype="object")
    is_seed = pd.Series(False, index=X.index, dtype=bool)

    counts = {label: 0 for label in LABELS_ABC}

    # 1. Fijar semillas
    for label in LABELS_ABC:
        mask = seed_labels == label

        if mask.any():
            labels.loc[mask] = label
            is_seed.loc[mask] = True
            counts[label] += int(mask.sum())

    unlabeled_idx = labels[labels.isna()].index.to_list()

    if not unlabeled_idx:
        return labels.astype(str), is_seed

    # 2. Calcular distancias de no etiquetados
    dist_df = compute_distance_matrix(
        X.loc[unlabeled_idx],
        centers,
    )

    # 3. Barajar para que n_init tenga efecto
    if shuffle_unlabeled:
        shuffled_idx = np.array(unlabeled_idx, dtype=object)
        rng.shuffle(shuffled_idx)
        dist_df = dist_df.loc[list(shuffled_idx)]

    # 4. Priorizar puntos con asignación más clara
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

    # 5. Asignar respetando capacidad
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

    return labels.astype(str), is_seed