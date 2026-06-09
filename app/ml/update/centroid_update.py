import numpy as np
import pandas as pd

LABELS_ABC = ("A", "B", "C")


def recompute_centroids(
    X: pd.DataFrame,
    labels: pd.Series,
    previous_centers: pd.DataFrame,
) -> pd.DataFrame:
    new_centers = {}

    for label in LABELS_ABC:
        members = X.loc[labels == label]

        if members.empty:
            new_centers[label] = previous_centers.loc[label].to_numpy(dtype=float)
        else:
            new_centers[label] = members.mean(axis=0).to_numpy(dtype=float)

    return (
        pd.DataFrame.from_dict(
            new_centers,
            orient="index",
            columns=X.columns,
        )
        .loc[list(LABELS_ABC)]
    )


def compute_center_shift(
    old_centers: pd.DataFrame,
    new_centers: pd.DataFrame,
) -> float:
    shift = (
        (old_centers.to_numpy(dtype=float) - new_centers.to_numpy(dtype=float)) ** 2
    ).sum(axis=1)

    return float(np.sqrt(shift).max())