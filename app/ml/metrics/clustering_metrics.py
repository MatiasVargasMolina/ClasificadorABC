from typing import Dict, Optional
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


LABELS_ABC = ("A", "B", "C")


def compute_inertia(
    X: pd.DataFrame,
    labels: pd.Series,
    centers: pd.DataFrame,
) -> float:
    total = 0.0

    for label in LABELS_ABC:
        members = X.loc[labels == label]

        if members.empty:
            continue

        diffs = (
            members.to_numpy(dtype=float)
            - centers.loc[label].to_numpy(dtype=float)
        )

        total += float((diffs ** 2).sum())

    return total


def compute_cluster_counts(labels: pd.Series) -> Dict[str, int]:
    return (
        labels
        .value_counts()
        .reindex(LABELS_ABC, fill_value=0)
        .to_dict()
    )


def evaluate_internal_metrics(
    X: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Optional[float]]:
    unique_labels = labels.nunique(dropna=True)

    metrics = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    if unique_labels < 2 or unique_labels >= len(X):
        return metrics

    metrics["silhouette"] = float(silhouette_score(X, labels))
    metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))

    return metrics