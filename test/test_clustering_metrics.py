import math

import pandas as pd
import pytest

from app.ml.metrics.clustering_metrics import (
    compute_cluster_counts,
    compute_inertia,
    evaluate_internal_metrics,
)


def test_compute_inertia_matches_manual_squared_distance_sum():
    X = pd.DataFrame({"x": [0.0, 2.0], "y": [0.0, 0.0]})
    labels = pd.Series(["A", "B"], index=X.index)
    centers = pd.DataFrame(
        {"x": [1.0, 1.0, 0.0], "y": [0.0, 0.0, 0.0]},
        index=["A", "B", "C"],
    )
    assert compute_inertia(X, labels, centers) == pytest.approx(2.0)


def test_compute_cluster_counts_includes_empty_abc_categories():
    counts = compute_cluster_counts(pd.Series(["A", "A", "C"]))
    assert counts == {"A": 2, "B": 0, "C": 1}


def test_evaluate_internal_metrics_returns_finite_values():
    X = pd.DataFrame(
        {
            "x": [0.0, 0.1, 5.0, 5.1, 10.0, 10.1],
            "y": [0.0, 0.1, 5.0, 5.1, 0.0, 0.1],
        }
    )
    labels = pd.Series(["A", "A", "B", "B", "C", "C"])
    metrics = evaluate_internal_metrics(X, labels)
    assert set(metrics) == {
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize(
    "labels",
    [
        ["A", "A", "A"],
        ["A", "B", "C"],
    ],
)
def test_evaluate_internal_metrics_handles_invalid_partitions(labels):
    X = pd.DataFrame({"x": range(len(labels)), "y": range(len(labels))})
    assert evaluate_internal_metrics(X, pd.Series(labels)) == {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
