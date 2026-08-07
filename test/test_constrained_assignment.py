import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from app.ml.assignment.constrained_assignment import (
    assign_with_capacity,
    compute_distance_matrix,
)


def test_compute_distance_matrix_uses_squared_euclidean_distance():
    X = pd.DataFrame({"x": [0.0, 2.0], "y": [0.0, 0.0]})
    centers = pd.DataFrame(
        {"x": [0.0, 2.0, 1.0], "y": [0.0, 0.0, 0.0]},
        index=["A", "B", "C"],
    )
    distances = compute_distance_matrix(X, centers)
    expected = pd.DataFrame(
        {"A": [0.0, 4.0], "B": [4.0, 0.0], "C": [1.0, 1.0]}
    )
    pdt.assert_frame_equal(distances, expected)


def test_assign_with_capacity_respects_exact_quotas():
    X = pd.DataFrame(
        {
            "x": [0.0, 0.1, 5.0, 5.1, 10.0, 10.1],
            "y": [0.0, 0.1, 5.0, 5.1, 0.0, 0.1],
        }
    )
    centers = pd.DataFrame(
        {"x": [0.0, 5.0, 10.0], "y": [0.0, 5.0, 0.0]},
        index=["A", "B", "C"],
    )
    labels = assign_with_capacity(
        X=X,
        centers=centers,
        capacities={"A": 2, "B": 2, "C": 2},
        rng=np.random.default_rng(999),
        shuffle_unlabeled=True,
    )
    assert labels.tolist() == ["A", "A", "B", "B", "C", "C"]
    assert labels.value_counts().to_dict() == {"A": 2, "B": 2, "C": 2}


def test_assign_with_capacity_is_deterministic_for_same_inputs():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    centers = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        index=["A", "B", "C"],
    )
    capacities = {"A": 1, "B": 1, "C": 1}
    first = assign_with_capacity(
        X,
        centers,
        capacities,
        rng=np.random.default_rng(1),
        shuffle_unlabeled=False,
    )
    second = assign_with_capacity(
        X,
        centers,
        capacities,
        rng=np.random.default_rng(2),
        shuffle_unlabeled=True,
    )
    pdt.assert_series_equal(first, second)


def test_assign_with_capacity_rejects_inconsistent_total():
    X = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    centers = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        index=["A", "B", "C"],
    )
    with pytest.raises(ValueError, match="suma de las capacidades"):
        assign_with_capacity(
            X,
            centers,
            {"A": 1, "B": 1, "C": 1},
            rng=np.random.default_rng(42),
        )


def test_assign_with_capacity_rejects_missing_or_negative_capacity():
    X = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    centers = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        index=["A", "B", "C"],
    )
    with pytest.raises(ValueError, match="No se encontró capacidad"):
        assign_with_capacity(
            X,
            centers,
            {"A": 1, "B": 1},
            rng=np.random.default_rng(42),
        )
    with pytest.raises(ValueError, match="no pueden ser negativas"):
        assign_with_capacity(
            X,
            centers,
            {"A": -1, "B": 1, "C": 2},
            rng=np.random.default_rng(42),
        )
