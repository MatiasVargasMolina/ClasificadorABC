import pandas as pd
import pandas.testing as pdt
import pytest

from app.ml.update.centroid_update import (
    compute_center_shift,
    recompute_centroids,
)


def test_recompute_centroids_calculates_group_means():
    X = pd.DataFrame(
        {
            "x": [0.0, 2.0, 10.0, 20.0],
            "y": [2.0, 4.0, 10.0, 20.0],
        }
    )
    labels = pd.Series(["A", "A", "B", "C"], index=X.index)
    previous = pd.DataFrame(0.0, index=["A", "B", "C"], columns=X.columns)
    centers = recompute_centroids(X, labels, previous)
    expected = pd.DataFrame(
        {
            "x": [1.0, 10.0, 20.0],
            "y": [3.0, 10.0, 20.0],
        },
        index=["A", "B", "C"],
    )
    pdt.assert_frame_equal(centers, expected)


def test_recompute_centroids_keeps_previous_center_for_empty_group():
    X = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    labels = pd.Series(["A", "B"], index=X.index)
    previous = pd.DataFrame(
        {"x": [0.0, 0.0, 99.0], "y": [0.0, 0.0, 88.0]},
        index=["A", "B", "C"],
    )
    centers = recompute_centroids(X, labels, previous)
    pdt.assert_series_equal(centers.loc["C"], previous.loc["C"])


def test_compute_center_shift_returns_largest_euclidean_displacement():
    old = pd.DataFrame(
        {"x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0]},
        index=["A", "B", "C"],
    )
    new = pd.DataFrame(
        {"x": [3.0, 1.0, 0.0], "y": [4.0, 1.0, 0.0]},
        index=["A", "B", "C"],
    )
    assert compute_center_shift(old, new) == pytest.approx(5.0)
