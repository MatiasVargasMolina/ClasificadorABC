import numpy as np
import pandas as pd
import pandas.testing as pdt

from app.ml.initialization.centroid_initializer import (
    compute_initial_score,
    initialize_centroids,
)


def test_compute_initial_score_uses_min_max_value_per_feature():
    X = pd.DataFrame(
        {
            "ventas": [0.0, 5.0, 10.0],
            "visitas": [0.0, 5.0, 10.0],
        }
    )
    score = compute_initial_score(X)
    pdt.assert_series_equal(
        score,
        pd.Series([0.0, 1.0, 2.0]),
    )


def test_compute_initial_score_handles_constant_columns_without_nan():
    X = pd.DataFrame(
        {
            "ventas": [0.0, 5.0, 10.0],
            "constante": [3.0, 3.0, 3.0],
        }
    )
    score = compute_initial_score(X)
    assert score.tolist() == [0.0, 0.5, 1.0]
    assert score.notna().all()


def test_initialize_centroids_uses_mean_of_available_seed_labels():
    X = pd.DataFrame(
        {
            "x": [10.0, 8.0, 4.0, 0.0],
            "y": [8.0, 6.0, 4.0, 2.0],
        }
    )
    seed_labels = pd.Series(["A", "A", "B", "C"], index=X.index)
    centers = initialize_centroids(
        X=X,
        seed_labels=seed_labels,
        scores=compute_initial_score(X),
        rng=np.random.default_rng(42),
    )
    assert centers.loc["A"].to_dict() == {"x": 9.0, "y": 7.0}
    assert centers.loc["B"].to_dict() == {"x": 4.0, "y": 4.0}
    assert centers.loc["C"].to_dict() == {"x": 0.0, "y": 2.0}


def test_initialize_centroids_without_external_seeds_is_reproducible():
    X = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "y": np.arange(12, dtype=float) * 2,
        }
    )
    seed_labels = pd.Series("", index=X.index, dtype=object)
    scores = compute_initial_score(X)
    first = initialize_centroids(
        X,
        seed_labels,
        scores,
        rng=np.random.default_rng(7),
    )
    second = initialize_centroids(
        X,
        seed_labels,
        scores,
        rng=np.random.default_rng(7),
    )
    pdt.assert_frame_equal(first, second)
    assert first.index.tolist() == ["A", "B", "C"]
    assert first.columns.tolist() == X.columns.tolist()
    assert first.isna().sum().sum() == 0
