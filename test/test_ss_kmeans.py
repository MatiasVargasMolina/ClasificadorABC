import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from app.ml.core.config import SSEKMeansConfig
from app.ml.core.ss_kmeans import SSEKMeans


def clustered_frame():
    return pd.DataFrame(
        {
            "ventas": [
                3.2, 3.0, 2.9, 2.8,
                0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
                -2.0, -2.1, -2.2, -2.3, -2.4,
                -2.5, -2.6, -2.7, -2.8, -2.9,
            ],
            "visitas": [
                3.1, 3.0, 2.9, 2.8,
                0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                -2.0, -2.1, -2.2, -2.3, -2.4,
                -2.5, -2.6, -2.7, -2.8, -2.9,
            ],
            "precio": [
                -1.0, -0.9, -0.8, -0.7,
                2.0, 1.9, 1.8, 1.7, 1.6, 1.5,
                -0.2, -0.3, -0.4, -0.5, -0.6,
                -0.7, -0.8, -0.9, -1.0, -1.1,
            ],
        }
    )


def make_model(seed=42):
    return SSEKMeans(
        proportions={"A": 0.20, "B": 0.30, "C": 0.50},
        max_iter=100,
        tol=1e-6,
        n_init=5,
        random_state=seed,
        shuffle_unlabeled=True,
    )


def test_fit_predict_respects_capacity_and_exposes_diagnostics():
    model = make_model()
    results = model.fit_predict(clustered_frame())
    assert results.columns.tolist() == [
        "categoria",
        "cluster",
        "score_inicial",
    ]
    assert results["categoria"].value_counts().to_dict() == {
        "C": 10,
        "B": 6,
        "A": 4,
    }
    assert model.capacities_ == {"A": 4, "B": 6, "C": 10}
    assert model.counts_ == model.capacities_
    assert model.converged_ is True
    assert model.stop_reason_ in {
        "labels_stable",
        "center_shift",
        "objective_improvement",
    }
    assert model.inertia_ is not None
    assert model.n_iter_ >= 1


def test_same_random_state_produces_same_partition():
    first = make_model(seed=7).fit_predict(clustered_frame())
    second = make_model(seed=7).fit_predict(clustered_frame())
    pdt.assert_frame_equal(first, second)


def test_config_object_overrides_individual_constructor_values():
    config = SSEKMeansConfig(
        max_iter=9,
        tol=0.002,
        n_init=3,
        random_state=17,
        shuffle_unlabeled=False,
    )
    model = SSEKMeans(max_iter=99, n_init=99, config=config)
    assert model.max_iter == 9
    assert model.tol == 0.002
    assert model.n_init == 3
    assert model.random_state == 17
    assert model.shuffle_unlabeled is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"n_init": 0},
        {"tol": -1},
        {"proportions": {"A": 0.2, "B": 0.3}},
        {"proportions": {"A": 0.2, "B": 0.3, "C": 0.4}},
        {"proportions": {"A": 0.2, "B": -0.1, "C": 0.9}},
    ],
)
def test_constructor_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SSEKMeans(**kwargs)


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame(),
        pd.DataFrame({"x": ["text"]}),
        pd.DataFrame({"x": [np.nan]}),
    ],
)
def test_fit_rejects_invalid_feature_matrices(X):
    with pytest.raises(ValueError):
        make_model().fit(X)
