import pandas as pd
import pytest

import app.ml.optimization.optuna_optimizer as optimizer


def optimization_frame():
    return pd.DataFrame(
        {
            "x": [0.0, 0.1, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2, 10.3, 10.4],
            "y": [0.0, 0.1, 5.0, 5.1, 5.2, 0.0, 0.1, 0.2, 0.3, 0.4],
        }
    )


class FakeSSEKMeans:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inertia_ = 12.0
        self.n_iter_ = 3
        self.converged_runs_ = kwargs["n_init"]
        self.discarded_runs_ = 0

    def fit_predict(self, X):
        return pd.DataFrame(
            {
                "categoria": ["A"] * 2 + ["B"] * 3 + ["C"] * 5,
                "cluster": [0] * 2 + [1] * 3 + [2] * 5,
            }
        )


def test_statistical_helpers_cover_single_and_multiple_runs():
    labels = pd.Series([0, 0, 1, 1])
    assert optimizer._mean([1.0, 3.0]) == 2.0
    assert optimizer._std([1.0]) == 0.0
    assert optimizer._mean_pairwise_ari([labels]) == 1.0
    assert optimizer._mean_pairwise_ari([labels, labels.copy()]) == 1.0
    assert optimizer._mean_exact_agreement([labels]) == 100.0
    assert optimizer._mean_exact_agreement(
        [labels, labels.copy()]
    ) == 100.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"X": pd.DataFrame()},
        {"X": optimization_frame(), "n_trials": 0},
        {"X": optimization_frame(), "optimization_seeds": ()},
    ],
)
def test_optimize_sse_kmeans_validates_inputs(kwargs):
    with pytest.raises(ValueError):
        optimizer.optimize_sse_kmeans(**kwargs)


def test_optimize_sse_kmeans_reports_best_trial_and_stability(monkeypatch):
    monkeypatch.setattr(optimizer, "SSEKMeans", FakeSSEKMeans)
    result = optimizer.optimize_sse_kmeans(
        X=optimization_frame(),
        n_trials=2,
        random_state=42,
        optimization_seeds=(0, 42),
    )
    assert result["best_trial"] in {0, 1}
    assert result["best_value"] == pytest.approx(
        result["best_metrics"]["silhouette_mean"]
    )
    assert result["best_params"]["n_init"] in optimizer.N_INIT_OPTIONS
    assert 1e-6 <= result["best_params"]["tol"] <= 1e-2
    assert result["best_params"]["max_iter"] == optimizer.MAX_ITER
    assert result["optimization_config"]["optimization_seeds"] == [0, 42]
    assert result["optimization_config"]["objective"] == "mean_silhouette"
    assert result["best_metrics"]["pairwise_ari_mean"] == 1.0
    assert result["best_metrics"]["exact_agreement_mean_pct"] == 100.0
    assert len(result["trials"]) == 2
    assert all(trial["state"] == "COMPLETE" for trial in result["trials"])
