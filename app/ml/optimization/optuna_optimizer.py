from __future__ import annotations

import time
from itertools import combinations
from typing import Any, Dict, Sequence

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from app.ml.assignment.constrained_assignment import MetodoAsignacion
from app.ml.core.ss_kmeans import SSEKMeans


PROPORTIONS = {
    "A": 0.20,
    "B": 0.30,
    "C": 0.50,
}

N_INIT_OPTIONS = [5, 10, 20, 30]
MAX_ITER = 300
SHUFFLE_UNLABELED = False
OPTIMIZATION_SEEDS = (0, 42, 123)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0

    return float(np.std(values, ddof=1))


def _mean_pairwise_ari(
    labels_by_seed: Sequence[pd.Series],
) -> float:
    if len(labels_by_seed) < 2:
        return 1.0

    ari_values = [
        adjusted_rand_score(
            first_labels,
            second_labels,
        )
        for first_labels, second_labels in combinations(
            labels_by_seed,
            2,
        )
    ]

    return _mean(ari_values)


def _mean_exact_agreement(
    labels_by_seed: Sequence[pd.Series],
) -> float:
    if len(labels_by_seed) < 2:
        return 100.0

    agreement_values = [
        float(
            (
                first_labels
                == second_labels
            ).mean()
            * 100
        )
        for first_labels, second_labels in combinations(
            labels_by_seed,
            2,
        )
    ]

    return _mean(agreement_values)


def optimize_sse_kmeans(
    X: pd.DataFrame,
    n_trials: int = 30,
    random_state: int = 42,
    optimization_seeds: Sequence[int] = OPTIMIZATION_SEEDS,
    metodo_asignacion: MetodoAsignacion = "global",
) -> Dict[str, Any]:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if X.empty:
        raise ValueError(
            "X no puede estar vacío."
        )

    if n_trials < 1:
        raise ValueError(
            "n_trials debe ser mayor o igual a 1."
        )

    if metodo_asignacion not in {
        "global",
        "secuencial",
    }:
        raise ValueError(
            "metodo_asignacion debe ser "
            "'global' o 'secuencial'. "
            f"Valor recibido: {metodo_asignacion!r}."
        )

    seeds = tuple(
        int(seed)
        for seed in optimization_seeds
    )

    if not seeds:
        raise ValueError(
            "Debe proporcionarse al menos "
            "una semilla de optimización."
        )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        n_init = trial.suggest_categorical(
            "n_init",
            N_INIT_OPTIONS,
        )

        tol = trial.suggest_float(
            "tol",
            1e-6,
            1e-2,
            log=True,
        )

        silhouettes: list[float] = []
        davies_bouldin_values: list[float] = []
        calinski_harabasz_values: list[float] = []
        inertias: list[float] = []
        normalized_inertias: list[float] = []
        elapsed_values: list[float] = []
        iteration_values: list[float] = []
        converged_run_values: list[float] = []
        discarded_run_values: list[float] = []
        labels_by_seed: list[pd.Series] = []

        for seed in seeds:
            model = SSEKMeans(
                proportions=PROPORTIONS,
                max_iter=MAX_ITER,
                tol=tol,
                n_init=int(n_init),
                random_state=seed,
                shuffle_unlabeled=SHUFFLE_UNLABELED,
                metodo_asignacion=metodo_asignacion,
            )

            start = time.perf_counter()

            try:
                result = model.fit_predict(X)
            except (
                RuntimeError,
                ValueError,
                FloatingPointError,
            ) as error:
                trial.set_user_attr(
                    "error",
                    (
                        f"Semilla {seed}: "
                        f"{error}"
                    ),
                )

                raise optuna.TrialPruned(
                    "La configuración falló "
                    f"con la semilla {seed}."
                ) from error

            elapsed = (
                time.perf_counter()
                - start
            )

            labels = (
                result["cluster"]
                .astype(int)
                .reset_index(drop=True)
            )

            if labels.nunique() < 2:
                trial.set_user_attr(
                    "error",
                    (
                        f"Semilla {seed}: "
                        "se obtuvo menos "
                        "de dos grupos."
                    ),
                )

                raise optuna.TrialPruned(
                    "La configuración no generó "
                    "suficientes grupos."
                )

            inertia = float(
                model.inertia_
                or 0.0
            )

            silhouettes.append(
                float(
                    silhouette_score(
                        X,
                        labels,
                    )
                )
            )

            davies_bouldin_values.append(
                float(
                    davies_bouldin_score(
                        X,
                        labels,
                    )
                )
            )

            calinski_harabasz_values.append(
                float(
                    calinski_harabasz_score(
                        X,
                        labels,
                    )
                )
            )

            inertias.append(
                inertia
            )

            normalized_inertias.append(
                inertia
                / max(
                    len(X),
                    1,
                )
            )

            elapsed_values.append(
                float(elapsed)
            )

            iteration_values.append(
                float(
                    model.n_iter_
                    or 0
                )
            )

            converged_run_values.append(
                float(
                    model.converged_runs_
                )
            )

            discarded_run_values.append(
                float(
                    model.discarded_runs_
                )
            )

            labels_by_seed.append(
                labels
            )

        mean_silhouette = _mean(
            silhouettes
        )

        trial.set_user_attr(
            "metodo_asignacion",
            metodo_asignacion,
        )

        trial.set_user_attr(
            "optimization_seeds",
            list(seeds),
        )

        trial.set_user_attr(
            "silhouette_mean",
            mean_silhouette,
        )

        trial.set_user_attr(
            "silhouette_std",
            _std(
                silhouettes
            ),
        )

        trial.set_user_attr(
            "davies_bouldin_mean",
            _mean(
                davies_bouldin_values
            ),
        )

        trial.set_user_attr(
            "davies_bouldin_std",
            _std(
                davies_bouldin_values
            ),
        )

        trial.set_user_attr(
            "calinski_harabasz_mean",
            _mean(
                calinski_harabasz_values
            ),
        )

        trial.set_user_attr(
            "calinski_harabasz_std",
            _std(
                calinski_harabasz_values
            ),
        )

        trial.set_user_attr(
            "inertia_mean",
            _mean(
                inertias
            ),
        )

        trial.set_user_attr(
            "inertia_std",
            _std(
                inertias
            ),
        )

        trial.set_user_attr(
            "normalized_inertia_mean",
            _mean(
                normalized_inertias
            ),
        )

        trial.set_user_attr(
            "elapsed_seconds_mean",
            _mean(
                elapsed_values
            ),
        )

        trial.set_user_attr(
            "elapsed_seconds_total",
            float(
                sum(
                    elapsed_values
                )
            ),
        )

        trial.set_user_attr(
            "iterations_mean",
            _mean(
                iteration_values
            ),
        )

        trial.set_user_attr(
            "converged_runs_mean",
            _mean(
                converged_run_values
            ),
        )

        trial.set_user_attr(
            "discarded_runs_mean",
            _mean(
                discarded_run_values
            ),
        )

        trial.set_user_attr(
            "pairwise_ari_mean",
            _mean_pairwise_ari(
                labels_by_seed
            ),
        )

        trial.set_user_attr(
            "exact_agreement_mean_pct",
            _mean_exact_agreement(
                labels_by_seed
            ),
        )

        return mean_silhouette

    sampler = optuna.samplers.TPESampler(
        seed=random_state,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=(
            "sse_kmeans_internal_quality"
        ),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    complete_trials = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
        )
    ]

    if not complete_trials:
        raise RuntimeError(
            "Optuna no completó ningún "
            "trial válido. Revisa los "
            "errores registrados en las "
            "corridas podadas."
        )

    best_trial = study.best_trial

    best_params = {
        "n_init": int(
            best_trial.params["n_init"]
        ),
        "max_iter": MAX_ITER,
        "tol": float(
            best_trial.params["tol"]
        ),
        "random_state": random_state,
        "shuffle_unlabeled": (
            SHUFFLE_UNLABELED
        ),
        "metodo_asignacion": (
            metodo_asignacion
        ),
    }

    return {
        "best_params": best_params,
        "best_value": float(
            study.best_value
        ),
        "best_trial": int(
            best_trial.number
        ),
        "best_metrics": dict(
            best_trial.user_attrs
        ),
        "optimization_config": {
            "n_trials": n_trials,
            "sampler": "TPESampler",
            "sampler_seed": random_state,
            "optimization_seeds": list(
                seeds
            ),
            "objective": (
                "mean_silhouette"
            ),
            "n_init_options": list(
                N_INIT_OPTIONS
            ),
            "tol_min": 1e-6,
            "tol_max": 1e-2,
            "max_iter_fixed": (
                MAX_ITER
            ),
            "shuffle_unlabeled_fixed": (
                SHUFFLE_UNLABELED
            ),
            "metodo_asignacion": (
                metodo_asignacion
            ),
        },
        "trials": [
            {
                "number": int(
                    trial.number
                ),
                "state": (
                    trial.state.name
                ),
                "value": (
                    None
                    if trial.value is None
                    else float(
                        trial.value
                    )
                ),
                "params": dict(
                    trial.params
                ),
                "attrs": dict(
                    trial.user_attrs
                ),
            }
            for trial in study.trials
        ],
    }