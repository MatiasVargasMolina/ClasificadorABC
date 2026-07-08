from __future__ import annotations

import time
from typing import Any, Dict

import optuna
import pandas as pd
from sklearn.metrics import silhouette_score

from app.ml.core.ss_kmeans import SSEKMeans


def optimize_sse_kmeans(
    X: pd.DataFrame,
    n_trials: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:

    def objective(trial: optuna.Trial) -> float:
        n_init = trial.suggest_int("n_init", 2, 15)
        max_iter = trial.suggest_int("max_iter", 50, 300)

        tol = trial.suggest_float(
            "tol",
            1e-6,
            1e-2,
            log=True,
        )

        shuffle_unlabeled = trial.suggest_categorical(
            "shuffle_unlabeled",
            [True, False],
        )

        inertia_weight = trial.suggest_float(
            "inertia_weight",
            1e-8,
            1e-6,
            log=True,
        )

        time_weight = trial.suggest_float(
            "time_weight",
            0.001,
            0.05,
            log=True,
        )

        model = SSEKMeans(
            proportions={
                "A": 0.20,
                "B": 0.30,
                "C": 0.50,
            },
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            random_state=random_state,
            shuffle_unlabeled=shuffle_unlabeled,
        )

        start = time.perf_counter()

        result = model.fit_predict(X)

        elapsed = time.perf_counter() - start

        labels = result["cluster"]

        if labels.nunique() < 2:
            return -9999.0

        silhouette = silhouette_score(X, labels)

        inertia = float(model.inertia_ or 0)
        inertia_normalizada = inertia / max(len(X), 1)

        inertia_penalty = inertia_normalizada * inertia_weight
        tiempo_penalty = elapsed * time_weight

        score_final = (
            silhouette
            - inertia_penalty
            - tiempo_penalty
        )

        trial.set_user_attr("silhouette", float(silhouette))
        trial.set_user_attr("inertia", float(inertia))
        trial.set_user_attr("inertia_normalizada", float(inertia_normalizada))
        trial.set_user_attr("elapsed_seconds", float(elapsed))
        trial.set_user_attr("n_iter", int(model.n_iter_ or 0))
        trial.set_user_attr("max_iter", int(max_iter))
        trial.set_user_attr("inertia_weight", float(inertia_weight))
        trial.set_user_attr("time_weight", float(time_weight))
        trial.set_user_attr("inertia_penalty", float(inertia_penalty))
        trial.set_user_attr("tiempo_penalty", float(tiempo_penalty))
        trial.set_user_attr("score_final", float(score_final))

        return float(score_final)

    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }