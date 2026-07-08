from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from app.ml.assignment.constrained_assignment import assign_with_capacity
from app.ml.constraints.capacity_constraint import (
    compute_capacities,
    validate_seed_capacity,
)
from app.ml.core.config import SSEKMeansConfig
from app.ml.core.types import LABELS_ABC, RunResult
from app.ml.initialization.centroid_initializer import (
    compute_initial_score,
    initialize_centroids,
)
from app.ml.metrics.clustering_metrics import (
    compute_cluster_counts,
    compute_inertia,
)
from app.ml.semi_supervision.seed_handler import (
    count_seed_labels,
    normalize_seed_labels,
)
from app.ml.update.centroid_update import (
    compute_center_shift,
    recompute_centroids,
)


class SSEKMeans:
    def __init__(
        self,
        proportions: Optional[Mapping[str, float]] = None,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 5,
        random_state: Optional[int] = 42,
        shuffle_unlabeled: bool = True,
        config: Optional[SSEKMeansConfig] = None,
    ) -> None:
        if config is not None:
            self.proportions = dict(config.proportions)
            self.max_iter = config.max_iter
            self.tol = config.tol
            self.n_init = config.n_init
            self.random_state = config.random_state
            self.shuffle_unlabeled = config.shuffle_unlabeled
        else:
            self.proportions = dict(
                proportions or {
                    "A": 0.20,
                    "B": 0.30,
                    "C": 0.50,
                }
            )
            self.max_iter = max_iter
            self.tol = tol
            self.n_init = n_init
            self.random_state = random_state
            self.shuffle_unlabeled = shuffle_unlabeled

        self._validate_parameters()

        self.labels_: Optional[pd.Series] = None
        self.cluster_centers_: Optional[pd.DataFrame] = None
        self.inertia_: Optional[float] = None
        self.objective_history_: Optional[list[float]] = None
        self.n_iter_: Optional[int] = None
        self.capacities_: Optional[Dict[str, int]] = None
        self.counts_: Optional[Dict[str, int]] = None
        self.seed_counts_: Optional[Dict[str, int]] = None
        self.score_: Optional[pd.Series] = None
        self.is_seed_: Optional[pd.Series] = None
        self.results_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: pd.DataFrame,
        seed_labels: Optional[Sequence[object]] = None,
    ) -> "SSEKMeans":
        X_df = self._validate_X(X)

        seed_series = normalize_seed_labels(
            seed_labels,
            index=X_df.index,
        )

        capacities = compute_capacities(
            n_samples=len(X_df),
            proportions=self.proportions,
            labels=LABELS_ABC,
        )

        seed_counts = count_seed_labels(seed_series)

        validate_seed_capacity(
            seed_counts=seed_counts,
            capacities=capacities,
            labels=LABELS_ABC,
        )

        scores = compute_initial_score(X_df)

        rng = np.random.default_rng(self.random_state)

        best_run: Optional[RunResult] = None

        for _ in range(self.n_init):
            run_rng = np.random.default_rng(
                rng.integers(0, np.iinfo(np.int32).max)
            )

            run = self._fit_single_run(
                X=X_df,
                seed_labels=seed_series,
                scores=scores,
                capacities=capacities,
                rng=run_rng,
            )

            if best_run is None or run.inertia < best_run.inertia:
                best_run = run

        if best_run is None:
            raise RuntimeError("No se pudo ejecutar ninguna corrida del modelo.")

        self._save_best_run(best_run, seed_counts)

        return self

    def fit_predict(
        self,
        X: pd.DataFrame,
        seed_labels: Optional[Sequence[object]] = None,
    ) -> pd.DataFrame:
        self.fit(X, seed_labels=seed_labels)

        if self.results_ is None:
            raise RuntimeError("El modelo no generó resultados.")

        return self.results_.copy()

    def _fit_single_run(
        self,
        X: pd.DataFrame,
        seed_labels: pd.Series,
        scores: pd.Series,
        capacities: Dict[str, int],
        rng: np.random.Generator,
    ) -> RunResult:
        centers = initialize_centroids(
            X=X,
            seed_labels=seed_labels,
            scores=scores,
            rng=rng,
        )

        previous_labels: Optional[pd.Series] = None
        objective_history: list[float] = []

        last_labels: Optional[pd.Series] = None
        last_is_seed: Optional[pd.Series] = None
        last_iteration = 0

        for iteration in range(1, self.max_iter + 1):
            labels, is_seed = assign_with_capacity(
                X=X,
                centers=centers,
                capacities=capacities,
                seed_labels=seed_labels,
                rng=rng,
                shuffle_unlabeled=self.shuffle_unlabeled,
            )

            new_centers = recompute_centroids(
                X=X,
                labels=labels,
                previous_centers=centers,
            )

            inertia = compute_inertia(
                X=X,
                labels=labels,
                centers=new_centers,
            )

            objective_history.append(float(inertia))

            labels_stable = (
                previous_labels is not None
                and labels.equals(previous_labels)
            )

            center_shift = compute_center_shift(
                old_centers=centers,
                new_centers=new_centers,
            )

            improvement = (
                float("inf")
                if len(objective_history) == 1
                else abs(objective_history[-2] - objective_history[-1])
            )

            centers = new_centers
            previous_labels = labels
            last_labels = labels
            last_is_seed = is_seed
            last_iteration = iteration

            if (
                labels_stable
                or center_shift <= self.tol
                or improvement <= self.tol
            ):
                break

        if last_labels is None or last_is_seed is None or not objective_history:
            raise RuntimeError("La corrida del modelo no generó resultados válidos.")

        counts = compute_cluster_counts(last_labels)

        return RunResult(
            labels=last_labels,
            centers=centers,
            inertia=float(objective_history[-1]),
            objective_history=objective_history,
            n_iter=last_iteration,
            capacities=dict(capacities),
            counts=counts,
            scores=scores.copy(),
            is_seed=last_is_seed,
        )

    def _save_best_run(
        self,
        best_run: RunResult,
        seed_counts: Dict[str, int],
    ) -> None:
        self.labels_ = best_run.labels
        self.cluster_centers_ = best_run.centers
        self.inertia_ = best_run.inertia
        self.objective_history_ = best_run.objective_history
        self.n_iter_ = best_run.n_iter
        self.capacities_ = best_run.capacities
        self.counts_ = best_run.counts
        self.seed_counts_ = seed_counts
        self.score_ = best_run.scores
        self.is_seed_ = best_run.is_seed
        self.results_ = self._build_results(best_run)

    def _validate_parameters(self) -> None:
        if self.max_iter < 1:
            raise ValueError("max_iter debe ser mayor o igual a 1.")

        if self.n_init < 1:
            raise ValueError("n_init debe ser mayor o igual a 1.")

        if self.tol < 0:
            raise ValueError("tol debe ser mayor o igual a 0.")

        missing_labels = [
            label for label in LABELS_ABC
            if label not in self.proportions
        ]

        if missing_labels:
            raise ValueError(
                "Faltan proporciones para las siguientes categorías: "
                f"{missing_labels}"
            )

        invalid_labels = [
            label for label in self.proportions
            if label not in LABELS_ABC
        ]

        if invalid_labels:
            raise ValueError(
                "Se recibieron categorías inválidas en proportions: "
                f"{invalid_labels}. Las categorías válidas son {LABELS_ABC}."
            )

        invalid_values = {
            label: value
            for label, value in self.proportions.items()
            if value < 0
        }

        if invalid_values:
            raise ValueError(
                "Las proporciones no pueden ser negativas. "
                f"Valores inválidos: {invalid_values}"
            )

        total = sum(self.proportions.values())

        if not np.isclose(total, 1.0):
            raise ValueError(
                "Las proporciones deben sumar 1.0. "
                f"Suma actual: {total}"
            )

    @staticmethod
    def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.empty:
            raise ValueError("X no puede estar vacío.")

        non_numeric = [
            col for col in X.columns
            if not pd.api.types.is_numeric_dtype(X[col])
        ]

        if non_numeric:
            raise ValueError(
                "Todas las columnas de X deben ser numéricas. "
                f"Columnas inválidas: {non_numeric}"
            )

        if X.isna().any().any():
            raise ValueError(
                "X contiene valores faltantes. "
                "Debes imputar o eliminar NaN antes de clasificar."
            )

        return X.astype(float).copy()

    @staticmethod
    def _build_results(run: RunResult) -> pd.DataFrame:
        cluster_map = {
            "A": 0,
            "B": 1,
            "C": 2,
        }

        return pd.DataFrame(
            {
                "categoria": run.labels,
                "cluster": run.labels.map(cluster_map).astype(int),
                "score_inicial": run.scores,
                "es_semilla": run.is_seed.astype(bool),
            },
            index=run.labels.index,
        )