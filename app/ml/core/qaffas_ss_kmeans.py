from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from app.ml.assignment.qaffas_assignment import (
    assign_with_capacity_qaffas,
)
from app.ml.constraints.capacity_constraint import (
    compute_capacities,
)
from app.ml.metrics.clustering_metrics import (
    compute_cluster_counts,
    compute_inertia,
)
from app.ml.update.centroid_update import (
    recompute_centroids,
)


LABELS_ABC = ("A", "B", "C")


def compute_qaffas_score(
    X: pd.DataFrame,
) -> pd.Series:
    """
    Calcula Ω_i como la suma de los criterios normalizados.

    X debe encontrarse previamente normalizado en el
    intervalo [0, 1].
    """
    return (
        X.sum(axis=1)
        .astype(float)
    )


def initialize_qaffas_centroids(
    X: pd.DataFrame,
    scores: pd.Series,
) -> pd.DataFrame:
    """
    Inicializa los representantes según Qaffas et al.:

    - A: observación con mayor Ω_i.
    - B: observación mediana según Ω_i.
    - C: observación con menor Ω_i.

    En caso de empate se conserva el orden estable del
    DataFrame.
    """
    if len(X) < 3:
        raise ValueError(
            "Se requieren al menos tres observaciones."
        )

    if not scores.index.equals(
        X.index
    ):
        scores = scores.reindex(
            X.index
        )

    if scores.isna().any():
        raise ValueError(
            "El score contiene valores faltantes."
        )

    ordered_indices = (
        scores
        .sort_values(
            ascending=True,
            kind="mergesort",
        )
        .index
        .tolist()
    )

    idx_c = ordered_indices[0]
    idx_b = ordered_indices[
        len(ordered_indices) // 2
    ]
    idx_a = ordered_indices[-1]

    if len(
        {
            idx_a,
            idx_b,
            idx_c,
        }
    ) != 3:
        raise RuntimeError(
            "No fue posible seleccionar tres "
            "representantes iniciales diferentes."
        )

    selected_indices = {
        "A": idx_a,
        "B": idx_b,
        "C": idx_c,
    }

    centers = pd.DataFrame(
        {
            label: (
                X.loc[index]
                .to_numpy(dtype=float)
            )
            for label, index
            in selected_indices.items()
        },
        index=X.columns,
    ).T

    return centers.loc[
        list(LABELS_ABC)
    ]


class QaffasSSEKMeans:
    """
    Implementación experimental del SS-E-k-means descrito
    por Qaffas et al.

    Características principales:

    - utiliza una inicialización determinista;
    - A, B y C se inicializan usando el score Ω_i;
    - utiliza asignación restringida según preferencias
      de distancia;
    - se detiene cuando no mejora la función objetivo
      o cuando alcanza max_iter;
    - espera variables previamente normalizadas mediante
      Min-Max en el intervalo [0, 1].
    """

    def __init__(
        self,
        proportions: Optional[
            Mapping[str, float]
        ] = None,
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        self.proportions = dict(
            proportions
            or {
                "A": 0.20,
                "B": 0.30,
                "C": 0.50,
            }
        )

        self.max_iter = int(
            max_iter
        )

        self.tol = float(
            tol
        )

        self._validate_parameters()

        self.labels_: Optional[
            pd.Series
        ] = None

        self.cluster_centers_: Optional[
            pd.DataFrame
        ] = None

        self.inertia_: Optional[
            float
        ] = None

        self.objective_history_: Optional[
            list[float]
        ] = None

        self.n_iter_: Optional[
            int
        ] = None

        self.best_iteration_: Optional[
            int
        ] = None

        self.capacities_: Optional[
            Dict[str, int]
        ] = None

        self.counts_: Optional[
            Dict[str, int]
        ] = None

        self.score_: Optional[
            pd.Series
        ] = None

        self.results_: Optional[
            pd.DataFrame
        ] = None

        self.converged_: Optional[
            bool
        ] = None

        self.stop_reason_: Optional[
            str
        ] = None

    def fit(
        self,
        X: pd.DataFrame,
    ) -> "QaffasSSEKMeans":
        X_df = self._validate_X(
            X
        )

        capacities = compute_capacities(
            n_samples=len(X_df),
            proportions=self.proportions,
            labels=LABELS_ABC,
        )

        scores = compute_qaffas_score(
            X_df
        )

        centers = initialize_qaffas_centroids(
            X=X_df,
            scores=scores,
        )

        previous_labels: Optional[
            pd.Series
        ] = None

        previous_inertia: Optional[
            float
        ] = None

        objective_history: list[
            float
        ] = []

        best_labels: Optional[
            pd.Series
        ] = None

        best_centers: Optional[
            pd.DataFrame
        ] = None

        best_inertia = float(
            "inf"
        )

        best_iteration = 0
        last_iteration = 0

        converged = False
        stop_reason = "max_iter"

        for iteration in range(
            1,
            self.max_iter + 1,
        ):
            labels = (
                assign_with_capacity_qaffas(
                    X=X_df,
                    centers=centers,
                    capacities=capacities,
                )
            )

            new_centers = (
                recompute_centroids(
                    X=X_df,
                    labels=labels,
                    previous_centers=centers,
                )
            )

            inertia = compute_inertia(
                X=X_df,
                labels=labels,
                centers=new_centers,
            )

            inertia = float(
                inertia
            )

            objective_history.append(
                inertia
            )

            last_iteration = iteration

            if inertia < best_inertia:
                best_inertia = inertia

                best_labels = (
                    labels.copy()
                )

                best_centers = (
                    new_centers.copy()
                )

                best_iteration = (
                    iteration
                )

            labels_stable = (
                previous_labels is not None
                and labels.equals(
                    previous_labels
                )
            )

            improvement = (
                None
                if previous_inertia is None
                else (
                    previous_inertia
                    - inertia
                )
            )

            centers = new_centers

            if labels_stable:
                converged = True

                stop_reason = (
                    "labels_stable"
                )

                break

            if (
                improvement is not None
                and improvement <= self.tol
            ):
                converged = True

                stop_reason = (
                    "no_objective_improvement"
                )

                break

            previous_labels = (
                labels.copy()
            )

            previous_inertia = inertia

        if (
            best_labels is None
            or best_centers is None
            or not np.isfinite(
                best_inertia
            )
        ):
            raise RuntimeError(
                "El modelo Qaffas no generó "
                "una solución válida."
            )

        self.labels_ = (
            best_labels
        )

        self.cluster_centers_ = (
            best_centers
        )

        self.inertia_ = (
            best_inertia
        )

        self.objective_history_ = (
            objective_history
        )

        self.n_iter_ = (
            last_iteration
        )

        self.best_iteration_ = (
            best_iteration
        )

        self.capacities_ = dict(
            capacities
        )

        self.counts_ = (
            compute_cluster_counts(
                best_labels
            )
        )

        self.score_ = (
            scores.copy()
        )

        self.converged_ = (
            converged
        )

        self.stop_reason_ = (
            stop_reason
        )

        self.results_ = (
            self._build_results(
                labels=best_labels,
                scores=scores,
            )
        )

        return self

    def fit_predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        self.fit(
            X
        )

        if self.results_ is None:
            raise RuntimeError(
                "El modelo no generó resultados."
            )

        return (
            self.results_
            .copy()
        )

    def _validate_parameters(
        self,
    ) -> None:
        if self.max_iter < 1:
            raise ValueError(
                "max_iter debe ser mayor o igual a 1."
            )

        if self.tol < 0:
            raise ValueError(
                "tol debe ser mayor o igual a 0."
            )

        missing_labels = [
            label
            for label in LABELS_ABC
            if label not in self.proportions
        ]

        if missing_labels:
            raise ValueError(
                "Faltan proporciones para: "
                f"{missing_labels}."
            )

        invalid_labels = [
            label
            for label in self.proportions
            if label not in LABELS_ABC
        ]

        if invalid_labels:
            raise ValueError(
                "Se recibieron categorías inválidas: "
                f"{invalid_labels}."
            )

        negative_values = {
            label: value
            for label, value
            in self.proportions.items()
            if value < 0
        }

        if negative_values:
            raise ValueError(
                "Las proporciones no pueden ser negativas: "
                f"{negative_values}."
            )

        total = sum(
            self.proportions.values()
        )

        if not np.isclose(
            total,
            1.0,
        ):
            raise ValueError(
                "Las proporciones deben sumar 1.0. "
                f"Suma actual: {total}."
            )

    @staticmethod
    def _validate_X(
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            X = pd.DataFrame(
                X
            )

        if X.empty:
            raise ValueError(
                "X no puede estar vacío."
            )

        if not X.index.is_unique:
            raise ValueError(
                "El índice de X debe ser único."
            )

        non_numeric = [
            column
            for column in X.columns
            if not pd.api.types.is_numeric_dtype(
                X[column]
            )
        ]

        if non_numeric:
            raise ValueError(
                "Todas las columnas deben ser numéricas. "
                f"Columnas inválidas: {non_numeric}."
            )

        if X.isna().any().any():
            raise ValueError(
                "X contiene valores faltantes."
            )

        X_float = (
            X.astype(float)
            .copy()
        )

        values = X_float.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "X contiene valores infinitos "
                "o no finitos."
            )

        # MinMaxScaler puede producir desviaciones
        # microscópicas como 1.0000000000000002.
        tolerance = 1e-12

        minimum_value = float(
            X_float.min().min()
        )

        maximum_value = float(
            X_float.max().max()
        )

        if (
            minimum_value < -tolerance
            or maximum_value > 1.0 + tolerance
        ):
            raise ValueError(
                "QaffasSSEKMeans espera variables "
                "normalizadas en el intervalo [0, 1]. "
                f"Mínimo encontrado={minimum_value}, "
                f"máximo encontrado={maximum_value}."
            )

        # Corrige solamente las pequeñas desviaciones
        # numéricas ocasionadas por punto flotante.
        X_float = X_float.clip(
            lower=0.0,
            upper=1.0,
        )

        return X_float

    @staticmethod
    def _build_results(
        labels: pd.Series,
        scores: pd.Series,
    ) -> pd.DataFrame:
        cluster_map = {
            "A": 0,
            "B": 1,
            "C": 2,
        }

        return pd.DataFrame(
            {
                "categoria": labels,
                "cluster": (
                    labels
                    .map(cluster_map)
                    .astype(int)
                ),
                "score_inicial": scores,
            },
            index=labels.index,
        )