from __future__ import annotations

from typing import Dict, Mapping, Optional

import pandas as pd

from app.ml.constraints.capacity_constraint import compute_capacities
from app.ml.core.config import DEFAULT_PROPORTIONS
from app.ml.core.types import LABELS_ABC
from app.ml.initialization.centroid_initializer import compute_initial_score
from app.ml.metrics.clustering_metrics import (
    compute_cluster_counts,
    compute_inertia,
)


class ScoreRankingABC:
    """
    Línea base determinista para clasificación ABC.

    Reutiliza el score y las capacidades del modelo actual, pero no
    ejecuta clustering ni actualización de centroides.
    """

    def __init__(
        self,
        proportions: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.proportions = dict(proportions or DEFAULT_PROPORTIONS)

        self.labels_: Optional[pd.Series] = None
        self.score_: Optional[pd.Series] = None
        self.capacities_: Optional[Dict[str, int]] = None
        self.counts_: Optional[Dict[str, int]] = None
        self.cluster_centers_: Optional[pd.DataFrame] = None
        self.inertia_: Optional[float] = None
        self.results_: Optional[pd.DataFrame] = None

        self._validate_proportions()

    def fit(self, X: pd.DataFrame) -> "ScoreRankingABC":
        X_df = self._validate_X(X)

        capacities = compute_capacities(
            n_samples=len(X_df),
            proportions=self.proportions,
            labels=LABELS_ABC,
        )
        scores = compute_initial_score(X_df)

        ordered_indices = (
            scores
            .sort_values(ascending=False, kind="mergesort")
            .index
            .tolist()
        )

        labels = pd.Series(
            index=X_df.index,
            dtype="object",
            name="categoria",
        )

        cursor = 0
        for label in LABELS_ABC:
            capacity = capacities[label]
            selected_indices = ordered_indices[cursor:cursor + capacity]
            labels.loc[selected_indices] = label
            cursor += capacity

        if labels.isna().any():
            missing = int(labels.isna().sum())
            raise RuntimeError(
                "La línea base no asignó una categoría a "
                f"{missing} publicaciones."
            )

        centers = (
            X_df
            .assign(_categoria=labels)
            .groupby("_categoria", sort=False)
            .mean()
            .reindex(LABELS_ABC)
        )

        inertia = compute_inertia(
            X=X_df,
            labels=labels,
            centers=centers,
        )

        self.labels_ = labels
        self.score_ = scores
        self.capacities_ = dict(capacities)
        self.counts_ = compute_cluster_counts(labels)
        self.cluster_centers_ = centers
        self.inertia_ = float(inertia)
        self.results_ = self._build_results(labels, scores)

        return self

    def fit_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)

        if self.results_ is None:
            raise RuntimeError("La línea base no generó resultados.")

        return self.results_.copy()

    def _validate_proportions(self) -> None:
        missing = [
            label
            for label in LABELS_ABC
            if label not in self.proportions
        ]
        if missing:
            raise ValueError(
                "Faltan proporciones para las categorías: "
                f"{missing}"
            )

        invalid = {
            label: value
            for label, value in self.proportions.items()
            if label not in LABELS_ABC or value < 0
        }
        if invalid:
            raise ValueError(
                "Se encontraron proporciones inválidas: "
                f"{invalid}"
            )

        total = sum(float(self.proportions[label]) for label in LABELS_ABC)
        if total <= 0:
            raise ValueError(
                "La suma de proporciones debe ser positiva."
            )

    @staticmethod
    def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.empty:
            raise ValueError("X no puede estar vacío.")

        non_numeric = [
            column
            for column in X.columns
            if not pd.api.types.is_numeric_dtype(X[column])
        ]
        if non_numeric:
            raise ValueError(
                "Todas las columnas deben ser numéricas. "
                f"Columnas inválidas: {non_numeric}"
            )

        if X.isna().any().any():
            raise ValueError("X contiene valores faltantes.")

        return X.astype(float).copy()

    @staticmethod
    def _build_results(
        labels: pd.Series,
        scores: pd.Series,
    ) -> pd.DataFrame:
        cluster_map = {"A": 0, "B": 1, "C": 2}

        return pd.DataFrame(
            {
                "categoria": labels,
                "cluster": labels.map(cluster_map).astype(int),
                "score_inicial": scores,
            },
            index=labels.index,
        )