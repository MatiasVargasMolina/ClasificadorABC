from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def normalize_shap_values(shap_values: Any) -> np.ndarray:
    """
    Normaliza la salida de SHAP para clasificación multiclase.

    SHAP puede devolver:
    - list[np.ndarray]: una matriz por clase
    - np.ndarray 2D
    - np.ndarray 3D

    Se normaliza a:
    (n_samples, n_features, n_classes)
    """

    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=-1)

    arr = np.asarray(shap_values)

    if arr.ndim == 2:
        return arr[:, :, np.newaxis]

    return arr


def build_local_contributions(
    feature_columns: List[str],
    feature_values: List[float],
    shap_values: List[float],
) -> List[Dict[str, Any]]:
    """
    Arma una lista ordenada de contribuciones SHAP locales.
    """

    contributions: List[Dict[str, Any]] = []

    for feature, feature_value, shap_value in zip(
        feature_columns,
        feature_values,
        shap_values,
    ):
        contributions.append(
            {
                "feature": feature,
                "feature_value": float(feature_value),
                "shap_value": float(shap_value),
                "direction": "sube" if shap_value > 0 else "baja",
            }
        )

    return sorted(
        contributions,
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )


def build_global_importance(
    feature_columns: List[str],
    shap_values: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Calcula importancia global promedio usando mean(abs(SHAP)).
    """

    importance: List[Dict[str, float]] = []

    for feature_index, feature_name in enumerate(feature_columns):
        if shap_values.shape[-1] == 1:
            mean_abs = np.mean(np.abs(shap_values[:, feature_index, 0]))
        else:
            mean_abs = np.mean(np.abs(shap_values[:, feature_index, :]))

        importance.append(
            {
                "feature": feature_name,
                "mean_abs_shap": float(mean_abs),
            }
        )

    return sorted(
        importance,
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )