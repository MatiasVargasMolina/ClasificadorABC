from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


ADDITIVITY_TOLERANCE = 1e-4
NEUTRAL_TOLERANCE = 1e-12


def normalize_shap_values(
    shap_values: Any,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """
    Normaliza la salida SHAP a:

        (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        arr = np.stack(
            [
                np.asarray(values, dtype=float)
                for values in shap_values
            ],
            axis=-1,
        )
    else:
        arr = np.asarray(
            shap_values,
            dtype=float,
        )

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if arr.ndim != 3:
        raise ValueError(
            "La salida SHAP debe tener dos o tres dimensiones. "
            f"Forma recibida: {arr.shape}"
        )

    expected_shape = (
        n_samples,
        n_features,
        n_classes,
    )

    if arr.shape == expected_shape:
        return arr

    if arr.shape == (
        n_classes,
        n_samples,
        n_features,
    ):
        return np.transpose(
            arr,
            (1, 2, 0),
        )

    if arr.shape == (
        n_samples,
        n_classes,
        n_features,
    ):
        return np.transpose(
            arr,
            (0, 2, 1),
        )

    raise ValueError(
        "No fue posible normalizar la salida SHAP. "
        f"Forma recibida: {arr.shape}; "
        f"forma esperada: {expected_shape}."
    )


def normalize_expected_values(
    expected_value: Any,
    n_classes: int,
) -> np.ndarray:
    values = np.asarray(
        expected_value,
        dtype=float,
    ).reshape(-1)

    if values.size == n_classes:
        return values

    if values.size == 1 and n_classes == 1:
        return values

    raise ValueError(
        "La cantidad de valores base SHAP no coincide con las clases. "
        f"Valores base: {values.size}; clases: {n_classes}."
    )


def _direction(
    shap_value: float,
) -> str:
    if shap_value > NEUTRAL_TOLERANCE:
        return "sube"

    if shap_value < -NEUTRAL_TOLERANCE:
        return "baja"

    return "neutro"


def build_local_contributions(
    feature_columns: Sequence[str],
    feature_values: Sequence[float],
    shap_values: Sequence[float],
) -> List[Dict[str, Any]]:
    contributions: List[Dict[str, Any]] = []

    for feature, feature_value, shap_value in zip(
        feature_columns,
        feature_values,
        shap_values,
    ):
        numeric_shap_value = float(shap_value)

        contributions.append(
            {
                "feature": str(feature),
                "feature_value": float(feature_value),
                "shap_value": numeric_shap_value,
                "direction": _direction(
                    numeric_shap_value
                ),
            }
        )

    return sorted(
        contributions,
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )


def _importance_for_matrix(
    feature_columns: Sequence[str],
    values: np.ndarray,
) -> List[Dict[str, float]]:
    importance: List[Dict[str, float]] = []

    for feature_index, feature_name in enumerate(
        feature_columns
    ):
        mean_abs = float(
            np.mean(
                np.abs(
                    values[:, feature_index]
                )
            )
        )

        importance.append(
            {
                "feature": str(feature_name),
                "mean_abs_shap": mean_abs,
            }
        )

    return sorted(
        importance,
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )


def build_global_importance(
    feature_columns: Sequence[str],
    shap_values: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Importancia general calculada mediante mean(abs(SHAP))
    sobre todas las publicaciones y categorías.
    """
    mean_over_classes = np.mean(
        np.abs(shap_values),
        axis=2,
    )

    return _importance_for_matrix(
        feature_columns=feature_columns,
        values=mean_over_classes,
    )


def build_global_importance_by_class(
    feature_columns: Sequence[str],
    shap_values: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, List[Dict[str, float]]]:
    importance_by_class: Dict[
        str,
        List[Dict[str, float]],
    ] = {}

    for class_index, class_name in enumerate(classes):
        importance_by_class[str(class_name)] = (
            _importance_for_matrix(
                feature_columns=feature_columns,
                values=shap_values[
                    :,
                    :,
                    class_index,
                ],
            )
        )

    return importance_by_class


def build_additivity_diagnostics(
    expected_values: np.ndarray,
    shap_values: np.ndarray,
    model_outputs: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, Any]:
    """
    Comprueba la propiedad de aditividad:

        salida_modelo =
        valor_base + suma(contribuciones_SHAP)
    """
    outputs = np.asarray(
        model_outputs,
        dtype=float,
    )

    reconstructed_outputs = (
        expected_values[np.newaxis, :]
        + np.sum(
            shap_values,
            axis=1,
        )
    )

    if reconstructed_outputs.shape != outputs.shape:
        raise ValueError(
            "Las salidas reconstruidas y las probabilidades del modelo "
            "no tienen la misma forma. "
            f"Reconstruidas: {reconstructed_outputs.shape}; "
            f"modelo: {outputs.shape}."
        )

    absolute_errors = np.abs(
        reconstructed_outputs - outputs
    )

    diagnostics_by_class: Dict[
        str,
        Dict[str, float],
    ] = {}

    for class_index, class_name in enumerate(classes):
        class_errors = absolute_errors[
            :,
            class_index,
        ]

        diagnostics_by_class[str(class_name)] = {
            "mean_absolute_error": float(
                np.mean(class_errors)
            ),
            "max_absolute_error": float(
                np.max(class_errors)
            ),
        }

    max_error = float(
        np.max(absolute_errors)
    )

    return {
        "mean_absolute_error": float(
            np.mean(absolute_errors)
        ),
        "max_absolute_error": max_error,
        "tolerance": ADDITIVITY_TOLERANCE,
        "cumple_tolerancia": (
            max_error <= ADDITIVITY_TOLERANCE
        ),
        "por_clase": diagnostics_by_class,
    }