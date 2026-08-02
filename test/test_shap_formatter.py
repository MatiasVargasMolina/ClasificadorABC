from __future__ import annotations

import numpy as np
import pytest

from app.ml.explainability.shap_formatter import (
    ADDITIVITY_TOLERANCE,
    build_additivity_diagnostics,
    build_global_importance,
    build_global_importance_by_class,
    build_local_contributions,
    normalize_expected_values,
    normalize_shap_values,
)


FEATURE_COLUMNS = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
]

CLASSES = [
    "A",
    "B",
    "C",
]


def test_normalize_shap_values_accepts_list_by_class():
    shap_raw = [
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
        np.array(
            [
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        ),
        np.array(
            [
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ]
        ),
    ]

    normalized = normalize_shap_values(
        shap_values=shap_raw,
        n_samples=2,
        n_features=3,
        n_classes=3,
    )

    assert normalized.shape == (2, 3, 3)

    np.testing.assert_allclose(
        normalized[0, :, 0],
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        normalized[1, :, 1],
        [10.0, 11.0, 12.0],
    )
    np.testing.assert_allclose(
        normalized[0, :, 2],
        [13.0, 14.0, 15.0],
    )


def test_normalize_shap_values_rejects_invalid_shape():
    shap_raw = np.zeros((2, 5))

    with pytest.raises(
        ValueError,
        match="No fue posible normalizar",
    ):
        normalize_shap_values(
            shap_values=shap_raw,
            n_samples=2,
            n_features=3,
            n_classes=3,
        )


def test_normalize_expected_values_accepts_one_value_per_class():
    expected_values = normalize_expected_values(
        expected_value=[0.20, 0.30, 0.50],
        n_classes=3,
    )

    np.testing.assert_allclose(
        expected_values,
        [0.20, 0.30, 0.50],
    )


def test_normalize_expected_values_rejects_wrong_number_of_classes():
    with pytest.raises(
        ValueError,
        match="no coincide con las clases",
    ):
        normalize_expected_values(
            expected_value=[0.40, 0.60],
            n_classes=3,
        )


def test_build_local_contributions_orders_by_absolute_importance():
    contributions = build_local_contributions(
        feature_columns=FEATURE_COLUMNS,
        feature_values=[2.0, 84.0, 9999.0],
        shap_values=[0.0, -0.40, 0.10],
    )

    assert [
        item["feature"]
        for item in contributions
    ] == [
        "visitas_30d",
        "precio_actual",
        "ventas_30d",
    ]

    assert contributions[0]["direction"] == "baja"
    assert contributions[1]["direction"] == "sube"
    assert contributions[2]["direction"] == "neutro"

    assert contributions[0]["feature_value"] == 84.0
    assert contributions[0]["shap_value"] == -0.40


def test_global_importance_is_calculated_globally_and_by_class():
    shap_values = np.zeros(
        (2, 3, 3),
        dtype=float,
    )

    # Categoría A: predominan las ventas.
    shap_values[:, :, 0] = [
        [3.0, 1.0, 0.0],
        [3.0, 1.0, 0.0],
    ]

    # Categoría B: predominan las visitas.
    shap_values[:, :, 1] = [
        [0.0, 4.0, 1.0],
        [0.0, 4.0, 1.0],
    ]

    # Categoría C: predomina el precio.
    shap_values[:, :, 2] = [
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 5.0],
    ]

    global_importance = build_global_importance(
        feature_columns=FEATURE_COLUMNS,
        shap_values=shap_values,
    )

    importance_by_class = (
        build_global_importance_by_class(
            feature_columns=FEATURE_COLUMNS,
            shap_values=shap_values,
            classes=CLASSES,
        )
    )

    assert [
        item["feature"]
        for item in global_importance
    ] == [
        "precio_actual",
        "visitas_30d",
        "ventas_30d",
    ]

    assert (
        importance_by_class["A"][0]["feature"]
        == "ventas_30d"
    )
    assert (
        importance_by_class["B"][0]["feature"]
        == "visitas_30d"
    )
    assert (
        importance_by_class["C"][0]["feature"]
        == "precio_actual"
    )

    assert (
        importance_by_class["A"][0][
            "mean_abs_shap"
        ]
        == pytest.approx(3.0)
    )
    assert (
        importance_by_class["B"][0][
            "mean_abs_shap"
        ]
        == pytest.approx(4.0)
    )
    assert (
        importance_by_class["C"][0][
            "mean_abs_shap"
        ]
        == pytest.approx(5.0)
    )


def test_additivity_diagnostics_accepts_exact_reconstruction():
    expected_values = np.array(
        [0.20, 0.30, 0.50],
        dtype=float,
    )

    shap_values = np.array(
        [
            [
                [0.10, -0.10, 0.00],
                [0.05, 0.00, -0.05],
                [0.00, 0.10, -0.10],
            ],
            [
                [-0.05, 0.05, 0.00],
                [0.00, -0.05, 0.05],
                [0.05, 0.00, -0.05],
            ],
        ],
        dtype=float,
    )

    model_outputs = (
        expected_values[np.newaxis, :]
        + np.sum(
            shap_values,
            axis=1,
        )
    )

    diagnostics = build_additivity_diagnostics(
        expected_values=expected_values,
        shap_values=shap_values,
        model_outputs=model_outputs,
        classes=CLASSES,
    )

    assert diagnostics["cumple_tolerancia"] is True
    assert diagnostics["tolerance"] == (
        ADDITIVITY_TOLERANCE
    )
    assert diagnostics[
        "mean_absolute_error"
    ] == pytest.approx(0.0)
    assert diagnostics[
        "max_absolute_error"
    ] == pytest.approx(0.0)

    for class_name in CLASSES:
        assert diagnostics["por_clase"][
            class_name
        ]["max_absolute_error"] == pytest.approx(
            0.0
        )


def test_additivity_diagnostics_detects_tolerance_violation():
    expected_values = np.array(
        [0.20, 0.30, 0.50],
        dtype=float,
    )

    shap_values = np.zeros(
        (1, 3, 3),
        dtype=float,
    )

    model_outputs = np.array(
        [[0.21, 0.30, 0.49]],
        dtype=float,
    )

    diagnostics = build_additivity_diagnostics(
        expected_values=expected_values,
        shap_values=shap_values,
        model_outputs=model_outputs,
        classes=CLASSES,
    )

    assert diagnostics["cumple_tolerancia"] is False
    assert diagnostics[
        "max_absolute_error"
    ] == pytest.approx(0.01)


def test_additivity_diagnostics_rejects_incompatible_outputs():
    expected_values = np.array(
        [0.20, 0.30, 0.50],
        dtype=float,
    )
    shap_values = np.zeros(
        (2, 3, 3),
        dtype=float,
    )
    invalid_outputs = np.zeros(
        (2, 2),
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="no tienen la misma forma",
    ):
        build_additivity_diagnostics(
            expected_values=expected_values,
            shap_values=shap_values,
            model_outputs=invalid_outputs,
            classes=CLASSES,
        )