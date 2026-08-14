import pytest

from app.ml.constraints.capacity_constraint import compute_capacities


def test_compute_capacities_respects_abc_proportions():
    capacities = compute_capacities(
        n_samples=10,
        proportions={"A": 0.20, "B": 0.30, "C": 0.50},
    )
    assert capacities == {"A": 2, "B": 3, "C": 5}


def test_compute_capacities_normalizes_weights_and_distributes_remainder():
    capacities = compute_capacities(
        n_samples=7,
        proportions={"A": 2, "B": 3, "C": 5},
    )
    assert capacities == {"A": 1, "B": 2, "C": 4}
    assert sum(capacities.values()) == 7


@pytest.mark.parametrize("n_samples", [0, -1])
def test_compute_capacities_rejects_non_positive_sample_count(n_samples):
    with pytest.raises(ValueError, match="mayor que 0"):
        compute_capacities(
            n_samples=n_samples,
            proportions={"A": 0.20, "B": 0.30, "C": 0.50},
        )


def test_compute_capacities_rejects_missing_category():
    with pytest.raises(ValueError, match="Faltan proporciones"):
        compute_capacities(
            n_samples=10,
            proportions={"A": 0.40, "B": 0.60},
        )


def test_compute_capacities_rejects_non_positive_total_weight():
    with pytest.raises(ValueError, match="debe ser positiva"):
        compute_capacities(
            n_samples=10,
            proportions={"A": 0.0, "B": 0.0, "C": 0.0},
        )
def test_compute_capacities_uses_largest_remainder_for_four_samples():
    capacities = compute_capacities(
        n_samples=4,
        proportions={"A": 0.20, "B": 0.30, "C": 0.50},
    )

    assert capacities == {"A": 1, "B": 1, "C": 2}
    assert sum(capacities.values()) == 4


def test_compute_capacities_rejects_a_zero_capacity_category():
    with pytest.raises(ValueError, match="Capacidades en cero"):
        compute_capacities(
            n_samples=20,
            proportions={"A": 0.01, "B": 0.49, "C": 0.50},
        )