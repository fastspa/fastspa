"""Tests for Monte Carlo uncertainty quantification and sensitivity."""

import pytest

from fastspa import MonteCarloResult


def test_sensitivity_elasticities_sum_to_one(sample_spa):
    sens = sample_spa.sensitivity(sector=1)
    assert set(sens.keys()) == {0, 1, 2}
    assert pytest.approx(sum(sens.values()), rel=1e-12) == 1.0


def test_monte_carlo_zero_uncertainty(sample_spa):
    baseline_total = float(sample_spa.total_intensity()[0])

    res = sample_spa.monte_carlo(
        sector=1,
        depth=3,
        n_samples=30,
        intensity_cv=0.0,
        seed=123,
    )

    assert isinstance(res, MonteCarloResult)
    assert res.total_intensity.std == pytest.approx(0.0, abs=1e-12)
    assert res.total_intensity.mean == pytest.approx(baseline_total, rel=1e-12)


def test_monte_carlo_nonzero_uncertainty(sample_spa):
    res = sample_spa.monte_carlo(
        sector=1,
        depth=3,
        n_samples=60,
        intensity_cv=0.2,
        seed=42,
    )

    assert res.total_intensity.std > 0


def test_monte_carlo_lhs(sample_spa):
    res = sample_spa.monte_carlo(
        sector=1,
        depth=3,
        n_samples=25,
        intensity_cv=0.2,
        seed=42,
        sampling="lhs",
    )

    assert res.samples_total_intensity.shape == (25,)
    assert res.samples_sector_contributions.shape == (25, 3)
