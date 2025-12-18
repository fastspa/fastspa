"""Tests for consequential (perturbation) mode."""

import numpy as np


def test_consequential_intensity_perturbation(sample_spa):
    # Increase direct intensity in sector 2 and expect higher total intensity
    mult = np.array([1.0, 1.0, 1.2])

    res = sample_spa.consequential(sector=1, depth=3, intensity_multiplier=mult)

    assert res.delta_total_intensity > 0
    assert res.baseline.target_sector == 0
    assert res.scenario.target_sector == 0
    assert res.satellite == "intensity"
