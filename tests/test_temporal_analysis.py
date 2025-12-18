"""Tests for time-series SPA workflows."""

from fastspa import SPA, TemporalSPAResult


def test_analyze_time_series(sample_a_matrix, sample_intensities, sample_sectors):
    series = {
        2020: (sample_a_matrix, sample_intensities),
        2021: (sample_a_matrix, sample_intensities * 1.1),
    }

    result = SPA.analyze_time_series(series, sector=1, depth=3, sectors=sample_sectors)

    assert isinstance(result, TemporalSPAResult)
    assert set(result.results.keys()) == {2020, 2021}

    totals = result.total_intensity_series()
    assert totals[2021] > totals[2020]
