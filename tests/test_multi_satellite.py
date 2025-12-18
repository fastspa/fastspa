"""
Tests for multi-satellite analysis.
"""
import numpy as np
import pytest

from fastspa import SPA, SPAResult


class TestMultiSatelliteCreation:
    """Test creation with multiple satellites."""

    def test_multi_satellite_creation(self, sample_a_matrix, sample_sectors):
        """Test SPA creation with multiple satellites."""
        intensities = {
            "ghg": np.array([0.5, 1.0, 0.8]),
            "water": np.array([10.0, 5.0, 15.0])
        }
        spa = SPA(sample_a_matrix, intensities, sample_sectors)
        assert spa.satellites == ["ghg", "water"]

    def test_multi_satellite_analysis(self, sample_multi_satellite):
        """Test analysis returns SPAResult for multiple satellites."""
        result = sample_multi_satellite.analyze(sector=1, depth=3)  # 1-indexed
        assert isinstance(result, SPAResult)
        assert "ghg" in result.satellites
        assert "water" in result.satellites
        assert "energy" in result.satellites

    def test_multi_satellite_single_analysis(self, sample_multi_satellite):
        """Test analysis of single satellite."""
        result = sample_multi_satellite.analyze(sector=1, depth=3, satellite="ghg")  # 1-indexed
        assert not isinstance(result, SPAResult)
        assert result.satellite_name == "ghg"

    def test_spa_result_access(self, sample_multi_satellite):
        """Test SPAResult access methods."""
        result = sample_multi_satellite.analyze(sector=1, depth=3)  # 1-indexed
        assert isinstance(result, SPAResult)

        # Test indexing
        ghg_paths = result["ghg"]
        assert ghg_paths.satellite_name == "ghg"

        # Test iteration
        satellites = list(result)
        assert set(satellites) == {"ghg", "water", "energy"}

        # Test items
        items = dict(result.items())
        assert len(items) == 3

    def test_spa_result_to_dataframe(self, sample_multi_satellite):
        """Test SPAResult to_dataframe method."""
        pytest.importorskip("pandas")
        result = sample_multi_satellite.analyze(sector=1, depth=3)  # 1-indexed
        df = result.to_dataframe()
        assert len(df) > 0
        assert "satellite" in df.columns