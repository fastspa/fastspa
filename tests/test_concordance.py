"""
Tests for sector concordance features.
"""
import numpy as np
import pytest

from fastspa import SPA, SPAResult, SectorConcordance, SatelliteWithConcordance
from fastspa.concordance import ConcordanceMetadata


class TestSectorConcordance:
    """Test SectorConcordance functionality."""

    def test_concordance_creation(self, sample_concordance):
        """Test concordance creation."""
        assert len(sample_concordance.source_sectors) == 3
        assert len(sample_concordance.target_sectors) == 3
        assert sample_concordance.matrix.shape == (3, 3)

    def test_concordance_allocate(self, sample_concordance):
        """Test concordance allocation."""
        source_values = np.array([100, 200, 150])  # Source sector values
        target_values = sample_concordance.allocate(source_values)
        assert len(target_values) == 3
        assert np.allclose(target_values.sum(), source_values.sum())  # Conservation of mass

    def test_concordance_from_supply_use_table(self):
        """Test creating concordance from supply-use table data."""
        pd = pytest.importorskip("pandas")

        # Mock supply-use data as DataFrame
        sut_data = pd.DataFrame([
            {"anzsic": "A", "ioig": "Agriculture", "flow": 100},
            {"anzsic": "A", "ioig": "Manufacturing", "flow": 20},
            {"anzsic": "B", "ioig": "Agriculture", "flow": 80},
            {"anzsic": "C", "ioig": "Manufacturing", "flow": 150},
        ])

        concordance = SectorConcordance.from_supply_use_table(
            sut_data,
            source_col="anzsic",
            target_col="ioig",
            value_col="flow",
            metadata=ConcordanceMetadata(
                source="Test SUT",
                method="supply_use_table",
                year=2023
            )
        )

        assert len(concordance.source_sectors) == 3  # A, B, C
        assert len(concordance.target_sectors) == 2  # Agriculture, Manufacturing


class TestSatelliteWithConcordance:
    """Test SatelliteWithConcordance functionality."""

    def test_satellite_concordance_creation(self, sample_satellite_with_concordance):
        """Test satellite with concordance creation."""
        satellite = sample_satellite_with_concordance
        assert satellite.name == "ghg_concordance"
        assert satellite.unit == "kg CO2-e"
        assert len(satellite.intensities_source) == 3  # Source intensities

    def test_satellite_concordance_get_intensities(self, sample_satellite_with_concordance):
        """Test getting target intensities."""
        target_intensities = sample_satellite_with_concordance.get_intensities_target()
        assert len(target_intensities) == 3  # Target sectors
        assert all(i >= 0 for i in target_intensities)


class TestSPAWithConcordance:
    """Test SPA with concordance satellites."""

    def test_spa_with_single_concordance(self, sample_a_matrix, sample_satellite_with_concordance):
        """Test SPA with single concordance satellite."""
        spa = SPA(sample_a_matrix, sample_satellite_with_concordance)
        assert spa.satellites == ["ghg_concordance"]
        assert spa.satellite_objects is not None

        # Test analysis
        paths = spa.analyze(sector=1, depth=3)  # 1-indexed
        assert len(paths) > 0

    def test_spa_with_multiple_concordance(self, sample_a_matrix, sample_concordance):
        """Test SPA with multiple concordance satellites."""
        satellites = {
            "ghg": SatelliteWithConcordance(
                intensities=np.array([0.6, 0.4, 1.2]),
                concordance=sample_concordance,
                name="ghg",
                unit="kg CO2-e"
            ),
            "water": SatelliteWithConcordance(
                intensities=np.array([50.0, 30.0, 80.0]),
                concordance=sample_concordance,
                name="water",
                unit="L"
            )
        }

        spa = SPA(sample_a_matrix, satellites)
        assert set(spa.satellites) == {"ghg", "water"}

        # Test analysis
        result = spa.analyze(sector=1, depth=3)  # 1-indexed
        assert isinstance(result, SPAResult)

    def test_spa_concordance_sector_alignment(self, sample_a_matrix, sample_concordance, sample_satellite_with_concordance):
        """Test sector alignment validation."""
        # Mismatched sectors should raise error
        wrong_sectors = ["Wrong1", "Wrong2", "Wrong3"]
        with pytest.raises(ValueError, match="Sector names must match"):
            SPA(sample_a_matrix, sample_satellite_with_concordance, wrong_sectors)