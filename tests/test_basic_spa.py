"""
Tests for basic SPA functionality.
"""
import numpy as np
import pytest

from fastspa import SPA


class TestSPACreation:
    """Test SPA instance creation."""

    def test_spa_creation_basic(self, sample_a_matrix, sample_intensities):
        """Test basic SPA creation."""
        spa = SPA(sample_a_matrix, sample_intensities)
        assert spa.n_sectors == 3
        assert spa.mode == "sector_specific"
        assert spa.satellites == ["intensity"]

    def test_spa_creation_with_sectors(self, sample_a_matrix, sample_intensities, sample_sectors):
        """Test SPA creation with sector names."""
        spa = SPA(sample_a_matrix, sample_intensities, sample_sectors)
        assert spa.sectors == tuple(sample_sectors)

    def test_spa_creation_system_wide(self, sample_a_matrix, sample_intensities):
        """Test system-wide mode creation."""
        spa = SPA(sample_a_matrix, sample_intensities, mode="system_wide")
        assert spa.mode == "system_wide"

    def test_spa_creation_invalid_mode(self, sample_a_matrix, sample_intensities):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be"):
            SPA(sample_a_matrix, sample_intensities, mode="invalid")

    def test_spa_creation_invalid_a_matrix(self, sample_intensities):
        """Test non-square A matrix raises error."""
        A_invalid = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        with pytest.raises(ValueError, match="A matrix must be square"):
            SPA(A_invalid, sample_intensities)


class TestSPAAnalysis:
    """Test SPA analysis methods."""

    def test_analyze_sector_specific(self, sample_spa):
        """Test sector-specific analysis."""
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed
        assert len(paths) > 0
        assert paths.total_intensity > 0
        assert paths.coverage > 0
        assert all(p.contribution >= 0 for p in paths)

    def test_analyze_system_wide(self, sample_a_matrix, sample_intensities, sample_sectors):
        """Test system-wide analysis."""
        spa = SPA(sample_a_matrix, sample_intensities, sample_sectors, mode="system_wide")
        paths = spa.analyze(sector=1, depth=3)  # 1-indexed
        assert len(paths) > 0
        assert paths.coverage > 0

    def test_analyze_by_name(self, sample_spa):
        """Test analysis by sector name."""
        paths = sample_spa.analyze(sector="Agriculture", depth=3)
        assert len(paths) > 0

    def test_analyze_invalid_sector(self, sample_spa):
        """Test invalid sector raises error."""
        with pytest.raises(ValueError):
            sample_spa.analyze(sector="InvalidSector")

    def test_analyze_with_threshold(self, sample_spa):
        """Test analysis with custom threshold."""
        paths_default = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        paths_strict = sample_spa.analyze(sector=1, depth=5, threshold=0.01)
        assert len(paths_strict) <= len(paths_default)

    def test_analyze_max_paths(self, sample_spa):
        """Test max_paths parameter."""
        paths = sample_spa.analyze(sector=1, depth=5, max_paths=5)  # 1-indexed
        # max_paths limits extracted paths, but stage_0 is added afterwards
        # so total can be max_paths + 1
        assert len(paths) <= 6

    def test_analyze_absolute_threshold(self, sample_spa):
        """Test absolute threshold type."""
        paths = sample_spa.analyze(sector=1, depth=5, threshold=1e-6, threshold_type="absolute")  # 1-indexed
        assert len(paths) > 0


class TestSPATotalIntensity:
    """Test total intensity calculations."""

    def test_total_intensity_single(self, sample_spa):
        """Test total intensity for single satellite."""
        total = sample_spa.total_intensity()
        assert len(total) == 3
        assert all(t >= 0 for t in total)

    def test_total_intensity_multi(self, sample_multi_satellite):
        """Test total intensity for multiple satellites."""
        total_ghg = sample_multi_satellite.total_intensity("ghg")
        total_water = sample_multi_satellite.total_intensity("water")
        assert len(total_ghg) == 3
        assert len(total_water) == 3

    def test_total_intensity_invalid_satellite(self, sample_spa):
        """Test invalid satellite name."""
        with pytest.raises(KeyError):
            sample_spa.total_intensity("invalid")


class TestSPARepresentation:
    """Test SPA string representation."""

    def test_repr(self, sample_spa):
        """Test __repr__ output."""
        repr_str = repr(sample_spa)
        assert "SPA" in repr_str
        assert "n_sectors=3" in repr_str
        assert "mode=sector_specific" in repr_str