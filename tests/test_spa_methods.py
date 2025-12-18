"""
Tests for SPA analysis methods.
"""
import pytest

from fastspa import SPAResult


class TestAnalyzeMany:
    """Test analyze_many method."""

    def test_analyze_many_basic(self, sample_spa):
        """Test analyzing multiple sectors."""
        sectors = [1, 2, 3]  # 1-indexed
        results = sample_spa.analyze_many(sectors, depth=3)
        assert len(results) == 3
        assert all(isinstance(v, type(sample_spa.analyze(1, depth=3))) for v in results.values())

    def test_analyze_many_with_names(self, sample_spa):
        """Test analyze_many with sector names."""
        sectors = ["Agriculture", "Manufacturing"]
        results = sample_spa.analyze_many(sectors, depth=3)
        assert len(results) == 2
        assert all(isinstance(v, type(sample_spa.analyze(1, depth=3))) for v in results.values())


class TestHotspots:
    """Test hotspots method."""

    def test_hotspots_basic(self, sample_spa):
        """Test basic hotspots identification."""
        hotspots = sample_spa.hotspots(sector=1, depth=5, top_n=5)  # 1-indexed
        assert len(hotspots) <= 5
        for idx, contribution, name in hotspots:
            assert isinstance(idx, int)
            assert isinstance(contribution, float)
            assert isinstance(name, str)
            assert contribution >= 0

    def test_hotspots_with_names(self, sample_spa):
        """Test hotspots with sector names."""
        hotspots = sample_spa.hotspots(sector="Agriculture", depth=5, top_n=3)
        assert len(hotspots) <= 3
        # Should have proper names
        names = [name for _, _, name in hotspots]
        assert "Agriculture" in names or any("Sector_" in name for name in names)


class TestCompareSectors:
    """Test compare_sectors method."""

    def test_compare_sectors_basic(self, sample_spa):
        """Test basic sector comparison."""
        pd = pytest.importorskip("pandas")
        sectors = [1, 2, 3]  # 1-indexed
        df = sample_spa.compare_sectors(sectors, depth=3)
        assert len(df) == 3
        assert "sector" in df.columns
        assert "total_intensity" in df.columns
        assert "num_paths" in df.columns
        assert "coverage" in df.columns

    def test_compare_sectors_with_names(self, sample_spa):
        """Test sector comparison with names."""
        pd = pytest.importorskip("pandas")
        sectors = ["Agriculture", "Manufacturing"]
        df = sample_spa.compare_sectors(sectors, depth=3)
        assert len(df) == 2
        assert all(sector in ["Agriculture", "Manufacturing"] for sector in df["sector"].values)

    def test_compare_sectors_multi_satellite(self, sample_multi_satellite):
        """Test sector comparison with multiple satellites."""
        pd = pytest.importorskip("pandas")
        sectors = [1, 2]  # 1-indexed
        df = sample_multi_satellite.compare_sectors(sectors, depth=3, satellite="ghg")
        assert len(df) == 2


class TestStream:
    """Test stream method."""

    def test_stream_basic(self, sample_spa):
        """Test basic streaming."""
        stream_iter = sample_spa.stream(sector=1, depth=5, threshold=0.001)  # 1-indexed
        paths = list(stream_iter)
        assert len(paths) > 0
        assert all(p.contribution >= 0 for p in paths)

    def test_stream_with_threshold(self, sample_spa):
        """Test streaming with threshold."""
        # Get results with regular analyze for comparison
        paths_analyzed = sample_spa.analyze(sector=1, depth=5, threshold=0.001)  # 1-indexed
        paths_streamed = list(sample_spa.stream(sector=1, depth=5, threshold=0.001))

        # Should have similar or fewer results (stream doesn't include stage_0)
        assert len(paths_streamed) <= len(paths_analyzed) + 1

    def test_stream_absolute_threshold(self, sample_spa):
        """Test streaming with absolute threshold."""
        paths = list(sample_spa.stream(sector=1, depth=5, threshold=1e-6, threshold_type="absolute"))  # 1-indexed
        assert len(paths) > 0
        # All paths should have positive contribution
        assert all(p.contribution >= 0 for p in paths)

    def test_stream_multi_satellite(self, sample_multi_satellite):
        """Test streaming with specific satellite."""
        paths = list(sample_multi_satellite.stream(sector=1, depth=5, satellite="ghg"))  # 1-indexed
        assert len(paths) > 0

    def test_stream_memory_efficiency(self, sample_spa):
        """Test that streaming doesn't load all paths at once."""
        # This is more of a conceptual test - in practice we'd need a larger system
        # to truly test memory efficiency, but we can verify it returns an iterator
        stream_iter = sample_spa.stream(sector=1, depth=5)  # 1-indexed
        assert hasattr(stream_iter, '__iter__')
        assert hasattr(stream_iter, '__next__')

        # Get first few paths
        first_paths = []
        for i, path in enumerate(stream_iter):
            first_paths.append(path)
            if i >= 2:  # Just get 3 paths
                break

        assert len(first_paths) == 3
        assert all(isinstance(p, type(first_paths[0])) for p in first_paths)