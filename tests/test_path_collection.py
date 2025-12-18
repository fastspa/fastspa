"""
Tests for PathCollection methods.
"""
import pytest
import tempfile
import os
import json

from fastspa import Path, SPA


class TestPathCollectionBasic:
    """Test basic PathCollection functionality."""

    def test_path_collection_creation(self, sample_spa):
        """Test PathCollection creation."""
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed
        assert len(paths) > 0
        assert paths.target_sector == 0  # Internally 0-indexed
        assert paths.total_intensity > 0
        assert paths.coverage > 0

    def test_path_collection_iteration(self, sample_spa):
        """Test PathCollection iteration."""
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed
        path_list = list(paths)
        assert len(path_list) == len(paths)
        assert all(isinstance(p, Path) for p in path_list)

    def test_path_collection_slicing(self, sample_spa):
        """Test PathCollection slicing."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        if len(paths) >= 3:
            subset = paths[:3]
            assert len(subset) == 3
            assert isinstance(subset, type(paths))

    def test_path_collection_top(self, sample_spa):
        """Test top() method."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        top_5 = paths.top(5)
        assert len(top_5) <= 5
        # Should be sorted by contribution
        contributions = [p.contribution for p in top_5]
        assert contributions == sorted(contributions, reverse=True)


class TestPathCollectionFiltering:
    """Test PathCollection filtering methods."""

    def test_filter_min_contribution(self, sample_spa):
        """Test filtering by minimum contribution."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        filtered = paths.filter(min_contribution=0.01)
        assert all(p.contribution >= 0.01 for p in filtered)

    def test_filter_max_depth(self, sample_spa):
        """Test filtering by maximum depth."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        filtered = paths.filter(max_depth=2)
        assert all(p.depth <= 2 for p in filtered)

    def test_filter_contains_sector(self, sample_spa):
        """Test filtering by sector containment."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        if len(paths) > 0:
            # Find a sector that appears in some paths
            all_sectors = set()
            for path in paths:
                all_sectors.update(path.nodes)
            if all_sectors:
                sector_to_filter = list(all_sectors)[0]
                filtered = paths.filter(contains_sector=sector_to_filter)
                assert all(sector_to_filter in p.nodes for p in filtered)

    def test_filter_predicate(self, sample_spa):
        """Test custom predicate filtering."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        # Filter paths with depth > 1
        filtered = paths.filter(predicate=lambda p: p.depth > 1)
        assert all(p.depth > 1 for p in filtered)

    def test_filter_combined(self, sample_spa):
        """Test combined filtering."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        filtered = paths.filter(min_contribution=0.001, max_depth=3)
        assert all(p.contribution >= 0.001 and p.depth <= 3 for p in filtered)


class TestPathCollectionAggregation:
    """Test PathCollection aggregation methods."""

    def test_aggregate_by_sector(self, sample_spa):
        """Test sector aggregation."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        aggregated = paths.aggregate_by_sector()
        assert isinstance(aggregated, dict)
        assert all(isinstance(k, int) and isinstance(v, float) for k, v in aggregated.items())

    def test_group_by_stage(self, sample_spa):
        """Test grouping by stage."""
        paths = sample_spa.analyze(sector=1, depth=5)  # 1-indexed
        groups = paths.group_by_stage(1)  # First-tier suppliers
        assert isinstance(groups, dict)
        for sector_idx, path_collection in groups.items():
            assert isinstance(sector_idx, int)
            # Check that paths in each group have that sector at stage 1
            for p in path_collection:
                if len(p.nodes) > 1:
                    assert p.nodes[1] == sector_idx


class TestPathCollectionExport:
    """Test PathCollection export methods."""

    def test_to_dataframe(self, sample_spa):
        """Test DataFrame export."""
        pd = pytest.importorskip("pandas")
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed
        df = paths.to_dataframe()
        assert len(df) == len(paths)
        assert "contribution" in df.columns
        assert "depth" in df.columns

    def test_to_csv(self, sample_spa):
        """Test CSV export."""
        pd = pytest.importorskip("pandas")
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
            try:
                paths.to_csv(tmp.name)
                # Verify file was created and has content
                assert os.path.exists(tmp.name)
                with open(tmp.name, 'r') as f:
                    content = f.read()
                    assert len(content) > 0
            finally:
                os.unlink(tmp.name)

    def test_to_json(self, sample_spa):
        """Test JSON export."""
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed

        # Test string output
        json_str = paths.to_json()
        assert json_str is not None
        data = json.loads(json_str)
        assert "paths" in data
        assert "metadata" in data

        # Test file output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
            try:
                paths.to_json(tmp.name)
                assert os.path.exists(tmp.name)
                with open(tmp.name, 'r') as f:
                    data = json.load(f)
                    assert "paths" in data
            finally:
                os.unlink(tmp.name)

    def test_summary(self, sample_spa):
        """Test summary generation."""
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed
        summary = paths.summary()
        assert isinstance(summary, str)
        assert "Structural Path Analysis Results" in summary
        assert "Top 10 paths:" in summary


class TestPathProperties:
    """Test Path object properties."""

    def test_path_properties(self, sample_spa):
        """Test Path object attributes."""
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed
        if len(paths) > 0:
            path = paths[0]
            assert isinstance(path.nodes, tuple)
            assert isinstance(path.contribution, float)
            assert isinstance(path.direct_intensity, float)
            assert isinstance(path.cumulative_weight, float)
            assert path.contribution >= 0
            assert path.depth >= 0
            assert path.root == 0  # Internally 0-indexed target sector
            assert path.leaf in path.nodes

    def test_path_sectors_with_names(self, sample_spa):
        """Test sector names in paths."""
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed
        if len(paths) > 0:
            path = paths[0]
            sectors = path.sectors
            assert len(sectors) == len(path.nodes)
            assert sectors[0] == "Agriculture"  # Target sector name

    def test_path_sectors_without_names(self, sample_a_matrix, sample_intensities):
        """Test sector names default when not provided."""
        spa = SPA(sample_a_matrix, sample_intensities)  # No sector names
        paths = spa.analyze(sector=1, depth=3)  # 1-indexed
        if len(paths) > 0:
            path = paths[0]
            sectors = path.sectors
            assert all(s.startswith("Sector_") for s in sectors)

    def test_path_str_representation(self, sample_spa):
        """Test Path string representation."""
        paths = sample_spa.analyze(sector=1, depth=3)  # 1-indexed
        if len(paths) > 0:
            path = paths[0]
            str_repr = str(path)
            # Check percentage format and sector name
            assert "%" in str_repr  # Percentage format
            assert "Agriculture" in str_repr or "→" in str_repr or "Sector_" in str_repr