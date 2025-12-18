"""
Tests for the visualization module.
"""

import pytest
import numpy as np
from fastspa import SPA, icicle_plot, sunburst_plot, sector_contribution_chart


@pytest.fixture
def sample_spa():
    """Create a sample SPA instance for testing."""
    n = 8
    A = np.random.uniform(0, 0.1, size=(n, n))
    np.fill_diagonal(A, 0)
    A = A / (np.sum(A, axis=0) * 1.2)
    
    emissions = np.random.uniform(0.1, 2.0, n)
    sectors = [f"Sector_{i}" for i in range(n)]
    
    return SPA(A, emissions, sectors=sectors)


class TestIciclePlot:
    """Test icicle plot functionality."""
    
    def test_icicle_plot_requires_plotly(self, sample_spa):
        """Test that icicle_plot requires plotly."""
        try:
            import plotly
            # plotly is installed, so we can test the function
            paths = sample_spa.analyze(sector=1, depth=4)
            fig = icicle_plot(paths)
            assert fig is not None
        except ImportError:
            # plotly not installed, test should be skipped
            pytest.skip("plotly not installed")
    
    def test_icicle_plot_with_path_collection(self, sample_spa):
        """Test icicle plot with single PathCollection."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")
        
        paths = sample_spa.analyze(sector=1, depth=3)
        fig = icicle_plot(paths)
        
        assert fig is not None
        assert fig.data[0].type == "icicle"
    
    def test_icicle_plot_with_multiple_satellites(self):
        """Test icicle plot with multiple satellites."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")
        
        n = 6
        A = np.random.uniform(0, 0.1, size=(n, n))
        np.fill_diagonal(A, 0)
        A = A / (np.sum(A, axis=0) * 1.2)
        
        satellites = {
            "ghg": np.random.uniform(0.1, 2.0, n),
            "water": np.random.uniform(0.05, 1.5, n),
        }
        
        spa = SPA(A, satellites, sectors=[f"S_{i}" for i in range(n)])
        result = spa.analyze(sector=1, depth=3)
        
        fig = icicle_plot(result)
        assert fig is not None
    
    def test_icicle_plot_save_html(self, sample_spa, tmp_path):
        """Test icicle plot HTML export."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")
        
        paths = sample_spa.analyze(sector=1, depth=3)
        output_file = tmp_path / "test_icicle.html"
        
        result = icicle_plot(paths, output_html=str(output_file))
        
        # When saving to file, should return None
        assert result is None
        # File should exist and have content
        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestSunburstPlot:
    """Test sunburst plot functionality."""
    
    def test_sunburst_plot_requires_plotly(self, sample_spa):
        """Test that sunburst_plot requires plotly."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")
        
        paths = sample_spa.analyze(sector=1, depth=4)
        fig = sunburst_plot(paths)
        assert fig is not None
    
    def test_sunburst_plot_with_max_depth(self, sample_spa):
        """Test sunburst plot with depth limiting."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")
        
        paths = sample_spa.analyze(sector=1, depth=6)
        fig = sunburst_plot(paths, max_depth=3)
        
        assert fig is not None
        assert fig.data[0].type == "sunburst"


class TestSectorContributionChart:
    """Test sector contribution chart."""
    
    def test_sector_contribution_chart_requires_plotly(self, sample_spa):
        """Test that sector_contribution_chart requires plotly."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")
        
        paths = sample_spa.analyze(sector=1, depth=4)
        fig = sector_contribution_chart(paths)
        assert fig is not None
    
    def test_sector_contribution_chart_top_n(self, sample_spa):
        """Test sector contribution chart with top_n parameter."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")
        
        paths = sample_spa.analyze(sector=1, depth=4)
        fig = sector_contribution_chart(paths, top_n=5)
        
        assert fig is not None
        assert fig.data[0].type == "bar"
    
    def test_sector_contribution_chart_save_html(self, sample_spa, tmp_path):
        """Test sector contribution chart HTML export."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")
        
        paths = sample_spa.analyze(sector=1, depth=3)
        output_file = tmp_path / "test_chart.html"
        
        result = sector_contribution_chart(paths, output_html=str(output_file))
        
        assert result is None
        assert output_file.exists()
        assert output_file.stat().st_size > 0
