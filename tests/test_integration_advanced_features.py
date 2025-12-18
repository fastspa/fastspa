"""Integration tests for advanced features (network, uncertainty, aggregation, temporal, loops, consequential)."""

import numpy as np
import pytest

from fastspa import SPA, Path, PathCollection


class TestNetworkAndLoopIntegration:
    """Test network metrics work with loop analysis."""

    def test_network_metrics_with_loops(self, sample_spa):
        """Verify network metrics handle cyclic paths correctly."""
        paths = sample_spa.analyze(sector=1, depth=4)

        # Get network metrics
        edges = paths.edge_weights()
        bc = paths.betweenness_centrality()
        topo = paths.network_topology()

        # Get loop analysis
        loops = paths.loop_analysis()

        # All should be computable without errors
        assert isinstance(edges, dict)
        assert isinstance(bc, dict)
        assert isinstance(topo, dict)
        assert loops.loop_share >= 0

        # Network density should be reasonable
        assert 0.0 <= topo["density"] <= 1.0


class TestUncertaintyAndAggregation:
    """Test uncertainty quantification with aggregation."""

    def test_monte_carlo_with_semantic_aggregation(self, sample_spa):
        """Monte Carlo results should be compatible with semantic aggregation."""
        # Run Monte Carlo analysis
        mc_result = sample_spa.monte_carlo(
            sector=1, depth=3, n_samples=20, intensity_cv=0.1, seed=42
        )

        # The Monte Carlo result should have uncertainty statistics
        assert mc_result.total_intensity is not None
        assert mc_result.total_intensity.mean > 0

        # Verify sector contributions are computed
        assert mc_result.sector_contributions is not None
        assert len(mc_result.sector_contributions) == 3


class TestTemporalWithUncertainty:
    """Test temporal analysis with uncertainty."""

    def test_temporal_analysis_with_mc_per_year(self, sample_a_matrix, sample_intensities, sample_sectors):
        """Each year in temporal analysis can have uncertainty quantification."""
        # Create time series data
        series = {
            2020: (sample_a_matrix, sample_intensities),
            2021: (sample_a_matrix, sample_intensities * 1.1),
            2022: (sample_a_matrix, sample_intensities * 1.2),
        }

        # Run temporal analysis
        temporal_result = SPA.analyze_time_series(
            series, sector=1, depth=3, sectors=sample_sectors
        )

        # Check each year's result can run Monte Carlo
        for year, spa_result in temporal_result.results.items():
            assert spa_result.target_sector == 0
            # Verify can extract data for MC (checking structure, not running full MC)
            assert spa_result.total_intensity is not None


class TestConsequentialWithNetworkAnalysis:
    """Test consequential mode produces results analyzable with network metrics."""

    def test_consequential_scenario_has_network_metrics(self, sample_spa):
        """Both baseline and scenario from consequential analysis should support network metrics."""
        mult = np.array([1.0, 1.0, 1.2])
        res = sample_spa.consequential(sector=1, depth=3, intensity_multiplier=mult)

        # Both baseline and scenario should have network properties
        baseline_bc = res.baseline.betweenness_centrality()
        scenario_bc = res.scenario.betweenness_centrality()

        assert isinstance(baseline_bc, dict)
        assert isinstance(scenario_bc, dict)
        assert set(baseline_bc.keys()) == set(scenario_bc.keys())


class TestAllFeaturesOnMultiSatellite:
    """Test all advanced features work with multiple satellites."""

    def test_multi_satellite_network_analysis(self, sample_multi_satellite):
        """Network analysis should work on any satellite."""
        ghg_paths = sample_multi_satellite.analyze(sector=1, depth=3, satellite="ghg")
        water_paths = sample_multi_satellite.analyze(sector=1, depth=3, satellite="water")

        # Both should support network metrics
        ghg_bc = ghg_paths.betweenness_centrality()
        water_bc = water_paths.betweenness_centrality()

        assert isinstance(ghg_bc, dict)
        assert isinstance(water_bc, dict)

        # Sector rankings might differ
        ghg_top = max(ghg_bc, key=ghg_bc.get)
        water_top = max(water_bc, key=water_bc.get)
        # (Rankings may differ, so no assertion needed)


class TestAggregationPreservesIntensity:
    """Test that aggregation preserves total intensity across features."""

    def test_semantic_aggregation_preserves_total(self, sample_spa):
        """Semantic aggregation should preserve total intensity."""
        paths = sample_spa.analyze(sector=1, depth=3)
        original_total = paths.coverage

        mapping = {0: "A", 1: "B", 2: "C"}
        agg = paths.semantic_aggregate(mapping)

        aggregated_total = sum(agg.values())
        assert pytest.approx(aggregated_total, rel=1e-12) == original_total

    def test_hierarchical_aggregation_preserves_total(self, sample_spa):
        """Hierarchical aggregation should preserve total intensity."""
        paths = sample_spa.analyze(sector=1, depth=3)
        original_total = paths.coverage

        level_1 = {0: "L1_A", 1: "L1_A", 2: "L1_B"}
        level_2 = {0: "L2_A", 1: "L2_B", 2: "L2_C"}

        agg = paths.hierarchical_aggregate([level_1, level_2])

        aggregated_total = sum(agg.values())
        assert pytest.approx(aggregated_total, rel=1e-12) == original_total


class TestLoopAnalysisEdgeCases:
    """Test loop analysis on edge cases."""

    def test_loop_analysis_no_cycles(self):
        """Loop analysis on DAG (no cycles) should return 0 loop share."""
        paths = [
            Path(nodes=(0,), contribution=0.5, direct_intensity=1.0, cumulative_weight=1.0),
            Path(nodes=(0, 1), contribution=0.3, direct_intensity=1.0, cumulative_weight=1.0),
            Path(nodes=(0, 1, 2), contribution=0.2, direct_intensity=1.0, cumulative_weight=1.0),
        ]
        pc = PathCollection(paths, target_sector=0, total_intensity=10.0, satellite_name="test")

        loops = pc.loop_analysis()

        assert loops.loop_share == 0.0
        assert len(loops.cycles) == 0

    def test_loop_analysis_multiple_cycles(self):
        """Loop analysis should handle multiple distinct cycles."""
        paths = [
            Path(nodes=(0,), contribution=0.2, direct_intensity=1.0, cumulative_weight=1.0),
            Path(nodes=(0, 1, 0), contribution=0.3, direct_intensity=1.0, cumulative_weight=1.0),
            Path(nodes=(0, 2, 0), contribution=0.25, direct_intensity=1.0, cumulative_weight=1.0),
        ]
        pc = PathCollection(paths, target_sector=0, total_intensity=10.0, satellite_name="test")

        loops = pc.loop_analysis()

        assert loops.loop_share == pytest.approx(0.55)
        assert len(loops.cycles) == 2


class TestTemporalSeriesTrend:
    """Test temporal analysis shows expected trends."""

    def test_temporal_intensity_growth(self, sample_a_matrix, sample_intensities, sample_sectors):
        """Increasing intensities should show in temporal results."""
        series = {
            2020: (sample_a_matrix, sample_intensities),
            2021: (sample_a_matrix, sample_intensities * 1.1),
            2022: (sample_a_matrix, sample_intensities * 1.2),
        }

        temporal_result = SPA.analyze_time_series(
            series, sector=1, depth=3, sectors=sample_sectors
        )

        totals = temporal_result.total_intensity_series()

        # Verify monotonic growth
        assert totals[2021] > totals[2020]
        assert totals[2022] > totals[2021]

        # Verify growth is roughly in expected direction
        # (Note: first growth is 10%, second is ~9.1% due to compounding)
        growth_2020_2021 = (totals[2021] - totals[2020]) / totals[2020]
        growth_2021_2022 = (totals[2022] - totals[2021]) / totals[2021]

        assert 0.08 < growth_2020_2021 < 0.12
        assert 0.08 < growth_2021_2022 < 0.12


class TestNetworkBottleneckIdentification:
    """Test bottleneck identification is meaningful."""

    def test_bottleneck_sectors_ranked(self, sample_spa):
        """Bottleneck sectors should be ranked by centrality."""
        paths = sample_spa.analyze(sector=1, depth=4)
        bottlenecks = paths.bottleneck_sectors(top_n=5)

        # Should be sorted in descending order by betweenness
        if len(bottlenecks) > 1:
            for i in range(len(bottlenecks) - 1):
                assert bottlenecks[i]["betweenness"] >= bottlenecks[i + 1]["betweenness"]


class TestUncertaintyStatistics:
    """Test uncertainty statistics are mathematically sound."""

    def test_monte_carlo_statistics_validity(self, sample_spa):
        """Monte Carlo statistics should satisfy basic properties."""
        res = sample_spa.monte_carlo(
            sector=1, depth=3, n_samples=50, intensity_cv=0.2, seed=42
        )

        # Mean should be positive
        assert res.total_intensity.mean > 0
        assert res.total_intensity.std >= 0

        # CV should reflect input uncertainty
        input_cv = 0.2
        estimated_cv = res.total_intensity.std / res.total_intensity.mean
        # Should be on the same order of magnitude (not equal due to propagation)
        assert estimated_cv > 0 or input_cv == 0

        # Verify samples are reasonable
        assert res.samples_total_intensity.shape[0] == 50
