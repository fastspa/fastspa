"""Tests for explicit loop detection and analysis."""

from fastspa import Path, PathCollection


def test_loop_analysis_detects_cycle():
    paths = [
        Path(nodes=(0,), contribution=0.1, direct_intensity=1.0, cumulative_weight=1.0),
        Path(nodes=(0, 1, 0), contribution=0.2, direct_intensity=1.0, cumulative_weight=1.0),
        Path(nodes=(0, 2), contribution=0.3, direct_intensity=1.0, cumulative_weight=1.0),
    ]
    pc = PathCollection(paths, target_sector=0, total_intensity=10.0, satellite_name="ghg")

    loops = pc.loop_analysis()

    assert loops.loop_share == 0.2
    assert loops.cycles[(0, 1, 0)] == 0.2
    assert loops.sector_participation[0] == 0.2
    assert loops.sector_participation[1] == 0.2
    assert len(loops.paths) == 1


def test_loop_analysis_absolute_units():
    paths = [
        Path(nodes=(0, 1, 0), contribution=0.25, direct_intensity=1.0, cumulative_weight=1.0),
    ]
    pc = PathCollection(paths, target_sector=0, total_intensity=20.0, satellite_name="ghg")

    loops = pc.loop_analysis(as_absolute=True)

    assert loops.loop_share == 5.0
