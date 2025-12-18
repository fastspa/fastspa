"""Tests for semantic and hierarchical aggregation."""

import pytest


def test_semantic_aggregate_by_index(sample_spa):
    paths = sample_spa.analyze(sector=1, depth=4)
    mapping = {0: "Primary", 1: "Secondary", 2: "Tertiary"}

    agg = paths.semantic_aggregate(mapping)

    assert set(agg.keys()).issubset({"Primary", "Secondary", "Tertiary"})
    assert pytest.approx(sum(agg.values()), rel=1e-12) == paths.coverage


def test_semantic_aggregate_by_name(sample_spa):
    paths = sample_spa.analyze(sector=1, depth=4)
    mapping = {
        "Agriculture": "Primary",
        "Manufacturing": "Secondary",
        "Services": "Tertiary",
    }

    agg = paths.semantic_aggregate(mapping)

    assert set(agg.keys()).issubset({"Primary", "Secondary", "Tertiary"})
    assert pytest.approx(sum(agg.values()), rel=1e-12) == paths.coverage


def test_hierarchical_aggregate(sample_spa):
    paths = sample_spa.analyze(sector=1, depth=4)

    level_1 = {0: "Goods", 1: "Goods", 2: "Services"}
    level_2 = {0: "Agriculture", 1: "Manufacturing", 2: "Services"}

    agg = paths.hierarchical_aggregate([level_1, level_2])

    assert all(isinstance(k, tuple) and len(k) == 2 for k in agg.keys())
    assert pytest.approx(sum(agg.values()), rel=1e-12) == paths.coverage
