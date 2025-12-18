"""Tests for network science metrics on PathCollection."""


def test_edge_weights(sample_spa):
    paths = sample_spa.analyze(sector=1, depth=4)
    edges = paths.edge_weights()

    assert isinstance(edges, dict)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in edges.keys())
    assert all(isinstance(v, float) for v in edges.values())
    assert all(v >= 0 for v in edges.values())


def test_network_topology(sample_spa):
    paths = sample_spa.analyze(sector=1, depth=4)
    topo = paths.network_topology()

    assert topo["n_nodes"] == 3
    assert topo["n_edges"] >= 0
    assert 0.0 <= topo["density"] <= 1.0
    assert isinstance(topo["in_degree"], dict)
    assert isinstance(topo["out_strength"], dict)


def test_betweenness_centrality(sample_spa):
    paths = sample_spa.analyze(sector=1, depth=4)
    bc = paths.betweenness_centrality()

    assert set(bc.keys()) == {0, 1, 2}
    assert all(isinstance(v, float) for v in bc.values())
    assert all(v >= 0 for v in bc.values())


def test_bottleneck_sectors(sample_spa):
    paths = sample_spa.analyze(sector=1, depth=4)
    bottlenecks = paths.bottleneck_sectors(top_n=2)

    assert isinstance(bottlenecks, list)
    assert len(bottlenecks) <= 2
    for item in bottlenecks:
        assert "sector" in item
        assert "betweenness" in item
        assert "in_strength" in item
        assert "out_strength" in item
