---
title: Network Analysis (Centrality & Bottlenecks)
---

fastspa can treat extracted structural paths as a *directed, weighted supply-chain network*.
This enables common network science workflows like bottleneck identification.

## Build a path-induced network

```python
from fastspa import SPA

paths = SPA(A, emissions, sectors=sectors).analyze(sector=42, depth=8)

edges = paths.edge_weights()  # (downstream, upstream) -> weight
```

By default, each edge weight is the sum of the contributing path fractions that traverse that edge.

## Betweenness centrality

```python
bc = paths.betweenness_centrality()

# Inspect the most central transmission sectors
bottlenecks = paths.bottleneck_sectors(top_n=10)
```

## Topology summary

```python
topo = paths.network_topology()
print(topo["n_nodes"], topo["n_edges"], topo["density"])
```

### Interpretation notes

- The graph is **directed** from the target sector (root) upstream to suppliers.
- Centrality is computed over the **path-induced** subnetwork (not the full A-matrix).
- For policy use, bottlenecks are typically the highest betweenness sectors (excluding the target).
