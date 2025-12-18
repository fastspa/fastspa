---
sidebar_position: 3
---

# PathCollection

A `PathCollection` is a container for paths with built-in filtering, aggregation, export, and post-processing analysis.

## Basic Properties

```python
paths = spa.analyze(sector=42, depth=8)

len(paths)              # Number of paths found
paths.coverage          # Sum of all contributions (fraction of total)
paths.total_intensity   # Total intensity for the target sector
paths.target_sector     # Index of the analyzed sector
paths.satellite_name    # Name of the satellite analyzed
paths.metadata          # AnalysisMetadata object
```

## Filtering Paths

Filter paths by various criteria:

```python
# By minimum contribution
significant = paths.filter(min_contribution=0.01)  # Paths > 1%

# By maximum depth
shallow = paths.filter(max_depth=3)  # Only up to 3 stages

# By sector presence
coal_related = paths.filter(contains_sector=15)  # Contains sector 15

# Custom predicate
coal_ending = paths.filter(predicate=lambda p: "Coal" in p.sectors[-1])

# Combine multiple filters
filtered = paths.filter(min_contribution=0.005, max_depth=5)
```

## Getting Top Paths

```python
# Top N paths by contribution
top_ten = paths.top(10)

# Equivalent to slicing
top_ten = paths[:10]
```

## Aggregation

### By emission source sector

Sum contributions by the sector where emissions occur (leaf node):

```python
hotspots = paths.aggregate_by_sector()
# Returns: {sector_idx: total_contribution, ...}

# Sort by contribution
sorted_hotspots = sorted(hotspots.items(), key=lambda x: x[1], reverse=True)
for sector_idx, contribution in sorted_hotspots[:5]:
    print(f"Sector {sector_idx}: {contribution:.2%}")
```

### By supply chain stage

Group paths by the sector at a specific stage:

```python
# Group by first-tier suppliers
by_tier1 = paths.group_by_stage(1)
# Returns: {sector_idx: PathCollection, ...}

for sector_idx, group in by_tier1.items():
    print(f"Via sector {sector_idx}: {len(group)} paths, {group.coverage:.2%}")
```

### Semantic aggregation (custom & hierarchical)

Aggregate by leaf sector into user-defined semantic groups:

```python
mapping = {
    0: "Primary",
    1: "Secondary",
    2: "Services",
}

by_group = paths.semantic_aggregate(mapping)
```

Hierarchical aggregation uses multiple mappings to build multi-level keys:

```python
level_1 = {0: "Goods", 1: "Goods", 2: "Services"}
level_2 = {0: "Agriculture", 1: "Manufacturing", 2: "Services"}

hier = paths.hierarchical_aggregate([level_1, level_2])
# Keys look like: ("Goods", "Manufacturing")
```

## Network analysis (centrality & bottlenecks)

Treat the extracted paths as a directed weighted supply-chain network:

```python
edges = paths.edge_weights()              # (downstream, upstream) -> weight
bc = paths.betweenness_centrality()       # node betweenness
bottlenecks = paths.bottleneck_sectors()  # ranked bottleneck sectors

topo = paths.network_topology()           # density, in/out strengths, etc
```

## Loop detection (circular economy)

Detect repeated-sector cycles in extracted paths:

```python
loops = paths.loop_analysis()

loops.loop_share
loops.top_cycles(10)
```

## Export Methods

### To DataFrame

```python
df = paths.to_dataframe()
# Columns: contribution, contribution_pct, direct_intensity,
#          cumulative_weight, depth, stage_0, stage_0_name, stage_1, ...
```

### To CSV

```python
# With metadata header (default)
paths.to_csv("results.csv")

# Without metadata
paths.to_csv("results.csv", include_metadata=False)
```

### To JSON

```python
# To file
paths.to_json("results.json")

# To string
json_str = paths.to_json()
```

## Summary

Generate a text summary:

```python
print(paths.summary())
```

## Metadata

Access analysis metadata for reproducibility:

```python
meta = paths.metadata

meta.analysis_date      # When analysis was run
meta.sector             # Target sector
meta.threshold          # Threshold used
meta.threshold_type     # 'percentage' or 'absolute'
meta.max_depth          # Maximum depth explored
meta.total_intensity    # Total intensity
meta.coverage           # Fraction covered
meta.n_paths            # Number of paths
meta.mode               # 'sector_specific' or 'system_wide'
meta.satellite          # Satellite name
meta.n_sectors          # Total sectors in system

# Export metadata
meta_dict = meta.to_dict()
```
