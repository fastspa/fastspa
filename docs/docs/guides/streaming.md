---
sidebar_position: 3
---

# Memory-Efficient Streaming

For large systems, stream paths without collecting all in memory.

## Basic Streaming

Use `stream()` instead of `analyze()`:

```python
for path in spa.stream(sector=42, depth=12, threshold=1e-9):
    if path.contribution > 0.001:
        process(path)
```

## Key Differences from `analyze()`

| Aspect | `analyze()` | `stream()` |
|--------|------------|-----------|
| Returns | `PathCollection` | Iterator of `Path` |
| Memory | Loads all paths | One path at a time |
| Order | Sorted by contribution | Discovery order |
| Features | Filtering, aggregation | Manual processing |

## Use Cases

### Large Systems

When the number of paths exceeds available memory:

```python
# System with 10,000+ sectors
spa = SPA(large_A_matrix, intensities)

# Stream instead of loading all
significant_paths = []
for path in spa.stream(sector=1, depth=15, threshold=1e-12):
    if path.contribution > 0.0001:
        significant_paths.append(path)

print(f"Found {len(significant_paths)} significant paths")
```

### Early Termination

Stop processing when you have enough:

```python
top_paths = []
for path in spa.stream(sector=42, depth=10):
    if path.contribution > 0.01:
        top_paths.append(path)
    if len(top_paths) >= 100:
        break  # Stop early

print(f"Collected {len(top_paths)} paths with >1% contribution")
```

### Real-Time Processing

Process paths as they're discovered:

```python
import csv

with open("paths.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["contribution", "path", "depth"])

    for path in spa.stream(sector=42, depth=8):
        writer.writerow([
            f"{path.contribution:.6f}",
            " → ".join(path.sectors),
            path.depth
        ])
```

### Custom Aggregation

Build custom aggregations without storing all paths:

```python
from collections import defaultdict

# Count paths by depth
depth_counts = defaultdict(int)
depth_contributions = defaultdict(float)

for path in spa.stream(sector=42, depth=10):
    depth_counts[path.depth] += 1
    depth_contributions[path.depth] += path.contribution

for depth in sorted(depth_counts.keys()):
    print(f"Depth {depth}: {depth_counts[depth]} paths, "
          f"{depth_contributions[depth]:.2%} contribution")
```

## Parameters

`stream()` accepts most of the same parameters as `analyze()`:

```python
spa.stream(
    sector=42,           # Target sector (required)
    depth=8,             # Maximum depth (default: 8)
    threshold=0.001,     # Minimum threshold (default: 0.001)
    threshold_type="percentage",  # or "absolute"
    satellite="ghg",     # Which satellite to analyze
)
```

## Combining with PathCollection

If you need PathCollection features after streaming:

```python
from fastspa import PathCollection

# Stream and collect significant paths
paths_list = [
    path for path in spa.stream(sector=42, depth=10)
    if path.contribution > 0.001
]

# Create PathCollection manually
paths = PathCollection(
    paths=paths_list,
    target_sector=42,
    total_intensity=spa.total_intensity()[42],
    satellite_name="intensity",
)

# Now use PathCollection features
df = paths.to_dataframe()
```
