---
sidebar_position: 3
---

# Quick Start

Get up and running with FastSPA in 5 minutes.

## Basic Usage

```python
import numpy as np
from fastspa import SPA

# Create a simple 3-sector economy
A = np.array([
    [0.1, 0.2, 0.1],  # Agriculture
    [0.2, 0.1, 0.3],  # Manufacturing
    [0.1, 0.2, 0.2],  # Services
])

# Direct emissions per unit output (e.g., kg CO2/$ output)
emissions = np.array([0.5, 1.2, 0.3])

# Create SPA instance with sector names
spa = SPA(
    A,
    emissions,
    sectors=["Agriculture", "Manufacturing", "Services"]
)

# Analyze Manufacturing sector (sector 2, using 1-indexed)
paths = spa.analyze(sector=2, depth=5)

# View results
print(paths.summary())
```

## Understanding the Output

Each path represents a supply chain route:

```python
for path in paths.top(5):
    print(path)
```

Output:
```
 45.23% | Manufacturing
 12.34% | Manufacturing → Agriculture
  8.76% | Manufacturing → Services
  5.43% | Manufacturing → Agriculture → Manufacturing
  3.21% | Manufacturing → Services → Manufacturing
```

## Key Properties

```python
# Get the top path
top_path = paths[0]

# Sector indices in the path
print(top_path.nodes)  # (1, 0, 1) - 0-indexed internally

# Sector names
print(top_path.sectors)  # ('Manufacturing', 'Agriculture', 'Manufacturing')

# Contribution to total intensity
print(f"{top_path.contribution:.2%}")  # 45.23%

# Direct intensity at the emission source (leaf node)
print(top_path.direct_intensity)  # 1.2

# Path depth (number of upstream stages)
print(top_path.depth)  # 0 for direct, 1+ for upstream
```

## Visualize Results

Create an interactive icicle plot to explore supply chain structure (requires Plotly):

```bash
pip install plotly
```

```python
from fastspa import icicle_plot

# Generate interactive icicle plot
icicle_plot(
    paths,
    output_html="supply_chain.html",
    title="Manufacturing Supply Chain"
)
```

This creates an interactive HTML file you can open in your browser. The visualization shows:
- **Hierarchy**: Supply chain depth from target sector to final suppliers
- **Size**: Rectangle area represents contribution magnitude
- **Color**: Gradient shows intensity values
- **Interactivity**: Click to zoom into supply chain stages

For more visualization options, see the [Visualization Guide](./guides/visualization).

## Export Results

```python
# To pandas DataFrame
df = paths.to_dataframe()

# To CSV
paths.to_csv("results.csv")

# To JSON
paths.to_json("results.json")
```

## Next Steps

- Learn about [Analysis Modes](./concepts/analysis-modes) (sector-specific vs system-wide)
- Explore [PathCollection](./concepts/pathcollection) filtering and aggregation
- Set up [Multi-satellite analysis](./guides/multi-satellite)
- Create [Interactive visualizations](./guides/visualization)
