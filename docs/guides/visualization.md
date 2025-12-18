---
sidebar_position: 5
---

# Visualization Guide

FastSPA provides interactive visualization tools for exploring supply chain analysis results using Plotly. These visualizations help identify bottlenecks, hotspots, and supply chain structure at a glance.

## Overview

The visualization module provides three main chart types:

- **Icicle Plots**: Hierarchical rectangles showing supply chain depth and contribution magnitude
- **Sunburst Charts**: Radial hierarchical view of supply chains
- **Sector Contribution Charts**: Bar charts showing top contributing sectors

## Installation

Visualization features require Plotly:

```bash
pip install plotly
```

## Icicle Plot

The icicle plot is the primary visualization for supply chain decomposition. Each rectangle represents a supply chain link, with:

- **Width/Area**: Relative contribution to total intensity
- **Color Intensity**: Magnitude of the intensity value
- **Hierarchy**: Supply chain depth from target sector to suppliers

### Basic Usage

```python
from fastspa import SPA, icicle_plot

# Run analysis
spa = SPA(A_matrix, emissions, sectors=sector_names)
paths = spa.analyze(sector=0, depth=6, threshold=0.001)

# Generate interactive plot
fig = icicle_plot(paths, output_html="supply_chain.html")
```

### Parameters

```python
icicle_plot(
    paths_dict,                # PathCollection or dict of PathCollections
    output_html=None,          # Path to save HTML file (None = return Figure)
    title="...",              # Plot title
    colorscale="Plasma",      # Plotly colorscale name
    include_stage_0=True      # Include direct requirements
)
```

### Example: Multi-Satellite Analysis

```python
# Analyze multiple satellites
result = spa.analyze(sector=0, depth=6)  # Returns SPAResult with all satellites

# Visualize all satellites together
icicle_plot(
    result,
    output_html="multi_satellite.html",
    title="Multi-Satellite Supply Chain Analysis"
)
```

## Sunburst Plot

Sunburst charts provide a circular/radial view of the same hierarchy, useful for exploring specific branches by clicking.

```python
from fastspa import sunburst_plot

paths = spa.analyze(sector=0, depth=6)

sunburst_plot(
    paths,
    output_html="sunburst.html",
    max_depth=4  # Limit depth for clarity
)
```

### Parameters

```python
sunburst_plot(
    paths_dict,           # PathCollection or dict of PathCollections
    output_html=None,     # Path to save HTML
    title="...",         # Plot title
    colorscale="Viridis", # Color scheme
    max_depth=None       # Maximum depth to display
)
```

## Sector Contribution Charts

Bar charts showing which sectors contribute most to the target sector's total intensity.

```python
from fastspa import sector_contribution_chart

paths = spa.analyze(sector=0, depth=6)

sector_contribution_chart(
    paths,
    output_html="hotspots.html",
    top_n=15  # Show top 15 sectors
)
```

This aggregates contributions by the leaf/upstream sector where the intensity occurs.

## Working with Results

### Analyzing Single Sectors

```python
paths = spa.analyze(sector=42, depth=8)

# Create all three visualizations
icicle_plot(paths, output_html="icicle.html")
sunburst_plot(paths, output_html="sunburst.html")
sector_contribution_chart(paths, output_html="chart.html")
```

### Comparing Multiple Sectors

```python
# Analyze multiple target sectors
results = spa.analyze_many(
    sectors=[0, 5, 10, 15],
    depth=6
)

# Each result can be visualized
for sector_id, paths in results.items():
    icicle_plot(
        paths,
        output_html=f"icicle_sector_{sector_id}.html",
        title=f"Supply Chain: Sector {sector_id}"
    )
```

### System-Wide Analysis

```python
# Use system_wide mode for total requirements (including indirect flows)
spa = SPA(A_matrix, emissions, sectors=names, mode="system_wide")
paths = spa.analyze(sector=0, depth=6)

icicle_plot(
    paths,
    output_html="total_requirements.html",
    title="Total (Direct + Indirect) Supply Chain Requirements"
)
```

## Customization

### Color Scales

Plotly provides many built-in color scales. Common options:

- `"Plasma"` (default) - Perceptually uniform
- `"Viridis"` - Colorblind-friendly
- `"Turbo"` - High contrast
- `"Blues"` - Monochromatic
- `"RdBu"` - Diverging (red-blue)

```python
icicle_plot(paths, output_html="plot.html", colorscale="Viridis")
```

### Custom Titles

```python
icicle_plot(
    paths,
    output_html="plot.html",
    title="Greenhouse Gas Emissions in Supply Chain"
)
```

## Export Formats

All functions return Plotly Figure objects:

```python
# Save to HTML (interactive)
fig = icicle_plot(paths, output_html="plot.html")

# Return Figure (for further customization)
fig = icicle_plot(paths)

# Export to image (requires kaleido)
fig.write_image("plot.png")
fig.write_image("plot.pdf")
```

## Performance Notes

- Icicle plots handle 100-500+ paths efficiently
- Sunburst plots can be slower with deep hierarchies (limit with `max_depth`)
- HTML files are self-contained and can be shared/embedded

## Interpretation Guide

### Reading an Icicle Plot

1. **Top Level**: All supply chains combined
2. **Second Level**: Breakdown by satellite (if multiple)
3. **Deeper Levels**: Supply chain stages and sectors
4. **Rectangle Size**: Larger = higher contribution
5. **Color**: Darker/brighter = higher intensity value

### Identifying Hotspots

- Look for large rectangles - these are major contributors
- Darker colors indicate high intensity values
- Multiple large siblings often indicate diversified supply chains

### Exploring Branching Patterns

- Many branches = complex, multi-supplier chains
- Few branches = focused, concentrated supply chains
- Deep hierarchies = long supply chains with multiple tiers

## Troubleshooting

### ImportError: plotly not found

Install plotly:

```bash
pip install plotly
```

### Plot looks empty or has missing sectors

- Check that paths were found: `len(paths.paths) > 0`
- Reduce threshold: `threshold=0.001` instead of default
- Increase depth: `depth=8` or higher

### HTML file is very large

This is normal for many paths. You can:

- Reduce the number of paths: use `max_paths` parameter in `analyze()`
- Increase threshold: use higher threshold value
- Limit depth: use smaller `depth` value

## See Also

- [SPA Class Documentation](../api/spa-class.md)
- [Path Objects](../concepts/path-objects.md)
- [Quick Start Guide](../quick-start.md)
