# fastspa

A modern Python interface for Structural Path Analysis in EEIO.

## Overview

Structural Path Analysis (SPA) is a powerful technique for decomposing environmental impacts across supply chains. This package provides a NumPy-native, functional interface for conducting SPA on input-output tables.

### Background

This package builds on the foundations established by tools like [pyspa](https://github.com/hybridlca/pyspa), which use object-oriented SPA in Python. While pyspa offers a comprehensive, object-oriented approach with CSV-based workflows well-suited to many research contexts, `fastspa` explores an alternative design emphasising:

- **Array-first design**: Direct integration with NumPy/pandas workflows
- **Minimal ceremony**: No configuration files required to get started  
- **Functional patterns**: Chainable operations for exploratory analysis

Both approaches have their place—choose the tool that fits your workflow.

## Installation

```bash
pip install fastspa
```

## Quick Start

```python
import numpy as np
from fastspa import SPA

# Your A-matrix and direct intensities as arrays
A = np.array([...])           # n×n technical coefficients
emissions = np.array([...])   # n-vector of direct intensities

# Sector-specific analysis (A-matrix, direct suppliers)
paths = SPA(A, emissions, mode="sector_specific").analyze(sector=42, depth=8)

# System-wide analysis (Leontief inverse, total requirements)
paths = SPA(A, emissions, mode="system_wide").analyze(sector=42, depth=8)

# Explore results
for path in paths.top(10):
    print(f"{path.contribution:.2%}: {' → '.join(path.sectors)}")

# Export to DataFrame
df = paths.to_dataframe()
```

## Analysis Modes

SPA supports two complementary analysis modes for different research questions:

### Sector-Specific Analysis (A-Matrix Mode)

Traces **direct supplier relationships** through the technical coefficients matrix.

```python
# Default mode: sector-specific
spa = SPA(A_matrix, direct_intensities, mode="sector_specific")
paths = spa.analyze(sector=42, depth=8)

# Results:
# - 1,312 pathways (example)
# - 93.14% coverage
# - Efficient for sector-specific analysis
# - Clear direct supply chain decomposition
```

**Use when:**
- Analyzing direct supplier relationships
- Need efficient computation for many sectors
- Want clear pathway attribution
- Coverage should be ~90-95%

### System-Wide Analysis (Leontief Mode)

Traces **total requirements** (direct + indirect) through the Leontief inverse matrix.
Captures circular flows and economy-wide supply chain decomposition.

```python
# System-wide mode: uses Leontief inverse
spa = SPA(A_matrix, direct_intensities, mode="system_wide")
paths = spa.analyze(sector=42, depth=8)

# Results:
# - 11,971 pathways (example)
# - 328.36% coverage (>100% due to circular flows)
# - Comprehensive system-wide analysis
# - Captures all circular flows
```

**Use when:**
- Analyzing total supply chain requirements
- Need comprehensive pathway discovery
# - Want to capture circular flows
- Studying economy-wide scenarios
- Coverage can exceed 100% (expected)

### Comparison

| Aspect | Sector-Specific (A-Matrix) | System-Wide (Leontief) |
|--------|---------------------------|------------------------|
| Input Matrix | A (technical coefficients) | L (Leontief inverse) |
| Pathways | ~1,300 | ~12,000 |
| Coverage | ~93% | ~328% |
| Circular Flows | Not captured | Captured |
| Computation | Faster | Slower |
| Use Case | Sector analysis | Economy-wide scenarios |

## Core Concepts

### Creating an SPA Instance

```python
from fastspa import SPA

# Basic: just arrays (defaults to sector-specific mode)
spa = SPA(A_matrix, direct_intensities)

# Explicit sector-specific mode (A-matrix)
spa = SPA(A_matrix, direct_intensities, mode="sector_specific")

# System-wide mode (Leontief inverse)
spa = SPA(A_matrix, direct_intensities, mode="system_wide")

# With sector names for readable output
spa = SPA(A_matrix, direct_intensities, sectors=["Agriculture", "Mining", ...])

# Multiple environmental satellites
spa = SPA(A_matrix, {
    "ghg": ghg_intensities,
    "water": water_intensities,
    "energy": energy_intensities
}, mode="system_wide")
```

### Running Analysis

```python
# By index
paths = spa.analyze(sector=42, depth=8)

# By name (if sectors provided)
paths = spa.analyze(sector="Electricity", depth=8)

# Adjust threshold (fraction of total intensity)
paths = spa.analyze(sector=42, depth=8, threshold=0.001)

# Limit results
paths = spa.analyze(sector=42, depth=8, max_paths=100)
```

### Working with Results

The `PathCollection` object supports filtering, aggregation, and export:

```python
# Filter paths
significant = paths.filter(min_contribution=0.01)
shallow = paths.filter(max_depth=3)
mining_related = paths.filter(contains_sector=5)

# Combine filters
filtered = paths.filter(min_contribution=0.005, max_depth=5)

# Custom predicates
paths.filter(predicate=lambda p: "Coal" in p.sectors[-1])

# Aggregate by emission source
hotspots = paths.aggregate_by_sector()

# Group by supply chain stage
by_tier1 = paths.group_by_stage(1)

# Export
paths.to_dataframe()          # pandas DataFrame
paths.to_csv("results.csv")   # CSV file
paths.to_json("results.json") # JSON file
```

### Path Objects

Each `Path` represents a single supply chain route:

```python
path = paths[0]

path.nodes          # (42, 15, 3) - sector indices
path.sectors        # ("Electricity", "Coal", "Mining") - sector names
path.contribution   # 0.0523 - fraction of total intensity
path.direct_intensity  # intensity at the leaf node
path.depth          # number of upstream stages
path.root           # target sector
path.leaf           # final upstream sector
```

## Multi-Satellite Analysis
 
Analyse multiple environmental flows simultaneously:
 
```python
spa = SPA(A, {
    "ghg": ghg_vector,
    "water": water_vector,
    "land": land_vector,
})
 
# Analyze all satellites
result = spa.analyze(sector="Manufacturing", depth=8)
 
# Access by satellite name
result["ghg"].top(10)
result["water"].to_dataframe()
 
# Or analyze one specifically
water_paths = spa.analyze(sector="Manufacturing", satellite="water")
```

## Evidence-based Sector Concordance

In many EEIO applications, direct intensities come from **national inventories by economic sector**
(e.g. a National Inventory by Economic Sector (NIBES) table with CO₂-e by ANZSIC division/group),
while the IO table uses a different industry classification (e.g. IO Industry Groups).

`fastspa` supports this workflow explicitly via sector **concordance matrices** and satellites with
embedded concordance logic:

- Use a national inventory (e.g. NIBES) as the **source satellite** (emissions by ANZSIC sector)
- Derive a **concordance matrix** from supply–use / IO tables that maps source sectors → IO sectors
- Let the library allocate emissions or intensities into the IO classification before running SPA

A typical carbon workflow looks like this:

```python
import pandas as pd
import numpy as np
from fastspa import SPA, SectorConcordance, ConcordanceMetadata, SatelliteWithConcordance

# 1. National inventory: emissions by NIBES / ANZSIC sector (e.g. NIBES 2018)
nibes = pd.read_csv("nibes_emissions_2018.csv")  # columns: year, sector_code, emissions_kt
nibes_2018 = nibes[nibes["year"] == 2018]

# 2. Supply–use based concordance: ANZSIC → IO industries
sut = pd.read_csv("sut_flows_2018.csv")  # columns: anzsic, ioig, flow_value
concordance = SectorConcordance.from_supply_use_table(
    sut,
    source_col="anzsic",   # NIBES / ANZSIC codes, e.g. 26, 27, 46, ...
    target_col="ioig",     # IO industries matching A-matrix
    value_col="flow_value",
    metadata=ConcordanceMetadata(
        source="ABS Supply-Use Tables 2018",
        method="supply_use_table",
        year=2018,
    ),
)

# 3. Build source emissions vector in concordance order (totals)
E_source_kt = np.array([
    nibes_2018.loc[nibes_2018["sector_code"] == code, "emissions_kt"].item()
    for code in concordance.source_sectors
])

# Option A: map totals then divide by IO output
E_io_kt = concordance.allocate(E_source_kt)      # emissions by IO sector (kt CO2-e)

io_output = pd.read_csv("io_output_2018.csv")   # columns: ioig, output_aud
x_io = np.array([
    io_output.loc[io_output["ioig"] == s, "output_aud"].item()
    for s in concordance.target_sectors
])

# Intensities in IO classification (kg CO2-e / AUD)
g_io = (E_io_kt * 1_000_000) / x_io

spa = SPA(A_matrix, g_io, sectors=concordance.target_sectors)
paths = spa.analyze(sector="Electricity", depth=8)

# Option B: use SatelliteWithConcordance for intensities
# (e.g. if you have NIBES sector outputs and want to map intensities directly)

nibes_output = pd.read_csv("nibes_output_2018.csv")  # sector_code, output_aud
x_nibes = np.array([
    nibes_output.loc[nibes_output["sector_code"] == code, "output_aud"].item()
    for code in concordance.source_sectors
])

source_intensities = (E_source_kt * 1_000_000) / x_nibes  # kg CO2-e / AUD per NIBES sector

satellite = SatelliteWithConcordance(
    intensities=source_intensities,
    concordance=concordance,
    name="carbon",
    unit="kg CO2-e / AUD",
)

spa = SPA(A_matrix, satellite, sectors=concordance.target_sectors)
paths = spa.analyze(sector="Electricity", depth=8)
```

Under the hood this uses [`SectorConcordance`](fastspa/concordance.py:90) and
[`SatelliteWithConcordance`](fastspa/concordance.py:260) to keep the mapping between
your **national inventory sectors** (e.g. NIBES / ANZSIC) and the **IO sectors**
explicit, auditable, and consistent with the OECD/UN extended SUT/EEIO guidance.

## Loading Data

### From Arrays (Primary Interface)
 
```python
from fastspa import SPA

# Directly with IO-sector intensities
spa = SPA(A_matrix, intensities, sectors)
```

If your direct intensities come from a **national inventory by economic sector** in a
classification that differs from the IO table (e.g. NIBES / ANZSIC vs IO Industry Groups),
combine them with a concordance matrix via [`SectorConcordance`](fastspa/concordance.py:90)
and [`SatelliteWithConcordance`](fastspa/concordance.py:260) (see
*Evidence-based Sector Concordance* above).

### From DataFrames

```python
from fastspa import from_dataframe

spa = from_dataframe(
    A_df,                                    # A-matrix as DataFrame
    intensities_df,                          # Intensities DataFrame  
    intensity_columns=["CO2", "Water"],      # Which columns to use
)
```

### From Pre-computed Leontief Inverse

```python
from fastspa import from_leontief

# Skip Leontief computation if you already have L
# Defaults to system_wide mode for Leontief-based analysis
spa = from_leontief(L_matrix, intensities, sectors)

# Or explicitly specify mode
spa = from_leontief(L_matrix, intensities, sectors, mode="system_wide")
```

### From CSV Files

```python
from fastspa import from_csv

spa = from_csv(
    "A_matrix.csv",
    "intensities.csv",
    intensity_columns=["ghg", "water"],
)
```

### From EEIO Databases

```python
from fastspa import from_io_table

# EXIOBASE (experimental)
spa = from_io_table("path/to/exiobase/", satellites=["GHG", "Water"])

# From pymrio objects
import pymrio
mrio = pymrio.parse_exiobase3(path="...")
mrio.calc_all()
spa = from_io_table(mrio)
```

## Memory-Efficient Streaming

For large systems, stream paths without collecting all in memory:

```python
for path in spa.stream(sector=42, depth=12, threshold=1e-9):
    if path.contribution > 0.001:
        process(path)
```

## Comparison Tools

```python
# Compare multiple sectors
df = spa.compare_sectors(
    ["Agriculture", "Mining", "Manufacturing"],
    depth=8
)

# Identify hotspots
hotspots = spa.hotspots(sector="Manufacturing", top_n=10)
for idx, contribution, name in hotspots:
    print(f"{name}: {contribution:.2%}")
```

## Interactive Visualization

Visualize SPA results with interactive, hierarchical plots. Requires `plotly`:

```bash
pip install plotly
```

### Icicle Plot (Recommended)

Interactive hierarchical visualization showing supply chain structure, depth, and contributions:

```python
from fastspa import icicle_plot, sunburst_plot, sector_contribution_chart

paths = spa.analyze(sector="Manufacturing", depth=8)

# Generate interactive icicle plot
fig = icicle_plot(
    paths,
    output_html="supply_chain.html",
    title="Supply Chain: Manufacturing"
)

# Customize appearance
fig = icicle_plot(
    paths,
    colorscale="Viridis",        # Choose color scheme
    include_stage_0=True,        # Include direct requirements
)
```

**Features:**
- Rectangle area = relative contribution magnitude
- Color gradient = intensity value
- Click to zoom into supply chain stages
- Hover for detailed metrics
- Export to HTML/PNG/PDF

### Sunburst Plot

Radial hierarchical view for exploring branching supply chains:

```python
fig = sunburst_plot(
    paths,
    output_html="sunburst.html",
    max_depth=4  # Limit depth for clarity
)
```

### Sector Contribution Chart

Bar chart showing top contributing sectors:

```python
fig = sector_contribution_chart(
    paths,
    output_html="hotspots.html",
    top_n=15
)
```

### Multi-Satellite Visualization

Visualize multiple satellites together:

```python
# Analyze all satellites
result = spa.analyze(sector="Manufacturing", depth=8)

# Create combined icicle plot
icicle_plot(
    result,
    output_html="multi_satellite.html",
    title="Multi-Satellite Supply Chain"
)
```

See [Visualization Guide](docs/docs/guides/visualization.md) for more details.

## Network analysis (centrality & bottlenecks)

You can treat extracted paths as a directed, weighted supply-chain network and compute bottleneck metrics:

```python
paths = spa.analyze(sector=42, depth=8)

# Directed edge weights induced by the extracted paths
edges = paths.edge_weights()

# Betweenness centrality (bottleneck identification)
bc = paths.betweenness_centrality()

# Ranked bottleneck sectors (excludes the target sector by default)
bottlenecks = paths.bottleneck_sectors(top_n=10)
```

See the docs: [Network Analysis](docs/docs/guides/network-analysis.md)

## Uncertainty quantification (Monte Carlo, LHS, sensitivity)

Propagate uncertainty in direct intensities through to totals and sector contributions:

```python
mc = spa.monte_carlo(
    sector=42,
    depth=8,
    n_samples=500,
    intensity_cv=0.2,        # 20% coefficient of variation
    distribution="lognormal",
    sampling="lhs",         # or "random"
    seed=42,
)

mc.total_intensity.mean, mc.total_intensity.ci_low, mc.total_intensity.ci_high
```

You can also compute elasticities of total intensity to direct intensities:

```python
sens = spa.sensitivity(sector=42)
```

See the docs: [Uncertainty](docs/docs/guides/uncertainty.md)

## Semantic aggregation (custom & hierarchical)

Aggregate results into stakeholder-friendly sector groups after calculation:

```python
paths = spa.analyze(sector=42, depth=8)

mapping = {
    0: "Primary",
    1: "Secondary",
    2: "Services",
}

by_group = paths.semantic_aggregate(mapping)
```

For hierarchical (multi-level) groupings:

```python
level_1 = {0: "Goods", 1: "Goods", 2: "Services"}
level_2 = {0: "Agriculture", 1: "Manufacturing", 2: "Services"}

hier = paths.hierarchical_aggregate([level_1, level_2])
```

See the docs: [Semantic Aggregation](docs/docs/guides/semantic-aggregation.md)

## Consequential mode (policy / perturbation scenarios)

Quantify scenario deltas by perturbing intensities or the A-matrix and re-running SPA:

```python
import numpy as np

mult = np.ones(spa.n_sectors)
mult[10] = 1.2  # 20% increase for one sector

res = spa.consequential(
    sector=42,
    depth=8,
    intensity_multiplier=mult,
)

res.delta_total_intensity
res.delta_sector_contributions  # absolute deltas by leaf sector
```

See the docs: [Consequential Mode](docs/docs/guides/consequential-mode.md)

## Temporal analysis (multi-year)

Run SPA over a time series of model snapshots:

```python
series = {
    2020: (A_2020, emissions_2020),
    2021: (A_2021, emissions_2021),
}

ts = SPA.analyze_time_series(series, sector=42, depth=8, sectors=sectors)

ts.total_intensity_series()
```

See the docs: [Temporal Analysis](docs/docs/guides/temporal-analysis.md)

## Loop detection (circular economy)

Explicitly detect repeated-sector cycles in extracted paths:

```python
paths = spa.analyze(sector=42, depth=8)
loops = paths.loop_analysis()

loops.loop_share
loops.top_cycles(10)
```

See the docs: [Loop Analysis](docs/docs/guides/loop-analysis.md)

## User stories

- As a policy analyst, I need to identify bottleneck sectors so that I can prioritise interventions where change will propagate through supply chains.
- As an academic researcher, I need Monte Carlo confidence intervals so that I can publish results with defensible uncertainty bounds.
- As an LCA practitioner, I need to aggregate IO sectors into stakeholder-friendly groupings so that I can communicate results clearly.

See more: [User Stories](docs/docs/user-stories.md)

## API Summary

### SPA Class

| Method | Description |
|--------|-------------|
| `analyze(sector, depth, ...)` | Run SPA for a sector |
| `analyze_many(sectors, ...)` | Analyze multiple sectors |
| `stream(sector, depth, ...)` | Memory-efficient path iteration |
| `hotspots(sector, ...)` | Identify top emission sources |
| `compare_sectors(sectors, ...)` | Compare sectors in a DataFrame |
| `total_intensity(satellite)` | Get total intensity vector |
| `sensitivity(sector, ...)` | Elasticities of total intensity to direct intensities |
| `monte_carlo(sector, ...)` | Monte Carlo / LHS uncertainty propagation |
| `consequential(sector, ...)` | Scenario deltas under input perturbations |
| `analyze_time_series(series, ...)` | Multi-year / dynamic SPA workflows |

### PathCollection Class

| Method | Description |
|--------|-------------|
| `top(n)` | Get top N paths |
| `filter(...)` | Filter by contribution, depth, sector |
| `group_by_stage(n)` | Group paths by sector at stage n |
| `aggregate_by_sector()` | Sum contributions by source sector |
| `semantic_aggregate(mapping, ...)` | Aggregate contributions into custom semantic groups |
| `hierarchical_aggregate(levels, ...)` | Multi-level aggregation (e.g. division → group) |
| `edge_weights()` | Build a weighted directed edge list from extracted paths |
| `network_topology()` | Basic topology metrics for the path-induced network |
| `betweenness_centrality()` | Betweenness centrality for bottleneck detection |
| `bottleneck_sectors()` | Ranked bottleneck sectors (betweenness + strength) |
| `loop_analysis()` | Detect and summarise circular loops in extracted paths |
| `to_dataframe()` | Export to pandas DataFrame |
| `to_csv(path)` | Export to CSV |
| `to_json(path)` | Export to JSON |
| `summary()` | Text summary of results |

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20

Optional:
- pandas (for DataFrame export)
- scipy (for sparse matrices, MATLAB loading)
- matplotlib (for visualization)
- plotly (for interactive Sankey diagrams)

## License

MIT

## Acknowledgments

This package was inspired by the structural path analysis methodology and builds on ideas from the broader EEIO research community, including tools like [pyspa](https://github.com/hybridlca/pyspa) and [pymrio](https://github.com/IndEcol/pymrio).

## How to Cite

If you use `fastspa` in your research or work, please cite:

```bibtex
@misc{fastspa2024,
  title = {{fastspa}: A modern Python interface for Structural Path Analysis in EEIO},
  author = {Weis, Edan and Bunster, Victor},
  year = {2024},
  url = {https://docs.fastspa.net},
  note = {Software available at \url{https://docs.fastspa.net}}
}
```

For inquiries, contact Victor Bunster (victor.bunster@monash.edu).