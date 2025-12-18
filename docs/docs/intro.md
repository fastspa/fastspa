---
sidebar_position: 1
---

# Introduction

**FastSPA** is a modern Python interface for Structural Path Analysis (SPA) in Environmentally Extended Input-Output (EEIO) analysis.

## What is Structural Path Analysis?

Structural Path Analysis is a powerful technique for decomposing environmental impacts across supply chains. It traces the pathways through which environmental pressures flow from direct sources through multiple tiers of suppliers.

For example, when analyzing the carbon footprint of electricity generation, SPA can reveal:
- Direct emissions from power plants
- Emissions from coal mining (first-tier supplier)
- Emissions from transport of coal (second-tier)
- And so on through the entire supply chain

## Why FastSPA?

FastSPA provides a NumPy-native, functional interface for conducting SPA on input-output tables:

- **Array-first design**: Direct integration with NumPy/pandas workflows
- **Minimal ceremony**: No configuration files required to get started
- **Functional patterns**: Chainable operations for exploratory analysis
- **Two analysis modes**: Choose between sector-specific (A-matrix) or system-wide (Leontief) analysis
- **Multi-satellite support**: Analyze multiple environmental flows simultaneously
- **IPF sector adjustments**: Create realistic economic scenarios with Iterative Proportional Fitting

## Quick Example

```python
import numpy as np
from fastspa import SPA

# Your A-matrix and direct intensities as arrays
A = np.array([...])           # n×n technical coefficients
emissions = np.array([...])   # n-vector of direct intensities

# Run structural path analysis
paths = SPA(A, emissions).analyze(sector=42, depth=8)

# Explore the top 10 paths
for path in paths.top(10):
    print(f"{path.contribution:.2%}: {' → '.join(path.sectors)}")

# Export to DataFrame
df = paths.to_dataframe()
```

## Next Steps

- [Installation](./installation) - Get FastSPA installed
- [Quick Start](./quick-start) - Start analyzing in 5 minutes
- [Core Concepts](./concepts/analysis-modes) - Understand the fundamentals
- [Sector Adjustments](./guides/sector-adjustments) - Create scenarios with IPF
