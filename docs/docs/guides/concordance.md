---
sidebar_position: 2
---

# Sector Concordance

Map emissions data from one sector classification to another (e.g., national inventory sectors to IO sectors).

## Why Concordance?

In many EEIO applications:
- Direct intensities come from **national inventories** (e.g., NIBES by ANZSIC division)
- IO tables use a **different industry classification** (e.g., IO Industry Groups)

FastSPA provides explicit support for mapping between classifications.

## Creating a Concordance Matrix

### From Supply-Use Tables

```python
import pandas as pd
from fastspa import SectorConcordance, ConcordanceMetadata

# Load supply-use flow data
sut = pd.read_csv("sut_flows_2018.csv")
# Columns: anzsic, ioig, flow_value

concordance = SectorConcordance.from_supply_use_table(
    sut,
    source_col="anzsic",     # Source classification (e.g., ANZSIC codes)
    target_col="ioig",       # Target classification (IO sectors)
    value_col="flow_value",
    metadata=ConcordanceMetadata(
        source="ABS Supply-Use Tables 2018",
        method="supply_use_table",
        year=2018,
    ),
)
```

### Manual Concordance

```python
import numpy as np
from fastspa import SectorConcordance

# Rows: source sectors, Columns: target sectors
# Values: allocation weights (row-normalized)
matrix = np.array([
    [0.8, 0.2, 0.0],  # Source sector 0 → 80% target 0, 20% target 1
    [0.0, 0.5, 0.5],  # Source sector 1 → 50% target 1, 50% target 2
    [1.0, 0.0, 0.0],  # Source sector 2 → 100% target 0
])

concordance = SectorConcordance(
    matrix=matrix,
    source_sectors=["ANZSIC_01", "ANZSIC_02", "ANZSIC_03"],
    target_sectors=["IO_A", "IO_B", "IO_C"],
)
```

## Using Concordance

### Option A: Allocate Totals, Then Compute Intensities

```python
# 1. Source emissions (e.g., kt CO2-e by NIBES sector)
E_source = np.array([100, 200, 50])  # kt CO2-e

# 2. Allocate to IO sectors
E_io = concordance.allocate(E_source)  # kt CO2-e by IO sector

# 3. Divide by IO output to get intensities
io_output = np.array([1e6, 2e6, 0.5e6])  # AUD
intensities = (E_io * 1e6) / io_output  # kg CO2-e / AUD

# 4. Run SPA
spa = SPA(A_matrix, intensities, sectors=concordance.target_sectors)
```

### Option B: Use SatelliteWithConcordance

For intensity mapping with automatic allocation:

```python
from fastspa import SPA, SatelliteWithConcordance

# Source intensities (by NIBES/source sector)
source_intensities = np.array([1.5, 2.3, 0.8])  # kg CO2-e / AUD

satellite = SatelliteWithConcordance(
    intensities=source_intensities,
    concordance=concordance,
    name="carbon",
    unit="kg CO2-e / AUD",
)

# SPA handles the mapping automatically
spa = SPA(A_matrix, satellite, sectors=concordance.target_sectors)
paths = spa.analyze(sector="Electricity", depth=8)
```

## Multiple Satellites with Concordance

```python
satellites = {
    "carbon": SatelliteWithConcordance(
        carbon_intensities, concordance, name="carbon"
    ),
    "water": SatelliteWithConcordance(
        water_intensities, concordance, name="water"
    ),
}

spa = SPA(A_matrix, satellites, sectors=concordance.target_sectors)
result = spa.analyze(sector="Manufacturing", depth=8)
```

## Concordance Metadata

Track the source and method of your concordance:

```python
from fastspa import ConcordanceMetadata

metadata = ConcordanceMetadata(
    source="ABS Supply-Use Tables",
    method="supply_use_table",
    year=2018,
    notes="Based on 2018 supply table, column-normalized",
)

# Attach to concordance
concordance = SectorConcordance(
    matrix=matrix,
    source_sectors=source_sectors,
    target_sectors=target_sectors,
    metadata=metadata,
)

# Access later
print(concordance.metadata.source)
print(concordance.metadata.year)
```

## Best Practices

1. **Document your concordance**: Use `ConcordanceMetadata` to record sources and methods
2. **Validate sector alignment**: Ensure source/target sectors match your data
3. **Check row sums**: Concordance matrices should typically have rows summing to 1.0
4. **Preserve original data**: Keep source emissions separate from allocated values
