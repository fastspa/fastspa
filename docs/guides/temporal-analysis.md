---
title: Temporal Analysis (Multi-year & Dynamic SPA)
---

Temporal and dynamic analysis is supported by running SPA over a sequence of model snapshots.
A snapshot typically corresponds to a year (or any time label).

## Multi-year path tracking

```python
import numpy as np
from fastspa import SPA

series = {
    2018: (A_2018, emissions_2018),
    2019: (A_2019, emissions_2019),
    2020: (A_2020, emissions_2020),
}

result = SPA.analyze_time_series(
    series,
    sector=42,
    depth=8,
    sectors=sectors,
)

# Total intensity over time
print(result.total_intensity_series())
```

## Export a summary table

If pandas is installed:

```python
result.to_dataframe()
```

## Notes

- Each timepoint runs a full SPA analysis (including traversal/pruning) with its own inputs.
- For comparing *specific pathways* across years, use consistent sector naming and thresholds.
