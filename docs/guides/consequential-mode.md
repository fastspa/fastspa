---
title: Consequential Mode (Perturbation / Policy Scenarios)
---

Consequential mode supports "what-if" analysis by applying a perturbation to inputs and recomputing SPA.

## Direct intensity perturbation

```python
import numpy as np
from fastspa import SPA

spa = SPA(A, emissions, sectors=sectors)

# e.g. 20% increase in direct intensity for sector 10
mult = np.ones(spa.n_sectors)
mult[10] = 1.2

res = spa.consequential(
    sector=42,
    depth=8,
    intensity_multiplier=mult,
)

print(res.delta_total_intensity)
```

`res.delta_sector_contributions` reports changes in *absolute* leaf-sector contributions.

## A-matrix perturbation

```python
# Additive delta to A (same shape as A)
A_delta = np.zeros_like(A)
A_delta[10, 5] = 1e-4

res = spa.consequential(
    sector=42,
    depth=8,
    A_delta=A_delta,
)
```

## Notes

- Perturbations are applied as: `(base + delta) * multiplier`.
- Use consequential mode when you need scenario deltas for policy levers.
