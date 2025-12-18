---
title: Uncertainty Quantification (Monte Carlo & LHS)
---

fastspa supports uncertainty propagation by repeatedly sampling uncertain inputs and re-running SPA.

## Monte Carlo uncertainty

```python
from fastspa import SPA

spa = SPA(A, emissions, sectors=sectors)

mc = spa.monte_carlo(
    sector=42,
    depth=8,
    n_samples=500,
    intensity_cv=0.2,      # 20% coefficient of variation on direct intensities
    distribution="lognormal",
    seed=42,
)

print(mc.total_intensity.mean, mc.total_intensity.ci_low, mc.total_intensity.ci_high)
```

### Sector contribution uncertainty

Monte Carlo results report uncertainty on leaf-sector contributions.

```python
sector_stats = mc.sector_contributions
print(sector_stats[10].mean, sector_stats[10].ci_low, sector_stats[10].ci_high)
```

If pandas is installed:

```python
mc.sector_contributions_dataframe()
```

## Latin Hypercube Sampling (LHS)

For the same distributional assumptions, LHS can reduce variance in estimates with fewer samples.

```python
mc_lhs = spa.monte_carlo(
    sector=42,
    depth=8,
    n_samples=200,
    intensity_cv=0.2,
    sampling="lhs",
    seed=42,
)
```

## Sensitivity (elasticities)

Compute elasticities of total intensity to each direct intensity:

```python
sens = spa.sensitivity(sector=42)
```

These elasticities sum to 1.0 for a given target sector.
