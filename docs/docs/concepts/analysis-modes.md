---
sidebar_position: 1
---

# Analysis Modes

FastSPA supports two complementary analysis modes for different research questions.

## Sector-Specific Analysis (A-Matrix Mode)

Traces **direct supplier relationships** through the technical coefficients matrix.

```python
# Default mode: sector-specific
spa = SPA(A_matrix, direct_intensities, mode="sector_specific")
paths = spa.analyze(sector=42, depth=8)
```

**Characteristics:**
- Uses the A-matrix (technical coefficients) directly
- Traces direct supplier relationships only
- More efficient computation
- Coverage typically ~90-95%
- Clear pathway attribution

**Use when:**
- Analyzing direct supplier relationships
- Need efficient computation for many sectors
- Want clear pathway attribution
- Focus on sector-specific supply chains

## System-Wide Analysis (Leontief Mode)

Traces **total requirements** (direct + indirect) through the Leontief inverse matrix.

```python
# System-wide mode: uses Leontief inverse
spa = SPA(A_matrix, direct_intensities, mode="system_wide")
paths = spa.analyze(sector=42, depth=8)
```

**Characteristics:**
- Uses L = (I - A)^(-1) (Leontief inverse)
- Captures circular flows and feedback loops
- More comprehensive pathway discovery
- Coverage can exceed 100% (expected due to circular flows)
- Economy-wide perspective

**Use when:**
- Analyzing total supply chain requirements
- Need comprehensive pathway discovery
- Want to capture circular flows
- Studying economy-wide scenarios

## Comparison

| Aspect | Sector-Specific (A-Matrix) | System-Wide (Leontief) |
|--------|---------------------------|------------------------|
| Input Matrix | A (technical coefficients) | L (Leontief inverse) |
| Pathways | Fewer (~1,300 typical) | More (~12,000 typical) |
| Coverage | ~93% | Can exceed 100% |
| Circular Flows | Not captured | Captured |
| Computation | Faster | Slower |
| Use Case | Sector analysis | Economy-wide scenarios |

## Example: Comparing Modes

```python
from fastspa import SPA

# Same data, different modes
spa_sector = SPA(A, emissions, mode="sector_specific")
spa_system = SPA(A, emissions, mode="system_wide")

# Analyze the same sector
paths_sector = spa_sector.analyze(sector=42, depth=8)
paths_system = spa_system.analyze(sector=42, depth=8)

print(f"Sector-specific: {len(paths_sector)} paths, {paths_sector.coverage:.1%} coverage")
print(f"System-wide: {len(paths_system)} paths, {paths_system.coverage:.1%} coverage")
```

## Pre-computed Leontief Inverse

If you already have a Leontief inverse computed, use the `from_leontief` factory function:

```python
from fastspa import from_leontief

# Skip L computation - defaults to system_wide mode
spa = from_leontief(L_matrix, intensities, sectors)
```
