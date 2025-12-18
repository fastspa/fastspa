---
sidebar_position: 2
---

# Path Objects

A `Path` represents a single supply chain pathway through which environmental impacts flow.

## Path Properties

```python
path = paths[0]  # Get the top path

# Core attributes
path.nodes          # Tuple of sector indices, e.g., (42, 15, 3, 7)
path.sectors        # Tuple of sector names, e.g., ('Electricity', 'Coal', 'Transport', 'Steel')
path.contribution   # Fraction of total intensity, e.g., 0.0523

# Intensity data
path.direct_intensity   # Direct intensity at the leaf node
path.cumulative_weight  # Product of A-matrix coefficients along the path

# Convenience properties
path.depth   # Number of upstream stages (len(nodes) - 1)
path.root    # Target sector (first node)
path.leaf    # Final upstream sector (last node)
```

## Path Structure

A path represents the flow from a target sector back through its supply chain:

```
Target → Tier 1 → Tier 2 → ... → Emission Source
  │         │         │                │
root     stage 1   stage 2           leaf
```

For example, a path `(42, 15, 3)` means:
- Sector 42 purchases from Sector 15
- Sector 15 purchases from Sector 3
- Emissions occur at Sector 3

## Understanding Contribution

The `contribution` represents what fraction of the target sector's total intensity flows through this specific pathway:

```python
# All path contributions should sum close to coverage
total_contribution = sum(p.contribution for p in paths)
print(f"Coverage: {total_contribution:.2%}")  # e.g., "Coverage: 93.14%"
```

## String Representation

Paths have a readable string format:

```python
print(path)
# Output: "  5.23% | Electricity → Coal Mining → Transport → Steel"
```

## Accessing Multiple Paths

Iterate over paths or use slicing:

```python
# Top 10 paths
for path in paths.top(10):
    print(path)

# Slice directly
first_five = paths[:5]

# Get specific path
third_path = paths[2]
```
