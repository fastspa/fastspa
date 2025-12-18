# IPF (Iterative Proportional Fitting) API

Fastspa provides comprehensive IPF functionality for adjusting IO tables, technical coefficients, and satellite data to match updated sector constraints while preserving underlying structure and cross-product ratios.

## Overview

IPF is widely used in EEIO analysis for:

- **IO table balancing**: Update transaction matrices to new sector outputs and uses
- **Technical coefficients adjustment**: Modify A matrices while preserving relationships
- **Satellite data scaling**: Adjust environmental data to match new constraints
- **Scenario analysis**: Create realistic economic scenarios for policy analysis

Based on the ipfn package and academic literature on iterative proportional fitting.

## Classes

### IPFBalancer

Main class for performing IPF adjustments across various data types.

```python
from fastspa import IPFBalancer

balancer = IPFBalancer()
```

#### Methods

##### adjust_io_matrix()

Adjust an IO transaction matrix to match new row and column totals.

```python
result = balancer.adjust_io_matrix(
    io_matrix,
    new_row_totals,
    new_col_totals,
    max_iterations=1000,
    convergence_rate=1e-6,
    rate_tolerance=0.0,
    verbose=0
)
```

**Parameters:**
- `io_matrix`: Original transaction matrix (n×n)
- `new_row_totals`: Target row sums (outputs by sector)
- `new_col_totals`: Target column sums (inputs by sector)
- `max_iterations`: Maximum number of iterations (default: 1000)
- `convergence_rate`: Convergence tolerance (default: 1e-6)
- `rate_tolerance`: Tolerance for early stopping based on rate change (default: 0.0)
- `verbose`: Verbosity level (0=silent, 1=status, 2=detailed, default: 0)

**Returns:** `IPFResult`

**Example:**
```python
import numpy as np
from fastspa import IPFBalancer

# Original IO matrix
io_matrix = np.array([
    [100, 50],
    [30, 200]
])

# New sector totals
new_outputs = np.array([180, 250])  # New gross outputs
new_inputs = np.array([150, 280])   # New sector inputs

# Apply IPF adjustment
balancer = IPFBalancer()
result = balancer.adjust_io_matrix(io_matrix, new_outputs, new_inputs)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Adjusted matrix:\n{result.adjusted_matrix}")
```

##### adjust_a_matrix()

Adjust technical coefficients matrix to new sector totals.

```python
result = balancer.adjust_a_matrix(
    A_matrix,
    new_row_totals,
    new_col_totals=None,
    preserve_structure=True,
    max_iterations=1000,
    convergence_rate=1e-6,
    verbose=0
)
```

**Parameters:**
- `A_matrix`: Technical coefficients matrix (n×n)
- `new_row_totals`: Target sector outputs
- `new_col_totals`: Target sector inputs (if None, uses row totals)
- `preserve_structure`: Whether to preserve zero structure (default: True)
- `max_iterations`: Maximum iterations (default: 1000)
- `convergence_rate`: Convergence tolerance (default: 1e-6)
- `verbose`: Verbosity level (default: 0)

**Returns:** `IPFResult`

**Example:**
```python
# Technical coefficients matrix
A_matrix = np.array([
    [0.1, 0.05],
    [0.03, 0.15]
])

# New gross outputs
new_outputs = np.array([1.2, 1.8])

# Adjust A matrix
result = balancer.adjust_a_matrix(A_matrix, new_outputs, preserve_structure=True)
```

##### adjust_satellite_data()

Adjust satellite data (emissions, resource use) to new sector totals.

```python
result = balancer.adjust_satellite_data(
    satellite_data,
    new_totals,
    dimensions=None,
    max_iterations=1000,
    convergence_rate=1e-6
)
```

**Parameters:**
- `satellite_data`: Satellite data array (1D, 2D, or 3D)
- `new_totals`: Target totals along specified dimensions
- `dimensions`: Which dimensions to constrain (default: [0])
- `max_iterations`: Maximum iterations (default: 1000)
- `convergence_rate`: Convergence tolerance (default: 1e-6)

**Returns:** `IPFResult`

**Examples:**

1D satellite data:
```python
# Emissions by sector
emissions = np.array([100, 50, 80, 30])
new_targets = np.array([120, 60, 95, 35])

result = balancer.adjust_satellite_data(emissions, [new_targets])
```

2D satellite data (by sector and region):
```python
# Emissions by sector and region
emissions_2d = np.array([
    [100, 50, 80],   # Region A
    [60, 120, 40],   # Region B
    [80, 30, 100]    # Region C
])

# New totals by sector (sum over regions)
sector_totals = np.array([260, 220, 200])

result = balancer.adjust_satellite_data(
    emissions_2d, 
    [sector_totals],
    dimensions=[1]  # Sum over regions
)
```

##### balance_multi_sector_data()

Balance multiple related data matrices simultaneously.

```python
results = balancer.balance_multi_sector_data(
    data_matrices,
    constraints,
    max_iterations=1000,
    convergence_rate=1e-6
)
```

**Parameters:**
- `data_matrices`: Dict mapping names to matrices to adjust
- `constraints`: Dict mapping matrix names to constraint dicts
- `max_iterations`: Maximum iterations (default: 1000)
- `convergence_rate`: Convergence tolerance (default: 1e-6)

**Returns:** Dict[str, IPFResult]

**Example:**
```python
# Multiple satellite matrices
matrices = {
    'GHG': np.array([[100, 50], [30, 200]]),
    'Water': np.array([[80, 40], [25, 160]]),
    'Energy': np.array([[120, 60], [35, 240]])
}

# Constraints for each matrix
constraints = {
    'GHG': {
        'rows': np.array([180, 250]),  # New GHG totals by sector
        'cols': np.array([150, 280])   # New sector inputs
    },
    'Water': {
        'rows': np.array([140, 200]),  # New water use by sector
        'cols': np.array([120, 220])
    },
    'Energy': {
        'rows': np.array([200, 320]),  # New energy use by sector
        'cols': np.array([180, 340])
    }
}

# Balance all matrices
results = balancer.balance_multi_sector_data(matrices, constraints)

# Access results
ghg_result = results['GHG']
water_result = results['Water']
```

### IPFResult

Dataclass containing results from IPF adjustment operations.

```python
@dataclass
class IPFResult:
    adjusted_matrix: NDArray
    original_matrix: NDArray
    iterations: int
    convergence_rate: float
    converged: bool
    row_constraints: NDArray
    col_constraints: NDArray
    relative_entropy: float
```

**Attributes:**
- `adjusted_matrix`: Final adjusted matrix after IPF convergence
- `original_matrix`: Original input matrix
- `iterations`: Number of iterations taken to converge
- `convergence_rate`: Final convergence rate
- `converged`: Whether algorithm converged within max_iterations
- `row_constraints`: Target row sums that were applied
- `col_constraints`: Target column sums that were applied
- `relative_entropy`: Relative entropy distance from original to adjusted

**Example:**
```python
result = balancer.adjust_io_matrix(io_matrix, new_outputs, new_inputs)

print(f"Convergence: {'✓' if result.converged else '✗'}")
print(f"Iterations: {result.iterations}")
print(f"Final rate: {result.convergence_rate:.2e}")
print(f"Relative entropy: {result.relative_entropy:.4f}")

# Verify constraints
adjusted_outputs = result.adjusted_matrix.sum(axis=1)
np.testing.assert_allclose(adjusted_outputs, result.row_constraints, rtol=1e-5)
```

## Functions

### adjust_io_table_with_concordance()

Adjust IO table using sector concordance to target totals.

```python
adjusted_io, result = adjust_io_table_with_concordance(
    io_matrix,
    concordance_matrix,
    target_sector_totals,
    max_iterations=1000,
    convergence_rate=1e-6
)
```

**Parameters:**
- `io_matrix`: Original IO matrix in source classification
- `concordance_matrix`: Mapping from source to target sectors
- `target_sector_totals`: Target totals in target classification
- `max_iterations`: Maximum IPF iterations (default: 1000)
- `convergence_rate`: Convergence tolerance (default: 1e-6)

**Returns:** Tuple[NDArray, IPFResult]

**Example:**
```python
from fastspa import adjust_io_table_with_concordance

# Original IO table in ANZSIC classification
io_anzsic = np.array([
    [100, 30, 20],
    [25, 200, 40],
    [15, 35, 150]
])

# Concordance: ANZSIC → IOIG
concordance = np.array([
    [0.8, 0.2],  # ANZSIC 1 → 80% IOIG 1, 20% IOIG 2
    [0.3, 0.7],  # ANZSIC 2 → 30% IOIG 1, 70% IOIG 2
    [0.1, 0.9]   # ANZSIC 3 → 10% IOIG 1, 90% IOIG 2
])

# Target totals in IOIG classification
target_ioig_totals = np.array([300, 450])

# Apply adjustment
adjusted_io, result = adjust_io_table_with_concordance(
    io_anzsic, concordance, target_ioig_totals
)
```

## Integration with SPA

IPF functionality integrates seamlessly with SPA analysis for scenario-based environmental assessment:

```python
from fastspa import SPA, IPFBalancer

# Step 1: Create baseline SPA
spa_baseline = SPA(A_baseline, emissions, sector_names)
baseline_paths = spa_baseline.analyze('Manufacturing', depth=5)

# Step 2: Create scenario with IPF
balancer = IPFBalancer()
A_result = balancer.adjust_a_matrix(A_baseline, new_outputs)
emissions_result = balancer.adjust_satellite_data(emissions, new_emissions)

# Step 3: Run SPA on adjusted data
spa_scenario = SPA(A_result.adjusted_matrix, emissions_result.adjusted_matrix, sector_names)
scenario_paths = spa_scenario.analyze('Manufacturing', depth=5)

# Step 4: Compare scenarios
print(f"Baseline intensity: {baseline_paths.total_intensity:.4f}")
print(f"Scenario intensity: {scenario_paths.total_intensity:.4f}")
change = (scenario_paths.total_intensity - baseline_paths.total_intensity) / baseline_paths.total_intensity * 100
print(f"Change: {change:+.1f}%")
```

## Best Practices

### 1. Convergence Monitoring
Always check convergence status and convergence rate:
```python
result = balancer.adjust_io_matrix(io_matrix, new_outputs, new_inputs)

if not result.converged:
    print(f"Warning: IPF did not converge after {result.iterations} iterations")
    print(f"Final convergence rate: {result.convergence_rate:.2e}")
```

### 2. Constraint Validation
Verify that constraints are satisfied:
```python
# Check row constraints
actual_rows = result.adjusted_matrix.sum(axis=1)
np.testing.assert_allclose(actual_rows, result.row_constraints, rtol=1e-5)

# Check column constraints
actual_cols = result.adjusted_matrix.sum(axis=0)
np.testing.assert_allclose(actual_cols, result.col_constraints, rtol=1e-5)
```

### 3. Structure Preservation
For A matrices, use `preserve_structure=True` to maintain zero elements:
```python
result = balancer.adjust_a_matrix(A_matrix, new_outputs, preserve_structure=True)

# Verify zero structure is preserved
zero_mask = (A_matrix == 0)
assert np.all(result.adjusted_matrix[zero_mask] == 0)
```

### 4. Multi-dimensional Data
For complex satellite data, carefully specify dimensions:
```python
# For 3D data (sector × region × time)
satellite_3d = np.array([...])  # shape: (n_sectors, n_regions, n_years)

# Constrain sector totals across regions and time
sector_totals = np.array([...])  # shape: (n_sectors,)

result = balancer.adjust_satellite_data(
    satellite_3d, 
    [sector_totals],
    dimensions=[0]  # Sum over regions and time
)
```

### 5. Scenario Analysis
Use IPF for realistic scenario creation:
```python
# Green transition scenario
new_outputs = np.array([1.1, 0.9, 1.3])  # Sector growth rates
new_emissions = np.array([0.9, 0.7, 1.1])  # Emission intensity changes

# Apply adjustments
A_result = balancer.adjust_a_matrix(A_matrix, new_outputs)
emissions_result = balancer.adjust_satellite_data(emissions, new_emissions)

# Run comparative SPA analysis
# ... (see integration example above)
```

## Error Handling

The IPF implementation includes comprehensive error handling:

### Input Validation
- Matrix dimension checking
- Constraint array size validation
- Non-negative constraint verification
- Concordance matrix normalization checks

### Convergence Issues
- Automatic convergence monitoring
- Detailed convergence metrics
- Graceful handling of non-convergence

### Numerical Stability
- Handling of zero and near-zero values
- Numerical precision management
- Relative entropy calculations

## References

The IPF implementation is based on:

1. **Deming, W.E. & Stephan, F.F. (1940)**: Original IPF paper for contingency tables
2. **Rüschendorf, L. (1995)**: Convergence properties of IPF under marginal constraints
3. **Norman, P. (1999)**: Practical IPF implementation guidelines
4. **ipfn package**: Python implementation of IPF algorithms

For detailed mathematical foundations, see the academic references in the main fastspa documentation.