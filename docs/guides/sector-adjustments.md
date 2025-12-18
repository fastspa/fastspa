# Sector Adjustments with IPF

This guide explains how to use Iterative Proportional Fitting (IPF) to adjust IO tables, technical coefficients, and satellite data for scenario analysis in EEIO models.

## Introduction

IPF is a powerful mathematical technique for adjusting economic and environmental data to match new constraints while preserving the underlying structure and relationships. In EEIO analysis, it's commonly used for:

- **Economic scenario analysis**: Updating IO tables to reflect policy changes or economic projections
- **Technology transitions**: Adjusting technical coefficients for new technologies
- **Environmental targets**: Scaling satellite data to meet emission reduction goals
- **Cross-classification adjustments**: Mapping data between different sector classifications

## When to Use IPF

### Use IPF when you need to:

1. **Update IO tables** to new economic output totals
2. **Adjust A matrices** while preserving input-output relationships
3. **Scale satellite data** to match new environmental constraints
4. **Create realistic scenarios** with preserved economic structure
5. **Balance multiple matrices** simultaneously with shared constraints

### Consider alternatives when:

- You have perfect data with no adjustment needs
- Constraints are so extreme they violate economic logic
- The system is highly non-linear (IPF works best with proportional relationships)

## Theoretical Context

### RAS vs. IPF in EEIO

While **mathematically identical** (RAS is simply the 2D application of IPF), they are often distinguished by their scope in EEIO:

1.  **RAS (Biproportional Scaling)**: The standard method for **balancing IO tables** (transaction matrices). It iteratively scales rows and columns to match new total output/input constraints while preserving the original structure (technological coefficients). It is simple, fast, and ideal for national table updates.

2.  **IPF (Iterative Proportional Fitting)**: Used for **multi-dimensional constraints** and more complex tasks. It handles situations beyond just row/column totals, such as regionalizing tables, disaggregating sectors, or adjusting multi-regional supply chain data where constraints exist across multiple dimensions (e.g., sectors × regions × emissions, trade margins, employment vectors).

3.  **Sector Adjustments**: For simple changes to specific coefficients (e.g., a technology shock), direct modification of the A-matrix is often sufficient. However, if this change must be consistent with new system-wide totals, RAS/IPF is required to re-balance the economy.

## Basic IPF Workflow

The typical IPF workflow involves:

1. **Prepare baseline data** (IO matrix, A matrix, or satellite data)
2. **Define target constraints** (new sector totals, emission targets, etc.)
3. **Apply IPF adjustment** using the appropriate method
4. **Validate results** (convergence, constraint satisfaction, reasonableness)
5. **Use adjusted data** in downstream analysis (SPA, hotspots, etc.)

## Step-by-Step Examples

### Example 1: Adjusting an IO Table

Suppose we have a 3-sector economy and want to model a green transition where services grow while manufacturing shrinks:

```python
import numpy as np
from fastspa import IPFBalancer

# Original IO transaction matrix
io_matrix = np.array([
    [100,  50,  30],   # Agriculture
    [ 40, 200,  60],   # Manufacturing  
    [ 20,  80, 150]    # Services
])

# Scenario: Services +30%, Manufacturing -10%, Agriculture +10%
baseline_totals = io_matrix.sum(axis=1)  # [180, 300, 250]
new_totals = baseline_totals * np.array([1.1, 0.9, 1.3])  # [198, 270, 325]

print(f"Baseline totals: {baseline_totals}")
print(f"Scenario totals: {new_totals}")

# Apply IPF adjustment
balancer = IPFBalancer()
result = balancer.adjust_io_matrix(
    io_matrix, 
    new_totals,  # Row constraints (outputs)
    new_totals   # Column constraints (inputs - assume symmetric)
)

print(f"\\nAdjustment results:")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Convergence rate: {result.convergence_rate:.2e}")

# Verify constraints
actual_outputs = result.adjusted_matrix.sum(axis=1)
print(f"\\nConstraint verification:")
print(f"Target outputs: {new_totals}")
print(f"Actual outputs: {actual_outputs}")
print(f"Difference: {abs(actual_outputs - new_totals).max():.2e}")
```

### Example 2: Adjusting Technical Coefficients

For SPA analysis, we often need to adjust the A matrix (technical coefficients) to reflect new economic structure:

```python
# Technical coefficients matrix
A_matrix = np.array([
    [0.10, 0.05, 0.02],  # Agriculture inputs
    [0.03, 0.15, 0.08],  # Manufacturing inputs
    [0.02, 0.07, 0.12]   # Services inputs
])

# Scenario: Change in gross outputs
new_gross_outputs = np.array([1.2, 0.8, 1.4])  # Agriculture +20%, Manufacturing -20%, Services +40%

# Adjust A matrix while preserving structure
A_result = balancer.adjust_a_matrix(
    A_matrix,
    new_gross_outputs,
    preserve_structure=True  # Maintain zero elements
)

print(f"\\nA matrix adjustment:")
print(f"Original A matrix:\\n{A_matrix}")
print(f"Adjusted A matrix:\\n{A_result.adjusted_matrix}")

# Check that zero structure is preserved
zero_mask = (A_matrix == 0)
print(f"Zero structure preserved: {np.all(A_result.adjusted_matrix[zero_mask] == 0)}")
```

### Example 3: Adjusting Satellite Data

Environmental satellite data (emissions, water use, etc.) can be adjusted to reflect technology changes or policy targets:

```python
# GHG emissions by sector (kg CO2-eq per unit output)
emissions = np.array([0.5, 2.1, 0.8])  # Agriculture, Manufacturing, Services

# Scenario: Technology improvements reduce manufacturing emissions by 30%
new_emissions = emissions.copy()
new_emissions[1] *= 0.7  # Manufacturing emissions reduced by 30%

print(f"\\nSatellite data adjustment:")
print(f"Original emissions: {emissions}")
print(f"Target emissions: {new_emissions}")

# Apply IPF to satellite data
emissions_result = balancer.adjust_satellite_data(emissions, [new_emissions])

print(f"Adjusted emissions: {emissions_result.adjusted_matrix}")
print(f"Total emissions preserved: {emissions_result.adjusted_matrix.sum():.3f} (original: {emissions.sum():.3f})")
```

### Example 4: Multi-dimensional Satellite Data

For more complex data (e.g., emissions by sector and region), IPF can handle multi-dimensional arrays:

```python
# 2D emissions data: sectors × regions
emissions_2d = np.array([
    [100,  50,  80],   # Region A: Agriculture, Manufacturing, Services
    [ 60, 120,  40],   # Region B
    [ 80,  30, 100]    # Region C
])

# Scenario: National emission targets by sector
sector_targets = np.array([260, 220, 200])  # Total emissions by sector across all regions

print(f"\\nMulti-dimensional adjustment:")
print(f"Original emissions by sector and region:\\n{emissions_2d}")
print(f"Sector targets: {sector_targets}")

# Adjust to meet sector targets (sum over regions)
emissions_2d_result = balancer.adjust_satellite_data(
    emissions_2d,
    [sector_targets],
    dimensions=[0]  # Sum over regions (dimension 0)
)

print(f"Adjusted emissions:\\n{emissions_2d_result.adjusted_matrix}")

# Verify sector totals
adjusted_sector_totals = emissions_2d_result.adjusted_matrix.sum(axis=0)
print(f"\\nSector totals verification:")
print(f"Target: {sector_targets}")
print(f"Actual: {adjusted_sector_totals}")
```

### Example 5: Scenario Analysis with SPA

The most powerful use case is combining IPF adjustments with SPA analysis to study environmental impacts under different scenarios:

```python
from fastspa import SPA

# Define sector names for interpretation
sector_names = ['Agriculture', 'Manufacturing', 'Services']

# Step 1: Create baseline SPA
spa_baseline = SPA(A_matrix, emissions, sector_names)
baseline_results = spa_baseline.analyze('Manufacturing', depth=5)

print(f"\\n=== Scenario Analysis ===")
print(f"\\nBaseline Analysis (Manufacturing sector):")
print(f"Total GHG intensity: {baseline_results.total_intensity:.4f}")
print(f"Number of significant pathways: {len(baseline_results.paths)}")

# Step 2: Create scenario data
scenario_gross_outputs = np.array([1.1, 0.9, 1.3])  # Green transition scenario
scenario_emissions_factor = np.array([0.9, 0.7, 1.1])  # Technology improvements

# Apply adjustments
A_scenario_result = balancer.adjust_a_matrix(A_matrix, scenario_gross_outputs)
emissions_scenario = emissions * scenario_emissions_factor
emissions_scenario_result = balancer.adjust_satellite_data(emissions, [emissions_scenario])

# Step 3: Run SPA on scenario data
spa_scenario = SPA(
    A_scenario_result.adjusted_matrix,
    emissions_scenario_result.adjusted_matrix,
    sector_names
)
scenario_results = spa_scenario.analyze('Manufacturing', depth=5)

print(f"\\nScenario Analysis (Manufacturing sector):")
print(f"Total GHG intensity: {scenario_results.total_intensity:.4f}")
print(f"Number of significant pathways: {len(scenario_results.paths)}")

# Step 4: Compare scenarios
change_pct = (scenario_results.total_intensity - baseline_results.total_intensity) / baseline_results.total_intensity * 100
print(f"\\nScenario Impact:")
print(f"Change in GHG intensity: {change_pct:+.1f}%")

# Show top pathways comparison
print(f"\\nTop 3 pathways - Baseline:")
for i, path in enumerate(baseline_results.top(3)):
    print(f"  {i+1}. {path}")

print(f"\\nTop 3 pathways - Scenario:")
for i, path in enumerate(scenario_results.top(3)):
    print(f"  {i+1}. {path}")
```

## Advanced Techniques

### Balancing Multiple Matrices

When you have multiple related matrices (e.g., different satellite accounts) that should be adjusted consistently:

```python
# Multiple satellite accounts
matrices = {
    'GHG': np.array([[100, 50], [30, 200]]),
    'Water': np.array([[80, 40], [25, 160]]),
    'Energy': np.array([[120, 60], [35, 240]])
}

# Constraints for each matrix
constraints = {
    'GHG': {
        'rows': np.array([180, 250]),  # New GHG totals by sector
        'cols': np.array([150, 280])
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

# Balance all matrices simultaneously
results = balancer.balance_multi_sector_data(matrices, constraints)

# Access adjusted matrices
ghg_adjusted = results['GHG'].adjusted_matrix
water_adjusted = results['Water'].adjusted_matrix
energy_adjusted = results['Energy'].adjusted_matrix

print(f"All matrices balanced consistently:")
print(f"GHG converged: {results['GHG'].converged}")
print(f"Water converged: {results['Water'].converged}")
print(f"Energy converged: {results['Energy'].converged}")
```

### Using Sector Concordance

When you need to adjust data between different sector classifications:

```python
from fastspa import adjust_io_table_with_concordance

# Original IO table in detailed classification (e.g., ANZSIC)
io_detailed = np.array([
    [100, 30, 20, 10],   # 4 detailed sectors
    [25, 200, 40, 15],
    [15, 35, 150, 25],
    [10, 20, 30, 80]
])

# Concordance to simplified classification (e.g., IOIG)
concordance = np.array([
    [1.0, 0.0],  # Sector 1 → Primary only
    [0.8, 0.2],  # Sector 2 → 80% Primary, 20% Secondary
    [0.3, 0.7],  # Sector 3 → 30% Primary, 70% Secondary
    [0.1, 0.9]   # Sector 4 → 10% Primary, 90% Secondary
])

# Target totals in simplified classification
target_simplified_totals = np.array([300, 450])

# Adjust IO table using concordance
adjusted_io, result = adjust_io_table_with_concordance(
    io_detailed, 
    concordance, 
    target_simplified_totals
)

print(f"IO table adjusted from detailed to simplified classification")
print(f"Converged: {result.converged}")
print(f"Adjusted matrix shape: {adjusted_io.shape}")
```

## Validation and Quality Checks

### Convergence Monitoring

Always check if IPF converged to your tolerance:

```python
result = balancer.adjust_io_matrix(io_matrix, new_outputs, new_inputs)

if not result.converged:
    print(f"⚠️  Warning: IPF did not converge")
    print(f"   Iterations: {result.iterations}")
    print(f"   Final convergence rate: {result.convergence_rate:.2e}")
    print(f"   Consider increasing max_iterations or relaxing convergence_rate")
else:
    print(f"✓ IPF converged successfully in {result.iterations} iterations")
```

### Constraint Verification

Manually verify that constraints are satisfied:

```python
def verify_constraints(result):
    \"\"\"Verify IPF constraints are satisfied.\"\"\"
    # Check row constraints
    actual_rows = result.adjusted_matrix.sum(axis=1)
    row_error = np.abs(actual_rows - result.row_constraints).max()
    
    # Check column constraints  
    actual_cols = result.adjusted_matrix.sum(axis=0)
    col_error = np.abs(actual_cols - result.col_constraints).max()
    
    print(f"Row constraint error: {row_error:.2e}")
    print(f"Column constraint error: {col_error:.2e}")
    
    tolerance = 1e-5
    if row_error < tolerance and col_error < tolerance:
        print("✓ Constraints satisfied within tolerance")
        return True
    else:
        print("✗ Constraints not satisfied")
        return False

verify_constraints(result)
```

### Reasonableness Checks

Check that adjusted values are reasonable:

```python
def check_reasonableness(original, adjusted):
    \"\"\"Check if adjusted values are reasonable.\"\"\"
    # Check for negative values
    if np.any(adjusted < 0):
        print("⚠️  Warning: Negative values in adjusted matrix")
        return False
    
    # Check for extreme changes
    change_ratios = adjusted / (original + 1e-12)  # Avoid division by zero
    extreme_changes = np.any((change_ratios > 10) | (change_ratios < 0.1))
    
    if extreme_changes:
        print("⚠️  Warning: Extreme changes detected (>10x or <0.1x)")
        print(f"Max ratio: {change_ratios.max():.2f}")
        print(f"Min ratio: {change_ratios.min():.2f}")
        return False
    
    # Check relative entropy (higher = more structural change)
    if result.relative_entropy > 1.0:
        print("⚠️  Warning: High relative entropy indicates major structural change")
        print(f"Relative entropy: {result.relative_entropy:.3f}")
    
    print("✓ Adjusted values appear reasonable")
    return True

check_reasonableness(result.original_matrix, result.adjusted_matrix)
```

## Common Pitfalls and Solutions

### 1. Non-Convergence

**Problem**: IPF doesn't converge within the maximum iterations.

**Solutions**:
- Increase `max_iterations`
- Relax `convergence_rate` (use larger tolerance)
- Check if constraints are internally consistent
- Verify input data quality

```python
# More permissive settings
result = balancer.adjust_io_matrix(
    io_matrix, new_outputs, new_inputs,
    max_iterations=5000,
    convergence_rate=1e-4  # More permissive
)
```

### 2. Inconsistent Constraints

**Problem**: Row and column constraints don't balance properly.

**Solution**: Ensure total row sum equals total column sum:

```python
total_row_constraint = new_outputs.sum()
total_col_constraint = new_inputs.sum()

if abs(total_row_constraint - total_col_constraint) > 1e-10:
    print(f"⚠️  Warning: Row total ({total_row_constraint}) != Column total ({total_col_constraint})")
    # Adjust one to match the other
    new_inputs = new_inputs * (total_row_constraint / total_col_constraint)
```

### 3. Zero Structure Issues

**Problem**: Preserving zero structure conflicts with constraint satisfaction.

**Solution**: Use `preserve_structure=False` for more flexible adjustment:

```python
result = balancer.adjust_a_matrix(
    A_matrix, new_outputs, 
    preserve_structure=False  # Allow filling of zero elements
)
```

### 4. Numerical Precision

**Problem**: Very small or very large numbers cause precision issues.

**Solution**: Scale your data appropriately:

```python
# Scale data to reasonable range
scaled_matrix = original_matrix / original_matrix.mean()
# ... apply IPF ...
# Scale back
result.adjusted_matrix = result.adjusted_matrix * original_matrix.mean()
```

## Best Practices Summary

1. **Always validate convergence** - Check `result.converged` and convergence metrics
2. **Verify constraints** - Manually check that constraints are satisfied
3. **Monitor reasonableness** - Watch for extreme changes or negative values
4. **Use appropriate tolerance** - Balance precision with computational efficiency
5. **Preserve structure when appropriate** - Use `preserve_structure=True` for A matrices
6. **Document scenarios clearly** - Keep track of what each scenario represents
7. **Test with simple cases** - Validate your approach with known examples first

## Integration with Other FastSPA Features

IPF integrates seamlessly with other FastSPA functionality:

- **SPA Analysis**: Use adjusted A matrices and satellite data for pathway analysis
- **Concordance**: Combine IPF with sector mapping for cross-classification adjustments  
- **Visualization**: Apply FastSPA plotting functions to scenario results
- **Multi-satellite**: Balance multiple environmental accounts consistently

For more advanced use cases, explore the [IPF API reference](../api/ipf.md) and example scripts in the `examples/` directory.