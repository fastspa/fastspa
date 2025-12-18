"""
IPF (Iterative Proportional Fitting) Examples

This module demonstrates how to use the IPF functionality for sector adjustments
in EEIO analysis.
"""

import numpy as np

try:
    from fastspa import IPFBalancer, adjust_io_table_with_concordance, SectorConcordance
    HAS_IPF = True
except ImportError:
    HAS_IPF = False
    print("IPF functionality not available. Install with: pip install ipfn")


def example_basic_io_adjustment():
    """Example: Basic IO table adjustment using IPF."""
    if not HAS_IPF:
        return
    
    print("=== Basic IO Table Adjustment ===")
    
    # Original IO transaction matrix (2x2 example)
    io_matrix = np.array([
        [100, 50],   # Sector 1: 100 internal, 50 to sector 2
        [30, 200]    # Sector 2: 30 from sector 1, 200 internal
    ])
    
    print("Original IO matrix:")
    print(io_matrix)
    
    # New sector totals we want to achieve
    new_outputs = np.array([180, 250])  # New gross outputs by sector
    new_inputs = np.array([150, 280])   # New inputs by sector
    
    print(f"\nTarget outputs: {new_outputs}")
    print(f"Target inputs: {new_inputs}")
    
    # Apply IPF adjustment
    balancer = IPFBalancer()
    result = balancer.adjust_io_matrix(
        io_matrix,
        new_outputs,
        new_inputs,
        convergence_rate=1e-6
    )
    
    print(f"\nAdjusted IO matrix:")
    print(result.adjusted_matrix)
    
    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final convergence rate: {result.convergence_rate:.2e}")
    print(f"Relative entropy: {result.relative_entropy:.4f}")
    
    # Verify constraints are satisfied
    adjusted_outputs = result.adjusted_matrix.sum(axis=1)
    adjusted_inputs = result.adjusted_matrix.sum(axis=0)
    
    print(f"\nVerification:")
    print(f"Actual outputs: {adjusted_outputs} (target: {new_outputs})")
    print(f"Actual inputs: {adjusted_inputs} (target: {new_inputs})")


def example_a_matrix_adjustment():
    """Example: Adjust technical coefficients matrix."""
    if not HAS_IPF:
        return
    
    print("\n=== A Matrix Adjustment ===")
    
    # Technical coefficients matrix
    A_matrix = np.array([
        [0.1, 0.05],  # Sector 1: 10% from self, 5% from sector 2
        [0.03, 0.15]  # Sector 2: 3% from sector 1, 15% from self
    ])
    
    print("Original A matrix:")
    print(A_matrix)
    
    # New gross outputs by sector
    new_outputs = np.array([1.2, 1.8])
    
    print(f"\nNew gross outputs: {new_outputs}")
    
    # Adjust A matrix
    balancer = IPFBalancer()
    result = balancer.adjust_a_matrix(
        A_matrix,
        new_outputs,
        preserve_structure=True
    )
    
    print(f"\nAdjusted A matrix:")
    print(result.adjusted_matrix)
    
    print(f"\nNote: A matrix preserves zero structure and cross-product ratios")


def example_satellite_data_adjustment():
    """Example: Adjust satellite data (emissions, water use, etc.)."""
    if not HAS_IPF:
        return
    
    print("\n=== Satellite Data Adjustment ===")
    
    # Emissions data by sector (1D example)
    emissions = np.array([100, 50, 80, 30])  # CO2 emissions by sector
    
    print("Original emissions by sector:")
    print(emissions)
    
    # New emission targets
    new_targets = np.array([120, 60, 95, 35])
    
    print(f"\nNew emission targets:")
    print(new_targets)
    
    # Adjust satellite data
    balancer = IPFBalancer()
    result = balancer.adjust_satellite_data(
        emissions,
        [new_targets]
    )
    
    print(f"\nAdjusted emissions:")
    print(result.adjusted_matrix)
    
    # Verify total is preserved
    original_total = emissions.sum()
    adjusted_total = result.adjusted_matrix.sum()
    
    print(f"\nTotal emissions:")
    print(f"Original: {original_total}")
    print(f"Adjusted: {adjusted_total}")
    print(f"Difference: {abs(adjusted_total - original_total):.2e}")


def example_multi_dimensional_satellite():
    """Example: Adjust 2D satellite data (e.g., by sector and region)."""
    if not HAS_IPF:
        return
    
    print("\n=== Multi-dimensional Satellite Data ===")
    
    # Emissions by sector and region (2D example)
    emissions_2d = np.array([
        [100, 50, 80],   # Region A: sectors 1,2,3
        [60, 120, 40],   # Region B: sectors 1,2,3
        [80, 30, 100]    # Region C: sectors 1,2,3
    ])
    
    print("Original emissions (regions × sectors):")
    print(emissions_2d)
    
    # New totals by sector (sum over regions)
    sector_totals = np.array([260, 220, 200])
    
    print(f"\nNew sector totals:")
    print(sector_totals)
    
    # Adjust 2D data
    balancer = IPFBalancer()
    result = balancer.adjust_satellite_data(
        emissions_2d,
        [sector_totals],
        dimensions=[0]  # Sum over regions (dimension 0)
    )
    
    print(f"\nAdjusted emissions:")
    print(result.adjusted_matrix)
    
    # Verify sector totals
    adjusted_sector_totals = result.adjusted_matrix.sum(axis=0)
    print(f"\nAdjusted sector totals: {adjusted_sector_totals}")
    print(f"Target sector totals: {sector_totals}")


def example_concordance_adjustment():
    """Example: Adjust IO table using sector concordance."""
    if not HAS_IPF:
        return
    
    print("\n=== IO Table Adjustment with Concordance ===")
    
    # Original IO table in source classification (e.g., ANZSIC)
    io_source = np.array([
        [100, 50, 30],
        [40, 200, 60],
        [20, 80, 150]
    ])
    
    print("Original IO table (source classification):")
    print(io_source)
    
    # Concordance matrix (source → target sectors)
    # Example: 3 ANZSIC sectors → 2 IOIG sectors
    concordance_matrix = np.array([
        [0.8, 0.2],  # ANZSIC 1 → 80% IOIG 1, 20% IOIG 2
        [0.3, 0.7],  # ANZSIC 2 → 30% IOIG 1, 70% IOIG 2
        [0.1, 0.9]   # ANZSIC 3 → 10% IOIG 1, 90% IOIG 2
    ])
    
    print(f"\nConcordance matrix (source → target):")
    print(concordance_matrix)
    
    # Target totals in target classification
    target_totals = np.array([300, 450])  # New IOIG sector totals
    
    print(f"\nTarget totals (target classification):")
    print(target_totals)
    
    # Apply adjustment with concordance
    adjusted_io, result = adjust_io_table_with_concordance(
        io_source,
        concordance_matrix,
        target_totals
    )
    
    print(f"\nAdjusted IO table (back in source classification):")
    print(adjusted_io)
    
    print(f"\nAdjustment result:")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Convergence rate: {result.convergence_rate:.2e}")


def example_multiple_matrices():
    """Example: Balance multiple matrices simultaneously."""
    if not HAS_IPF:
        return
    
    print("\n=== Balancing Multiple Matrices ===")
    
    # Multiple satellite matrices (GHG, Water, Energy)
    matrices = {
        'GHG': np.array([[100, 50], [30, 200]]),
        'Water': np.array([[80, 40], [25, 160]]),
        'Energy': np.array([[120, 60], [35, 240]])
    }
    
    print("Original matrices:")
    for name, matrix in matrices.items():
        print(f"{name}:")
        print(matrix)
        print()
    
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
    
    print("Constraints:")
    for name, constraint in constraints.items():
        print(f"{name} - rows: {constraint['rows']}, cols: {constraint['cols']}")
    print()
    
    # Balance all matrices simultaneously
    balancer = IPFBalancer()
    results = balancer.balance_multi_sector_data(matrices, constraints)
    
    print("Adjusted matrices:")
    for name, result in results.items():
        print(f"{name}:")
        print(result.adjusted_matrix)
        print(f"Converged: {result.converged}, Iterations: {result.iterations}")
        print()


def run_all_examples():
    """Run all IPF examples."""
    if not HAS_IPF:
        print("IPF functionality not available. Install with: pip install ipfn")
        return
    
    print("IPF (Iterative Proportional Fitting) Examples")
    print("=" * 50)
    
    example_basic_io_adjustment()
    example_a_matrix_adjustment()
    example_satellite_data_adjustment()
    example_multi_dimensional_satellite()
    example_concordance_adjustment()
    example_multiple_matrices()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()