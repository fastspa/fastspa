"""
IPF and SPA Integration Example

This example demonstrates how to use IPF to adjust sector data before
running Structural Path Analysis (SPA), enabling scenario-based analysis
with updated economic structures.
"""

import numpy as np

try:
    from fastspa import SPA, IPFBalancer, SectorConcordance, ConcordanceMetadata
    HAS_IPF = True
except ImportError:
    HAS_IPF = False
    print("Required functionality not available. Install with: pip install ipfn fastspa")


def scenario_analysis_example():
    """
    Example: Using IPF to create economic scenarios and analyze their environmental impacts.
    
    This demonstrates a common workflow:
    1. Start with baseline IO table and environmental data
    2. Use IPF to adjust to new economic scenario
    3. Run SPA on adjusted data to analyze environmental pathways
    4. Compare results across scenarios
    """
    if not HAS_IPF:
        return
    
    print("=== IPF + SPA Integration: Scenario Analysis ===")
    
    # Step 1: Baseline data (simplified 3-sector economy)
    print("Step 1: Loading baseline data...")
    
    # Technical coefficients matrix (A matrix)
    A_baseline = np.array([
        [0.1, 0.05, 0.02],   # Agriculture: 10% self, 5% from Manufacturing, 2% from Services
        [0.03, 0.15, 0.08],  # Manufacturing: 3% from Agriculture, 15% self, 8% from Services
        [0.02, 0.07, 0.12]   # Services: 2% from Agriculture, 7% from Manufacturing, 12% self
    ])
    
    # GHG emissions by sector (kg CO2-eq per unit output)
    ghg_baseline = np.array([0.5, 2.1, 0.8])
    
    # Water use by sector (L per unit output)
    water_baseline = np.array([100, 50, 20])
    
    sector_names = ['Agriculture', 'Manufacturing', 'Services']
    
    print(f"Baseline A matrix:\n{A_baseline}")
    print(f"Baseline GHG intensities: {ghg_baseline}")
    print(f"Baseline water intensities: {water_baseline}")
    
    # Step 2: Create economic scenario
    print("\nStep 2: Creating economic scenario...")
    
    # Scenario: Green transition - reduce manufacturing emissions, increase services
    new_outputs = np.array([1.1, 0.9, 1.3])  # Agriculture +10%, Manufacturing -10%, Services +30%
    
    # Adjust technical coefficients
    balancer = IPFBalancer()
    
    # First adjust A matrix to new output structure
    A_result = balancer.adjust_a_matrix(
        A_baseline,
        new_outputs,
        preserve_structure=True
    )
    A_scenario = A_result.adjusted_matrix
    
    print(f"Scenario outputs: {new_outputs}")
    print(f"Adjusted A matrix:\n{A_scenario}")
    print(f"A matrix adjustment converged: {A_result.converged}")
    
    # Adjust environmental intensities to reflect technology changes
    # Scenario: New technologies reduce manufacturing emissions by 30%
    new_ghg = np.array([0.5, 2.1 * 0.7, 0.8])  # Manufacturing emissions reduced
    
    ghg_result = balancer.adjust_satellite_data(
        ghg_baseline,
        [new_ghg]
    )
    ghg_scenario = ghg_result.adjusted_matrix
    
    # Water use changes: Agriculture becomes more efficient (-20%)
    new_water = np.array([100 * 0.8, 50, 20])
    
    water_result = balancer.adjust_satellite_data(
        water_baseline,
        [new_water]
    )
    water_scenario = water_result.adjusted_matrix
    
    print(f"Scenario GHG intensities: {ghg_scenario}")
    print(f"Scenario water intensities: {water_scenario}")
    
    # Step 3: Run SPA analysis on both scenarios
    print("\nStep 3: Running SPA analysis...")
    
    # Baseline SPA
    spa_baseline = SPA(A_baseline, {'GHG': ghg_baseline, 'Water': water_baseline}, sector_names)
    baseline_ghg = spa_baseline.analyze('Manufacturing', depth=5, satellite='GHG')
    baseline_water = spa_baseline.analyze('Manufacturing', depth=5, satellite='Water')
    
    # Scenario SPA
    spa_scenario = SPA(A_scenario, {'GHG': ghg_scenario, 'Water': water_scenario}, sector_names)
    scenario_ghg = spa_scenario.analyze('Manufacturing', depth=5, satellite='GHG')
    scenario_water = spa_scenario.analyze('Manufacturing', depth=5, satellite='Water')
    
    # Step 4: Compare results
    print("\nStep 4: Scenario comparison...")
    
    print(f"\nManufacturing sector analysis:")
    print(f"{'Metric':<25} {'Baseline':<12} {'Scenario':<12} {'Change':<12}")
    print("-" * 65)
    
    # GHG analysis
    baseline_ghg_total = baseline_ghg.total_intensity
    scenario_ghg_total = scenario_ghg.total_intensity
    ghg_change = (scenario_ghg_total - baseline_ghg_total) / baseline_ghg_total * 100
    
    print(f"{'GHG total intensity':<25} {baseline_ghg_total:<12.4f} {scenario_ghg_total:<12.4f} {ghg_change:>8.1f}%")
    
    # Water analysis
    baseline_water_total = baseline_water.total_intensity
    scenario_water_total = scenario_water.total_intensity
    water_change = (scenario_water_total - baseline_water_total) / baseline_water_total * 100
    
    print(f"{'Water total intensity':<25} {baseline_water_total:<12.1f} {scenario_water_total:<12.1f} {water_change:>8.1f}%")
    
    # Top pathways comparison
    print(f"\nTop 3 GHG pathways - Baseline:")
    for i, path in enumerate(baseline_ghg.top(3)):
        print(f"  {i+1}. {path}")
    
    print(f"\nTop 3 GHG pathways - Scenario:")
    for i, path in enumerate(scenario_ghg.top(3)):
        print(f"  {i+1}. {path}")
    
    # Calculate pathway changes
    print(f"\nPathway analysis:")
    print(f"Number of significant pathways (baseline): {len(baseline_ghg.paths)}")
    print(f"Number of significant pathways (scenario): {len(scenario_ghg.paths)}")


def concordance_scenario_example():
    """
    Example: Using IPF with sector concordance for regional scenario analysis.
    
    Shows how to:
    1. Map between different sector classifications
    2. Apply scenario adjustments in target classification
    3. Transform back to source classification for SPA analysis
    """
    if not HAS_IPF:
        return
    
    print("\n=== IPF + Concordance: Regional Scenario Analysis ===")
    
    # Step 1: Create synthetic data
    print("Step 1: Setting up multi-regional data...")
    
    # Original IO table in detailed sector classification (ANZSIC - 4 sectors)
    io_detailed = np.array([
        [100, 30, 20, 10],   # Agriculture
        [25, 200, 40, 15],   # Mining
        [15, 35, 150, 25],   # Manufacturing
        [10, 20, 30, 80]     # Services
    ])
    
    # Simplified sector classification (IOIG - 2 sectors)
    io_simplified = np.array([
        [250, 100],          # Primary (Agriculture + Mining)
        [80, 260]            # Secondary (Manufacturing + Services)
    ])
    
    # Concordance mapping (ANZSIC → IOIG)
    concordance_data = {
        'ANZSIC': ['Agriculture', 'Mining', 'Manufacturing', 'Services'],
        'IOIG': ['Primary', 'Secondary'],
        'value': [100, 80, 40, 60]  # Allocation weights
    }
    
    # Create concordance matrix manually for this example
    concordance_matrix = np.array([
        [1.0, 0.0],  # Agriculture → Primary
        [0.8, 0.2],  # Mining → 80% Primary, 20% Secondary
        [0.3, 0.7],  # Manufacturing → 30% Primary, 70% Secondary
        [0.1, 0.9]   # Services → 10% Primary, 90% Secondary
    ])
    
    # Environmental data in detailed classification
    ghg_detailed = np.array([0.5, 2.5, 1.8, 0.6])  # GHG by ANZSIC sector
    
    print(f"Detailed IO table (ANZSIC):\n{io_detailed}")
    print(f"Concordance matrix:\n{concordance_matrix}")
    print(f"Detailed GHG intensities: {ghg_detailed}")
    
    # Step 2: Apply scenario in simplified classification
    print("\nStep 2: Applying green scenario in simplified classification...")
    
    # Scenario: Increase primary sector output by 20%, decrease secondary by 10%
    new_simplified_totals = np.array([1.2, 0.9]) * io_simplified.sum(axis=1)
    
    # Transform detailed IO table to simplified classification
    io_transformed = concordance_matrix.T @ io_detailed @ concordance_matrix
    
    print(f"Transformed IO table (IOIG):\n{io_transformed}")
    print(f"Scenario totals (IOIG): {new_simplified_totals}")
    
    # Apply IPF in simplified classification
    balancer = IPFBalancer()
    result = balancer.adjust_io_matrix(
        io_transformed,
        new_simplified_totals,
        new_simplified_totals
    )
    
    io_scenario_simplified = result.adjusted_matrix
    print(f"Adjusted IO table (IOIG):\n{io_scenario_simplified}")
    
    # Step 3: Transform back to detailed classification
    print("\nStep 3: Transforming back to detailed classification...")
    
    io_scenario_detailed = concordance_matrix @ io_scenario_simplified @ concordance_matrix.T
    
    print(f"Scenario IO table (ANZSIC):\n{io_scenario_detailed}")
    
    # Transform environmental data
    ghg_scenario = concordance_matrix @ new_simplified_totals / new_simplified_totals.sum() * ghg_detailed.sum()
    # Alternative: apply technology changes in simplified classification
    ghg_technology_factor = np.array([0.8, 0.7])  # 20% and 30% reduction in primary/secondary
    ghg_scenario = concordance_matrix @ ghg_technology_factor * ghg_detailed.sum() / len(ghg_detailed)
    
    print(f"Scenario GHG intensities: {ghg_scenario}")
    
    # Step 4: Run SPA analysis
    print("\nStep 4: Running SPA analysis...")
    
    sector_names = ['Agriculture', 'Mining', 'Manufacturing', 'Services']
    
    # Baseline SPA
    spa_baseline = SPA(io_detailed / io_detailed.sum(axis=1)[:, np.newaxis], ghg_detailed, sector_names)
    baseline_paths = spa_baseline.analyze('Manufacturing', depth=4)
    
    # Scenario SPA
    A_scenario = io_scenario_detailed / io_scenario_detailed.sum(axis=1)[:, np.newaxis]
    spa_scenario = SPA(A_scenario, ghg_scenario, sector_names)
    scenario_paths = spa_scenario.analyze('Manufacturing', depth=4)
    
    # Compare results
    print(f"\nManufacturing sector - SPA comparison:")
    print(f"Baseline total intensity: {baseline_paths.total_intensity:.4f}")
    print(f"Scenario total intensity: {scenario_paths.total_intensity:.4f}")
    
    change_pct = (scenario_paths.total_intensity - baseline_paths.total_intensity) / baseline_paths.total_intensity * 100
    print(f"Change: {change_pct:+.1f}%")
    
    print(f"\nPathway coverage:")
    print(f"Baseline: {baseline_paths.coverage:.1%} ({len(baseline_paths.paths)} paths)")
    print(f"Scenario: {scenario_paths.coverage:.1%} ({len(scenario_paths.paths)} paths)")


def run_integration_examples():
    """Run all integration examples."""
    if not HAS_IPF:
        print("Required functionality not available.")
        return
    
    print("IPF + SPA Integration Examples")
    print("=" * 50)
    
    scenario_analysis_example()
    concordance_scenario_example()
    
    print("\n" + "=" * 50)
    print("Integration examples completed!")
    print("\nKey benefits of IPF + SPA integration:")
    print("• Create realistic economic scenarios with preserved structure")
    print("• Analyze environmental impacts under different futures")
    print("• Support policy analysis with quantitative pathways")
    print("• Enable scenario-based structural path analysis")


if __name__ == "__main__":
    run_integration_examples()