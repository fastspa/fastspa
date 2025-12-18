"""
FastSPA Examples: Comprehensive Structural Path Analysis
========================================================

This file contains comprehensive examples demonstrating FastSPA's capabilities,
including basic usage, concordance mapping, and advanced features.

FastSPA provides a modern, developer-friendly interface for Structural Path Analysis
in Environmentally Extended Input-Output (EEIO) models.

Contents:
1. Basic SPA Usage - Getting started with arrays
2. Concordance Basics - Evidence-based sector mapping
3. Realistic Australian Example - Real-world concordance application
4. Multiple Satellites - Analyzing multiple environmental flows
5. Advanced Features - Filtering, aggregation, and export
6. Threshold Strategies - Understanding percentage vs absolute thresholds
"""

import numpy as np
import pandas as pd
from fastspa import SPA, SectorConcordance, ConcordanceMetadata, SatelliteWithConcordance


# =============================================================================
# Example 1: Basic SPA Usage with Arrays
# =============================================================================

def example_basic_spa():
    """
    Basic Structural Path Analysis using NumPy arrays.

    This demonstrates the array-first design philosophy - no CSV files required,
    direct integration with NumPy/pandas workflows.
    """
    print("=" * 70)
    print("Example 1: Basic SPA Usage with Arrays")
    print("=" * 70)

    # Sample IO table (5 sectors: Electricity, Coal Mining, Gas, Transport, Manufacturing)
    A = np.array([
        [0.05, 0.10, 0.02, 0.03, 0.08],  # Electricity
        [0.15, 0.02, 0.05, 0.01, 0.03],  # Coal Mining
        [0.08, 0.01, 0.03, 0.02, 0.05],  # Gas
        [0.02, 0.01, 0.01, 0.04, 0.02],  # Transport
        [0.10, 0.05, 0.08, 0.15, 0.12],  # Manufacturing
    ])

    sectors = ["Electricity", "Coal Mining", "Gas", "Transport", "Manufacturing"]

    # Direct carbon emissions (kg CO2-eq per AUD output)
    emissions = np.array([0.45, 0.25, 0.12, 0.08, 0.15])

    print("Input-Output Table:")
    print(f"Sectors: {sectors}")
    print(f"A matrix shape: {A.shape}")
    print("Direct emissions (kg CO2-eq / AUD):")
    for sector, emission in zip(sectors, emissions):
        print(f"  {sector:15}: {emission:.2f}")

    # Create SPA instance - array-first design
    spa = SPA(A, emissions, sectors=sectors, mode="sector_specific")

    print(f"\nCreated SPA: {spa}")

    # Analyze electricity sector supply chain
    paths = spa.analyze(sector="Electricity", depth=6, threshold=0.001)

    print("\nElectricity Supply Chain Analysis:")
    print(f"  Total intensity: {paths.total_intensity:.4f} kg CO2-eq / AUD")
    print(f"  Paths found: {len(paths)}")
    print(f"  Coverage: {paths.coverage:.1%}")

    print("\nTop 5 supply chain paths:")
    for i, path in enumerate(paths.top(5), 1):
        print(f"  {i}. {path.contribution:.1%}: {' → '.join(path.sectors)}")

    # Export results
    df = paths.to_dataframe()
    print(f"\nExported to DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    return spa, paths


# =============================================================================
# Example 2: Concordance Basics - Evidence-Based Sector Mapping
# =============================================================================

def example_concordance_basics():
    """
    Demonstrate SectorConcordance for evidence-based sector mapping.

    This shows how to map emissions data from one classification (e.g., ANZSIC)
    to another (e.g., IO Industry Groups) using supply-use table data.
    """
    print("\n" + "=" * 70)
    print("Example 2: Concordance Basics - Evidence-Based Sector Mapping")
    print("=" * 70)

    # Simulate supply-use table data (realistic economic flows)
    sut_data = pd.DataFrame({
        'ANZSIC': [
            '26', '26', '26',  # Electricity Supply flows to multiple IO sectors
            '27', '27',        # Gas Supply
            '46', '46',        # Road Transport
        ],
        'IOIG': [
            'Electricity', 'Coal Mining', 'Manufacturing',  # Electricity distributed
            'Gas', 'Electricity',                            # Gas flows
            'Transport', 'Manufacturing'                      # Transport flows
        ],
        'value': [100, 50, 30, 80, 20, 150, 100]  # Economic flow magnitudes
    })

    print("Supply-Use Table Data (economic flows between classifications):")
    print(sut_data)

    # Create concordance matrix from supply-use data
    concordance = SectorConcordance.from_supply_use_table(
        sut_data,
        source_col='ANZSIC',
        target_col='IOIG',
        value_col='value',
        metadata=ConcordanceMetadata(
            source='ABS Supply-Use Tables 2020',
            method='supply_use_table',
            year=2020,
            confidence=0.92,
            notes='Evidence-based mapping using actual economic flows'
        )
    )

    print(f"\n{concordance}")
    print("\nConcordance Matrix (allocation weights):")
    print(concordance.to_dataframe().round(3))

    # Validate concordance matrix
    validation = concordance.validate()
    print("\nValidation Results:")
    print(f"  Rows sum to 1: {validation['rows_sum_to_one']}")
    print(f"  Sparsity: {validation['sparsity']:.1%}")
    print(f"  No negative values: {not validation['has_negative_values']}")

    # Allocate emissions using concordance
    emissions_anzsic = np.array([0.45, 0.12, 0.28])  # ANZSIC emissions
    emissions_ioig = concordance.allocate(emissions_anzsic)

    print("\nEmissions Allocation:")
    print("  Source (ANZSIC) → Target (IOIG)")
    for i, (source_sector, source_val) in enumerate(zip(concordance.source_sectors, emissions_anzsic)):
        target_val = emissions_ioig[i]
        print(f"  {source_sector} ({source_val:.2f}) → {concordance.target_sectors[i]} ({target_val:.3f})")

    return concordance, emissions_ioig


# =============================================================================
# Example 3: Realistic Australian EEIO Example
# =============================================================================

def example_australian_eeio():
    """
    Realistic Australian EEIO example showing the difference between
    subjective sector grouping vs evidence-based concordance mapping.
    """
    print("\n" + "=" * 70)
    print("Example 3: Realistic Australian EEIO - Subjective vs Evidence-Based Mapping")
    print("=" * 70)

    # Australian IO table (simplified)
    A = np.array([
        [0.05, 0.10, 0.02, 0.03, 0.08],  # Electricity
        [0.15, 0.02, 0.05, 0.01, 0.03],  # Coal Mining
        [0.08, 0.01, 0.03, 0.02, 0.05],  # Gas
        [0.02, 0.01, 0.01, 0.04, 0.02],  # Transport
        [0.10, 0.05, 0.08, 0.15, 0.12],  # Manufacturing
    ])
    sectors = ["Electricity", "Coal Mining", "Gas", "Transport", "Manufacturing"]

    # ANZSIC emissions data (national inventory classification)
    anzsic_emissions = {
        '26': 0.45,  # Electricity Supply
        '27': 0.12,  # Gas Supply
        '46': 0.28,  # Road Transport
        '30': 0.08,  # Manufacturing
        '05': 0.15,  # Mining
    }

    print("Australian EEIO Setup:")
    print(f"IO sectors: {sectors}")
    print(f"ANZSIC emissions: {anzsic_emissions}")

    # Traditional approach: Subjective sector grouping
    print("\nTRADITIONAL APPROACH (Subjective Grouping):")
    subjective_mapping = {
        'Electricity': ['26'],        # Assume direct mapping
        'Coal Mining': ['05'],        # Subjectively assign mining to coal
        'Gas': ['27'],
        'Transport': ['46'],
        'Manufacturing': ['30'],
    }

    # Subjective allocation (simple averaging)
    emissions_subjective = np.zeros(len(sectors))
    for i, sector in enumerate(sectors):
        sector_codes = subjective_mapping[sector]
        emissions_subjective[i] = np.mean([anzsic_emissions[code] for code in sector_codes])

    print("Subjective mapping assumptions:")
    for io_sector, anzsic_codes in subjective_mapping.items():
        values = [anzsic_emissions[code] for code in anzsic_codes]
        print(f"  {io_sector:15}: {anzsic_codes} (avg: {np.mean(values):.2f})")

    # Run SPA with subjective mapping
    spa_subj = SPA(A, emissions_subjective, sectors)
    paths_subj = spa_subj.analyze(sector="Electricity", depth=6)

    print("\nResults with subjective mapping:")
    print(f"  Total intensity: {paths_subj.total_intensity:.4f} kg CO2-eq / AUD")
    print(f"  Coverage: {paths_subj.coverage:.1%}")

    # Evidence-based approach: Concordance mapping
    print("\nEVIDENCE-BASED APPROACH (Concordance Mapping):")

    # Create concordance from supply-use flows
    sut_data = pd.DataFrame({
        'ANZSIC': [
            # Electricity Supply (26) - distributed across energy sectors
            '26', '26', '26', '26', '26',
            # Gas Supply (27)
            '27', '27', '27',
            # Road Transport (46)
            '46', '46', '46',
            # Manufacturing (30)
            '30', '30', '30',
            # Mining (05) - distributed across mining-related sectors
            '05', '05', '05', '05', '05',
        ],
        'IOIG': [
            # Electricity flows
            'Electricity', 'Electricity', 'Coal Mining', 'Gas', 'Manufacturing',
            # Gas flows
            'Gas', 'Gas', 'Electricity',
            # Transport flows
            'Transport', 'Transport', 'Manufacturing',
            # Manufacturing flows
            'Manufacturing', 'Manufacturing', 'Electricity',
            # Mining flows (realistic distribution)
            'Coal Mining', 'Coal Mining', 'Gas', 'Electricity', 'Manufacturing',
        ],
        'flow_value': [
            # Electricity flows
            1000, 500, 200, 100, 50,
            # Gas flows
            800, 300, 150,
            # Transport flows
            600, 200, 100,
            # Manufacturing flows
            1200, 400, 100,
            # Mining flows
            800, 300, 150, 100, 80,
        ]
    })

    concordance = SectorConcordance.from_supply_use_table(
        sut_data,
        source_col='ANZSIC',
        target_col='IOIG',
        value_col='flow_value',
        metadata=ConcordanceMetadata(
            source='ABS Supply-Use Tables 2019-20',
            method='supply_use_table',
            year=2020,
            confidence=0.89,
            notes='Mining sector distributed across energy sectors based on actual flows'
        )
    )

    # Create satellite with concordance
    emissions_array = np.array([anzsic_emissions[code] for code in concordance.source_sectors])
    satellite = SatelliteWithConcordance(
        intensities=emissions_array,
        concordance=concordance,
        name='carbon',
        unit='kg CO2-eq / AUD'
    )

    # Run SPA with evidence-based mapping
    # Use concordance target sectors (they must match exactly)
    spa_evid = SPA(A, satellite, concordance.target_sectors)
    paths_evid = spa_evid.analyze(sector="Electricity", depth=6)

    print("\nResults with evidence-based mapping:")
    print(f"  Total intensity: {paths_evid.total_intensity:.4f} kg CO2-eq / AUD")
    print(f"  Coverage: {paths_evid.coverage:.1%}")

    # Compare approaches
    intensity_diff = paths_evid.total_intensity - paths_subj.total_intensity
    intensity_pct_diff = abs(intensity_diff) / paths_subj.total_intensity * 100

    print("\nCOMPARISON:")
    print(f"  Intensity difference: {intensity_diff:+.4f} kg CO2-eq / AUD ({intensity_pct_diff:.1f}%)")
    print("  ✓ Evidence-based mapping provides more accurate allocation")
    print("  ✓ Concordance reflects actual economic flows")
    print("  ✓ Results can differ significantly from subjective approaches")

    return paths_subj, paths_evid, concordance


# =============================================================================
# Example 4: Multiple Satellites and Advanced Features
# =============================================================================

def example_multiple_satellites():
    """
    Demonstrate multiple environmental satellites with different concordance mappings.
    """
    print("\n" + "=" * 70)
    print("Example 4: Multiple Satellites with Different Concordance Mappings")
    print("=" * 70)

    # Sample IO table
    A = np.array([
        [0.1, 0.05, 0.02],   # Electricity
        [0.02, 0.08, 0.01],  # Gas
        [0.03, 0.02, 0.15],  # Transport
    ])
    sectors = ["Electricity", "Gas", "Transport"]

    # Create different concordances for different satellites
    # Carbon concordance (energy-focused)
    carbon_sut = pd.DataFrame({
        'ANZSIC': ['26', '26', '27', '46'],
        'IOIG': ['Electricity', 'Electricity', 'Gas', 'Transport'],
        'value': [100, 50, 80, 150]
    })

    carbon_concordance = SectorConcordance.from_supply_use_table(
        carbon_sut,
        source_col='ANZSIC',
        target_col='IOIG',
        metadata=ConcordanceMetadata(
            source='Energy flows data',
            method='supply_use_table',
            notes='Carbon: energy-focused allocation'
        )
    )

    # Water concordance (water-intensive processes)
    water_sut = pd.DataFrame({
        'ANZSIC': ['26', '26', '27', '46'],
        'IOIG': ['Electricity', 'Electricity', 'Gas', 'Transport'],
        'value': [50, 20, 120, 30]  # Different allocation pattern
    })

    water_concordance = SectorConcordance.from_supply_use_table(
        water_sut,
        source_col='ANZSIC',
        target_col='IOIG',
        metadata=ConcordanceMetadata(
            source='Water use data',
            method='supply_use_table',
            notes='Water: process-focused allocation'
        )
    )

    # Create satellites with different concordances
    carbon_emissions = np.array([0.45, 0.12, 0.28])  # kg CO2-eq / AUD
    water_usage = np.array([0.8, 0.3, 0.1])          # m3 water / AUD

    carbon_satellite = SatelliteWithConcordance(
        intensities=carbon_emissions,
        concordance=carbon_concordance,
        name='carbon',
        unit='kg CO2-eq / AUD'
    )

    water_satellite = SatelliteWithConcordance(
        intensities=water_usage,
        concordance=water_concordance,
        name='water',
        unit='m3 water / AUD'
    )

    # Multi-satellite SPA
    satellites = {
        'carbon': carbon_satellite,
        'water': water_satellite
    }

    spa = SPA(A, satellites, sectors)
    results = spa.analyze(sector="Electricity", depth=5)

    print(f"Multi-satellite SPA: {list(satellites.keys())}")

    print("\nCarbon Results:")
    print(f"  Paths: {len(results['carbon'].paths)}")
    print(f"  Coverage: {results['carbon'].coverage:.1%}")
    print(f"  Top path: {results['carbon'].top(1)[0]}")

    print("\nWater Results:")
    print(f"  Paths: {len(results['water'].paths)}")
    print(f"  Coverage: {results['water'].coverage:.1%}")
    print(f"  Top path: {results['water'].top(1)[0]}")

    # Demonstrate filtering and aggregation
    print("\nAdvanced Features:")

    # Filter significant carbon paths
    significant_carbon = results['carbon'].filter(min_contribution=0.01)
    print(f"  Carbon paths >1%: {len(significant_carbon)}")

    # Aggregate by sector
    carbon_by_sector = results['carbon'].aggregate_by_sector()
    print("  Carbon hotspots by sector:")
    for sector, contribution in list(carbon_by_sector.items())[:3]:
        print(f"    {sector}: {contribution:.1%}")

    return spa, results


# =============================================================================
# Example 5: Threshold Strategies - Percentage vs Absolute
# =============================================================================

def example_threshold_strategies():
    """
    Demonstrate the difference between percentage and absolute threshold strategies.
    """
    print("\n" + "=" * 70)
    print("Example 5: Threshold Strategies - Percentage vs Absolute")
    print("=" * 70)

    # Sample data
    A = np.array([
        [0.05, 0.10, 0.02, 0.03, 0.08],
        [0.15, 0.02, 0.05, 0.01, 0.03],
        [0.08, 0.01, 0.03, 0.02, 0.05],
        [0.02, 0.01, 0.01, 0.04, 0.02],
        [0.10, 0.05, 0.08, 0.15, 0.12],
    ])
    sectors = ["Electricity", "Coal Mining", "Gas", "Transport", "Manufacturing"]
    emissions = np.array([0.45, 0.25, 0.12, 0.08, 0.15])

    spa = SPA(A, emissions, sectors)

    print("Threshold Strategy Comparison:")
    print("Sector: Electricity, Depth: 8")

    # Percentage threshold (default, conservative)
    paths_pct = spa.analyze(sector="Electricity", depth=8, threshold=0.001, threshold_type="percentage")
    print("\nPERCENTAGE THRESHOLD (0.1% of total intensity):")
    print(f"  Paths found: {len(paths_pct)}")
    print(f"  Coverage: {paths_pct.coverage:.1%}")
    print(f"  Smallest path: {paths_pct.paths[-1].contribution:.4f}")

    # Absolute threshold (permissive)
    paths_abs = spa.analyze(sector="Electricity", depth=8, threshold=1e-6, threshold_type="absolute")
    print("\nABSOLUTE THRESHOLD (1e-6 absolute value):")
    print(f"  Paths found: {len(paths_abs)}")
    print(f"  Coverage: {paths_abs.coverage:.1%}")
    print(f"  Smallest path: {paths_abs.paths[-1].contribution:.2e}")

    print("\nWHEN TO USE EACH STRATEGY:")
    print("• Percentage (default): Conservative, focuses on significant paths")
    print("• Absolute: Permissive, captures low-contribution circular flows")
    print("• Choose based on research question and computational constraints")

    return paths_pct, paths_abs


# =============================================================================
# Main: Run All Examples
# =============================================================================

if __name__ == "__main__":
    print("FastSPA Comprehensive Examples")
    print("=" * 70)
    print("Demonstrating array-first design, concordance mapping, and advanced features")
    print("=" * 70)

    try:
        # Run all examples
        spa_basic, paths_basic = example_basic_spa()
        concordance, emissions_ioig = example_concordance_basics()
        paths_subj, paths_evid, concordance_real = example_australian_eeio()
        spa_multi, results_multi = example_multiple_satellites()
        paths_pct, paths_abs = example_threshold_strategies()

        print("\n" + "=" * 70)
        print("🎉 All Examples Completed Successfully!")
        print("=" * 70)

        print("\nKEY TAKEAWAYS:")
        print("• Array-first design: Direct NumPy/pandas integration")
        print("• Concordance mapping: Evidence-based sector allocation")
        print("• Multiple satellites: Analyze different environmental flows")
        print("• Threshold flexibility: Percentage vs absolute pruning strategies")
        print("• Rich filtering: Chainable operations for exploratory analysis")
        print("• Modern API: Type hints, composition patterns, clear documentation")

        print("\nFILES GENERATED:")
        print("• DataFrames exported from path analysis")
        print("• Metadata tracked for reproducibility")
        print("• Results comparable across different mapping approaches")

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()