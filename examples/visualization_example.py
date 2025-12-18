"""
Example: Visualizing SPA Results with Icicle Plots

This example demonstrates how to use the icicle plot visualization
to display supply chain analysis results in an interactive HTML plot.
"""

import numpy as np
from fastspa import SPA, icicle_plot, sunburst_plot, sector_contribution_chart


def create_dummy_data(n_sectors=10):
    """Create dummy EEIO data for demonstration."""
    # Create a technical coefficients matrix (A matrix)
    # In a real scenario, this would come from actual EEIO data
    A = np.random.uniform(0, 0.1, size=(n_sectors, n_sectors))
    np.fill_diagonal(A, 0)  # No self-loops
    
    # Ensure Leontief convergence: ||A|| < 1 (max row sum < 1)
    A = A / (np.sum(A, axis=0) * 1.2)
    
    # Create emissions intensity vector
    emissions = np.random.uniform(0.1, 2.0, n_sectors)
    
    # Create realistic sector names (e.g., from a typical EEIO classification)
    realistic_sectors = [
        "Agriculture, forestry, fishing",
        "Mining and quarrying",
        "Food, beverage and tobacco",
        "Textiles, wearing apparel",
        "Wood and paper products",
        "Petroleum and chemical products",
        "Non-metallic mineral products",
        "Metals and metal products",
        "Machinery and equipment",
        "Transport equipment",
        "Other manufactured goods",
        "Electricity, gas, steam",
    ]
    
    sectors = realistic_sectors[:n_sectors] if n_sectors <= len(realistic_sectors) else [
        f"Sector_{i:02d}" for i in range(n_sectors)
    ]
    
    return A, emissions, sectors


def example_single_sector():
    """Example 1: Analyze a single target sector and visualize with icicle plot."""
    print("\n" + "="*60)
    print("Example 1: Single Sector Icicle Visualization")
    print("="*60)
    
    # Create dummy data
    A, emissions, sectors = create_dummy_data(n_sectors=12)
    
    # Initialize SPA
    spa = SPA(A, emissions, sectors=sectors)
    
    # Analyze a specific sector (1-based indexing)
    target_sector_idx = 6  # Will be converted to 0-based internally
    paths = spa.analyze(
        sector=target_sector_idx,
        depth=5,
        threshold=0.001,  # 0.1% of total intensity
    )
    
    print(f"\nAnalyzing {sectors[target_sector_idx]}")
    print(paths.summary())
    
    # Generate icicle plot
    print("\nGenerating icicle plot...")
    fig = icicle_plot(
        paths,
        output_html="./icicle_example.html",
        title=f"Supply Chain: {sectors[target_sector_idx]}"
    )
    print("✓ Icicle plot saved to: ./icicle_example.html")
    
    # Generate sunburst plot
    print("\nGenerating sunburst plot...")
    fig = sunburst_plot(
        paths,
        output_html="./sunburst_example.html",
        title=f"Supply Chain Sunburst: {sectors[target_sector_idx]}",
        max_depth=4
    )
    print("✓ Sunburst plot saved to: ./sunburst_example.html")
    
    # Generate sector contribution chart
    print("\nGenerating sector contribution chart...")
    fig = sector_contribution_chart(
        paths,
        output_html="./sector_contribution_example.html",
        top_n=10
    )
    print("✓ Sector contribution chart saved to: ./sector_contribution_example.html")


def example_multiple_satellites():
    """Example 2: Multiple satellites and combined visualization."""
    print("\n" + "="*60)
    print("Example 2: Multiple Satellites Comparison")
    print("="*60)
    
    # Create dummy data with multiple satellites
    A, _, sectors = create_dummy_data(n_sectors=10)
    
    # Create multiple satellite indicators (ghg, water, energy)
    satellites = {
        "ghg": np.random.uniform(0.1, 2.0, len(sectors)),
        "water": np.random.uniform(0.05, 1.5, len(sectors)),
        "energy": np.random.uniform(0.2, 3.0, len(sectors)),
    }
    
    # Initialize SPA
    spa = SPA(A, satellites, sectors=sectors)
    
    # Analyze multiple sectors (1-based indexing)
    target_sectors = [1, 4, 8]
    
    for target in target_sectors:
        print(f"\n--- Analyzing {sectors[target]} ---")
        
        # Get results for all satellites
        result = spa.analyze(sector=target, depth=4, threshold=0.001)
        
        # Generate combined icicle plot
        print(f"Generating visualization for {sectors[target]}...")
        icicle_plot(
            result,
            output_html=f"./icicle_multi_{target}.html",
            title=f"Multi-Satellite Supply Chain: {sectors[target]}"
        )
        print(f"✓ Saved to: ./icicle_multi_{target}.html")


def example_system_wide_analysis():
    """Example 3: System-wide (Leontief) analysis."""
    print("\n" + "="*60)
    print("Example 3: System-Wide (Leontief) Analysis")
    print("="*60)
    
    # Create dummy data
    A, emissions, sectors = create_dummy_data(n_sectors=8)
    
    # Initialize SPA in system_wide mode (uses Leontief inverse)
    spa = SPA(A, emissions, sectors=sectors, mode="system_wide")
    
    print(f"\nSPA Mode: {spa.mode}")
    print(f"Number of sectors: {spa.n_sectors}")
    
    # Analyze with system-wide approach (1-based indexing)
    paths = spa.analyze(
        sector=3,  # 1-based sector index
        depth=6,
        threshold=0.5,  # 0.5% threshold
        threshold_type="percentage"
    )
    
    print(paths.summary())
    
    # Generate visualization
    print("\nGenerating system-wide analysis visualization...")
    icicle_plot(
        paths,
        output_html="./icicle_system_wide.html",
        title="System-Wide Supply Chain Analysis"
    )
    print("✓ Saved to: ./icicle_system_wide.html")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FastSPA Visualization Examples")
    print("="*60)
    
    try:
        example_single_sector()
        example_multiple_satellites()
        example_system_wide_analysis()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Open the generated HTML files in your browser to view the plots.")
        print("="*60)
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install plotly: pip install plotly")
