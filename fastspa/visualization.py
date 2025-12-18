"""
Visualization module for fastspa.

Provides tools for visualizing supply chain analysis results, including:
- Icicle plots for hierarchical supply chain decomposition
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional visualization libraries
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def icicle_plot(
    paths_dict: Union[Dict[str, "PathCollection"], "PathCollection"],
    output_html: Optional[str] = None,
    title: str = "Supply Chain Analysis (Icicle Plot)",
    colorscale: str = "Plasma",
    include_stage_0: bool = True,
) -> Optional[go.Figure]:
    """
    Generate an interactive icicle plot visualization from SPA results.
    
    Creates a hierarchical visualization where:
    - Rectangle area represents contribution magnitude
    - Color intensity represents intensity value
    - Hierarchical structure shows supply chain depth
    
    Args:
        paths_dict: Either:
            - PathCollection object (single sector)
            - SPAResult/dict mapping satellite names to PathCollection objects
        output_html: Optional path to save HTML output
        title: Plot title
        colorscale: Plotly colorscale name
        include_stage_0: Whether to include the root (Stage 0) in hierarchy
        
    Returns:
        Plotly Figure object (or None if saving to file)
        
    Raises:
        ImportError: If plotly is not installed
        
    Example:
        >>> from fastspa import SPA
        >>> spa = SPA(A_matrix, emissions, sectors=sector_names)
        >>> paths = spa.analyze(sector=0, depth=6)
        >>> fig = icicle_plot(paths, output_html="supply_chain.html")
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for icicle_plot(). Install with: pip install plotly")
    
    # Handle both single PathCollection and SPAResult (dict of PathCollections)
    if hasattr(paths_dict, 'paths'):
        # Single PathCollection
        path_collections = {"primary": paths_dict}
    else:
        # Dictionary of PathCollections (from SPAResult)
        path_collections = paths_dict
    
    # Build icicle data structure
    ids: List[str] = []
    parents: List[str] = []
    labels: List[str] = []
    values: List[float] = []
    colors: List[float] = []
    hover_texts: List[str] = []
    
    # Create synthetic root
    synthetic_root_id = "root"
    node_counter = 0
    
    total_by_satellite: Dict[str, float] = {}
    
    for satellite_name, paths_collection in path_collections.items():
        target_sector = paths_collection.target_sector
        sector_name = (
            paths_collection.metadata.sector_name
            if paths_collection.metadata and paths_collection.metadata.sector_name
            else f"Sector_{target_sector}"
        )
        
        # Build path hierarchy
        node_id_map: Dict[Tuple[int, ...], str] = {}  # path nodes -> unique id
        node_labels: Dict[str, str] = {}
        node_values: Dict[str, float] = defaultdict(float)
        node_parents: Dict[str, str] = {}
        
        for path in paths_collection.paths:
            if not include_stage_0 and len(path.nodes) == 1:
                continue
            
            current_parent = f"{satellite_name}_root"
            
            for stage_idx, sector_idx in enumerate(path.nodes):
                # Create a hierarchical key that includes all stages up to this point
                path_key = (satellite_name, ) + path.nodes[:stage_idx + 1]
                
                if path_key not in node_id_map:
                    node_counter_str = f"node_{node_counter}"
                    node_counter += 1
                    node_id_map[path_key] = node_counter_str
                    
                    # Label: stage and sector
                    stage_label = f"Stage {stage_idx}: {sector_idx}"
                    if path.sectors:
                        stage_label = f"Stage {stage_idx}: {path.sectors[stage_idx]}"
                    
                    node_labels[node_counter_str] = stage_label
                    node_parents[node_counter_str] = current_parent
                
                node_id = node_id_map[path_key]
                node_values[node_id] += path.contribution
                current_parent = node_id
        
        # Add all nodes for this satellite
        satellite_root_id = f"{satellite_name}_root"
        satellite_total = sum(node_values.values())
        total_by_satellite[satellite_name] = satellite_total
        
        # Add satellite root
        ids.append(satellite_root_id)
        parents.append(synthetic_root_id)
        labels.append(f"{satellite_name.replace('_', ' ').title()} ({sector_name})")
        values.append(satellite_total)
        colors.append(satellite_total)
        hover_texts.append(
            f"<b>{satellite_name.replace('_', ' ').title()}</b><br>"
            f"Sector: {sector_name}<br>"
            f"Total: {satellite_total:.4f} ({satellite_total/sum(total_by_satellite.values())*100:.1f}%)"
            if total_by_satellite else ""
        )
        
        # Add sector nodes
        for node_id in node_id_map.values():
            ids.append(node_id)
            parents.append(node_parents[node_id])
            
            # Use the pre-computed label (which includes sector names if available)
            label = node_labels[node_id]
            
            labels.append(label)
            values.append(node_values[node_id])
            colors.append(node_values[node_id])
            hover_texts.append(
                f"<b>{label}</b><br>"
                f"Contribution: {node_values[node_id]:.4f}"
            )
    
    # Add root node
    total_all = sum(total_by_satellite.values())
    ids.insert(0, synthetic_root_id)
    parents.insert(0, "")
    labels.insert(0, "Supply Chains")
    values.insert(0, total_all if total_all > 0 else 1)
    colors.insert(0, total_all if total_all > 0 else 1)
    hover_texts.insert(0, f"<b>All Supply Chains</b><br>Total: {total_all:.4f}")
    
    # Create the figure
    fig = go.Figure(go.Icicle(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale=colorscale,
            cmid=np.median([v for v in values if v > 0]) if any(v > 0 for v in values) else 0,
            colorbar=dict(title="Contribution")
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        textinfo="label+value+percent parent",
        tiling=dict(orientation='v'),
        branchvalues="total",
        root_color="lightgrey"
    ))
    
    fig.update_layout(
        title=title,
        margin=dict(t=50, l=25, r=25, b=25),
        height=800,
        font=dict(size=11),
        showlegend=False,
    )
    
    if output_html is not None:
        fig.write_html(output_html)
        return None
    
    return fig


def sunburst_plot(
    paths_dict: Union[Dict[str, "PathCollection"], "PathCollection"],
    output_html: Optional[str] = None,
    title: str = "Supply Chain Analysis (Sunburst)",
    colorscale: str = "Viridis",
    max_depth: Optional[int] = None,
) -> Optional[go.Figure]:
    """
    Generate an interactive sunburst plot visualization from SPA results.
    
    Similar to icicle plot but with circular/radial layout.
    
    Args:
        paths_dict: PathCollection or dict of PathCollections
        output_html: Optional path to save HTML output
        title: Plot title
        colorscale: Plotly colorscale name
        max_depth: Maximum supply chain depth to show (None = all)
        
    Returns:
        Plotly Figure object (or None if saving to file)
        
    Raises:
        ImportError: If plotly is not installed
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for sunburst_plot(). Install with: pip install plotly")
    
    # Reuse icicle plot logic but with sunburst
    if hasattr(paths_dict, 'paths'):
        path_collections = {"primary": paths_dict}
    else:
        path_collections = paths_dict
    
    ids: List[str] = []
    parents: List[str] = []
    labels: List[str] = []
    values: List[float] = []
    colors: List[float] = []
    
    synthetic_root_id = "root"
    ids.append(synthetic_root_id)
    parents.append("")
    labels.append("Supply Chains")
    
    total_value = 0.0
    
    for satellite_name, paths_collection in path_collections.items():
        target_sector = paths_collection.target_sector
        sector_name = (
            paths_collection.metadata.sector_name 
            if paths_collection.metadata and paths_collection.metadata.sector_name
            else f"Sector_{target_sector}"
        )
        
        satellite_root_id_local = f"{satellite_name}_{target_sector}"
        ids.append(satellite_root_id_local)
        parents.append(synthetic_root_id)
        labels.append(f"{satellite_name.title()} ({sector_name})")
        
        satellite_value = sum(p.contribution for p in paths_collection.paths)
        values.append(satellite_value)
        colors.append(satellite_value)
        total_value += satellite_value
        
        # Add paths (limited by depth if specified)
        for path_idx, path in enumerate(paths_collection.paths):
            if max_depth and path.depth > max_depth:
                continue
            
            current_parent = satellite_root_id_local
            path_label = " → ".join(path.sectors) if path.sectors else f"Path {path_idx}"
            
            node_id = f"path_{satellite_name}_{path_idx}"
            ids.append(node_id)
            parents.append(current_parent)
            labels.append(f"{path_label} ({path.contribution:.2%})")
            values.append(path.contribution)
            colors.append(path.direct_intensity)
    
    values.insert(0, total_value)
    colors.insert(0, total_value)
    
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale=colorscale,
            cmid=np.median([v for v in colors if v > 0]) if any(v > 0 for v in colors) else 0,
        ),
        branchvalues="total",
    ))
    
    fig.update_layout(
        title=title,
        margin=dict(t=50, l=25, r=25, b=25),
        height=800,
    )
    
    if output_html is not None:
        fig.write_html(output_html)
        return None
    
    return fig


def sector_contribution_chart(
    paths_collection: "PathCollection",
    output_html: Optional[str] = None,
    title: Optional[str] = None,
    top_n: int = 15,
) -> Optional[go.Figure]:
    """
    Create a bar chart showing top contributing sectors.
    
    Aggregates path contributions by leaf sector (where emissions occur).
    
    Args:
        paths_collection: PathCollection from SPA analysis
        output_html: Optional path to save HTML output
        title: Plot title (auto-generated if None)
        top_n: Number of top sectors to display
        
    Returns:
        Plotly Figure object (or None if saving to file)
        
    Raises:
        ImportError: If plotly not installed
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for sector_contribution_chart(). Install with: pip install plotly")
    
    # Aggregate by leaf sector
    contributions = defaultdict(float)
    sector_names_map: Dict[int, str] = {}
    
    for path in paths_collection.paths:
        leaf_sector = path.leaf
        contributions[leaf_sector] += path.contribution * paths_collection.total_intensity
        
        if path.sectors:
            sector_names_map[leaf_sector] = path.sectors[-1]
    
    # Sort and get top N
    sorted_sectors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    sector_indices = [s[0] for s in sorted_sectors]
    sector_contribs = [s[1] for s in sorted_sectors]
    sector_labels = [
        sector_names_map.get(idx, f"Sector {idx}") 
        for idx in sector_indices
    ]
    
    if title is None:
        sat_name = paths_collection.satellite_name
        title = f"Top {top_n} Contributing Sectors ({sat_name})"
    
    fig = go.Figure(data=[
        go.Bar(
            x=sector_labels,
            y=sector_contribs,
            marker_color="indianred",
            hovertemplate="<b>%{x}</b><br>Contribution: %{y:.6g}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Sector",
        yaxis_title="Contribution to Total Intensity",
        height=500,
        showlegend=False,
    )
    
    if output_html is not None:
        fig.write_html(output_html)
        return None
    
    return fig


__all__ = [
    "icicle_plot",
    "sunburst_plot", 
    "sector_contribution_chart",
]
