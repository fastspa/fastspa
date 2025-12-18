"""
fastspa - Modern Structural Path Analysis for EEIO

A developer-friendly Python package for Structural Path Analysis.

Example:
    >>> from fastspa import SPA
    >>> paths = SPA(A_matrix, emissions).analyze(sector=42, depth=8)
    >>> paths.to_dataframe()
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Optional, Union, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Optional pandas support
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Optional sparse support
try:
    from scipy import sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import concordance module
from .concordance import (
    ConcordanceMetadata,
    SatelliteWithConcordance,
    SectorConcordance,
)

# Import IPF module
try:
    from .ipf import (
        IPFBalancer,
        IPFResult,
        adjust_io_table_with_concordance,
    )
except ImportError:
    # IPF functionality optional
    pass

__version__ = "1.1.0"
__all__ = [
    "SPA",
    "Path",
    "PathCollection",
    "SPAResult",
    "AnalysisMetadata",
    "UncertaintyStats",
    "MonteCarloResult",
    "ConsequentialResult",
    "TemporalSPAResult",
    "LoopAnalysisResult",
    "from_leontief",
    "from_dataframe",
    "from_io_table",
    "from_csv",
    "SectorConcordance",
    "ConcordanceMetadata",
    "SatelliteWithConcordance",
    "IPFBalancer",
    "IPFResult",
    "adjust_io_table_with_concordance",
    "icicle_plot",
    "sunburst_plot",
    "sector_contribution_chart",
]

# Import visualization functions
try:
    from .visualization import (
        icicle_plot,
        sector_contribution_chart,
        sunburst_plot,
    )
except ImportError:
    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class Path:
    """
    A single supply chain pathway.

    Attributes:
        nodes: Tuple of sector indices in the path (root to leaf)
        contribution: Fraction of total intensity this path represents
        direct_intensity: Direct intensity of the final node
        cumulative_weight: Product of A-matrix coefficients along the path

    Example:
        >>> path.nodes
        (42, 15, 3, 7)
        >>> path.contribution
        0.0234
        >>> path.sectors  # If names were provided
        ('Electricity', 'Coal Mining', 'Transport', 'Steel')
    """

    nodes: tuple[int, ...]
    contribution: float
    direct_intensity: float
    cumulative_weight: float
    _sector_names: Optional[tuple[str, ...]] = field(default=None, repr=False, compare=False)

    @property
    def sectors(self) -> tuple[str, ...]:
        """Sector names for this path (if available)."""
        sector_names = self._sector_names
        if sector_names is not None:
            return tuple(sector_names[i] for i in self.nodes)
        return tuple(f"Sector_{i}" for i in self.nodes)

    @property
    def depth(self) -> int:
        """Number of stages in this path (excluding root)."""
        return len(self.nodes) - 1

    @property
    def root(self) -> int:
        """The target sector (first node)."""
        return self.nodes[0]

    @property
    def leaf(self) -> int:
        """The final upstream sector."""
        return self.nodes[-1]

    def __str__(self) -> str:
        sectors = self.sectors
        arrow_chain = " → ".join(sectors)
        return f"{self.contribution:6.2%} | {arrow_chain}"

    def __len__(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class AnalysisMetadata:
    """
    Metadata documenting an SPA analysis.

    Tracks analysis parameters for reproducibility and audit trails.

    Attributes:
        analysis_date: When the analysis was performed
        sector: Target sector (index or name)
        sector_name: Sector name (if available)
        threshold: Threshold value used
        threshold_type: Type of threshold: 'percentage' or 'absolute'
        max_depth: Maximum supply chain depth explored
        total_intensity: Total intensity for the target sector
        coverage: Fraction of total intensity covered by paths
        n_paths: Number of paths found
        mode: Analysis mode: 'sector_specific' or 'system_wide'
        satellite: Satellite name analyzed
        n_sectors: Total number of sectors in the system
    """

    analysis_date: datetime
    sector: Union[int, str]
    sector_name: Optional[str]
    threshold: float
    threshold_type: str
    max_depth: int
    total_intensity: float
    coverage: float
    n_paths: int
    mode: str
    satellite: str
    n_sectors: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "analysis_date": self.analysis_date.isoformat(),
            "sector": self.sector,
            "sector_name": self.sector_name,
            "threshold": self.threshold,
            "threshold_type": self.threshold_type,
            "max_depth": self.max_depth,
            "total_intensity": self.total_intensity,
            "coverage": self.coverage,
            "n_paths": self.n_paths,
            "mode": self.mode,
            "satellite": self.satellite,
            "n_sectors": self.n_sectors,
        }


@dataclass
class PathCollection:
    """
    A collection of structural paths with analysis methods.

    Supports iteration, slicing, and various export formats.

    Attributes:
        paths: List of Path objects
        target_sector: Index of the target sector
        total_intensity: Total intensity for the target sector
        satellite_name: Name of the satellite analyzed
        metadata: Optional AnalysisMetadata for reproducibility
        coverage: Sum of path contributions (computed automatically)

    Example:
        >>> for path in paths.top(10):
        ...     print(path)
        >>> df = paths.to_dataframe()
        >>> paths.filter(min_contribution=0.01)
    """

    paths: list[Path]
    target_sector: int
    total_intensity: float
    satellite_name: str = "intensity"
    metadata: Optional[AnalysisMetadata] = None
    coverage: float = field(init=False)

    def __post_init__(self):
        self.coverage = sum(p.contribution for p in self.paths)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    @overload
    def __getitem__(self, idx: int) -> Path: ...

    @overload
    def __getitem__(self, idx: slice) -> PathCollection: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Path, PathCollection]:
        if isinstance(idx, slice):
            return PathCollection(
                self.paths[idx],
                self.target_sector,
                self.total_intensity,
                self.satellite_name,
                self.metadata,
            )
        return self.paths[idx]

    def top(self, n: int = 10) -> PathCollection:
        """Return the top N paths by contribution."""
        return self[:n]

    def filter(
        self,
        *,
        min_contribution: Optional[float] = None,
        max_depth: Optional[int] = None,
        contains_sector: Optional[int] = None,
        predicate: Optional[Callable[[Path], bool]] = None,
    ) -> PathCollection:
        """
        Filter paths by various criteria.

        Args:
            min_contribution: Minimum contribution threshold
            max_depth: Maximum path depth
            contains_sector: Must contain this sector index
            predicate: Custom filter function

        Returns:
            New PathCollection with filtered paths
        """
        filtered = self.paths

        if min_contribution is not None:
            filtered = [p for p in filtered if p.contribution >= min_contribution]

        if max_depth is not None:
            filtered = [p for p in filtered if p.depth <= max_depth]

        if contains_sector is not None:
            filtered = [p for p in filtered if contains_sector in p.nodes]

        if predicate is not None:
            filtered = [p for p in filtered if predicate(p)]

        return PathCollection(
            filtered, self.target_sector, self.total_intensity, self.satellite_name, self.metadata
        )

    def group_by_stage(self, stage: int) -> dict[int, PathCollection]:
        """
        Group paths by the sector at a given stage.

        Args:
            stage: Stage index (1 = first-tier suppliers)

        Returns:
            Dictionary mapping sector indices to PathCollections
        """
        groups: dict[int, list[Path]] = {}
        for path in self.paths:
            if len(path.nodes) > stage:
                key = path.nodes[stage]
                groups.setdefault(key, []).append(path)

        return {
            k: PathCollection(
                v, self.target_sector, self.total_intensity, self.satellite_name, self.metadata
            )
            for k, v in groups.items()
        }

    def aggregate_by_sector(self) -> dict[int, float]:
        """
        Aggregate contributions by sector (appearing anywhere in paths).

        Returns:
            Dictionary mapping sector indices to total contribution
        """
        contributions: dict[int, float] = {}
        for path in self.paths:
            # Attribute contribution to the leaf node (where emission occurs)
            leaf = path.leaf
            contributions[leaf] = contributions.get(leaf, 0) + path.contribution
        return contributions

    def summary(self) -> str:
        """Generate a text summary of the analysis."""
        lines = [
            "Structural Path Analysis Results",
            "================================",
            f"Target sector: {self.target_sector}",
            f"Satellite: {self.satellite_name}",
            f"Total intensity: {self.total_intensity:.6g}",
            f"Paths found: {len(self.paths)}",
            f"Coverage: {self.coverage:.2%}",
        ]

        if self.metadata:
            lines.extend(
                [
                    f"Threshold: {self.metadata.threshold} ({self.metadata.threshold_type})",
                    f"Max depth: {self.metadata.max_depth}",
                    f"Analysis date: {self.metadata.analysis_date.isoformat()}",
                ]
            )

        lines.extend(
            [
                "",
                "Top 10 paths:",
            ]
        )
        for path in self.paths[:10]:
            lines.append(f"  {path}")
        return "\n".join(lines)

    def to_dataframe(self):
        """
        Export paths to a pandas DataFrame.

        Returns:
            DataFrame with columns for contribution, intensity, and path stages

        Raises:
            ImportError: If pandas is not installed
        """
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: pip install pandas"
            )

        max(len(p.nodes) for p in self.paths) if self.paths else 0

        records = []
        for path in self.paths:
            record = {
                "contribution": path.contribution,
                "contribution_pct": path.contribution * 100,
                "direct_intensity": path.direct_intensity,
                "cumulative_weight": path.cumulative_weight,
                "depth": path.depth,
            }
            # Add stage columns
            for i, node in enumerate(path.nodes):
                record[f"stage_{i}"] = node
                record[f"stage_{i}_name"] = path.sectors[i]
            records.append(record)

        return pd.DataFrame(records)

    def to_csv(self, path: str, include_metadata: bool = True, **kwargs) -> None:
        """
        Export to CSV file with optional metadata header.

        Args:
            path: Output file path
            include_metadata: Whether to include metadata as comments
            **kwargs: Additional arguments passed to pandas.to_csv()
        """
        df = self.to_dataframe()

        if include_metadata and self.metadata:
            # Write metadata as comments
            with open(path, "w") as f:
                f.write("# Analysis Metadata\n")
                for key, value in self.metadata.to_dict().items():
                    f.write(f"# {key}: {value}\n")
                f.write("#\n")

            # Append data
            df.to_csv(path, index=False, mode="a", **kwargs)
        else:
            df.to_csv(path, index=False, **kwargs)

    def to_json(
        self, path: Optional[str] = None, include_metadata: bool = True, **kwargs
    ) -> Optional[str]:
        """
        Export to JSON (file or string).

        Args:
            path: Output file path (None = return as string)
            include_metadata: Whether to include metadata
            **kwargs: Additional arguments passed to json.dump()

        Returns:
            JSON string if path is None, otherwise None
        """
        import json

        data = {
            "metadata": self.metadata.to_dict() if self.metadata and include_metadata else None,
            "target_sector": self.target_sector,
            "satellite": self.satellite_name,
            "total_intensity": self.total_intensity,
            "coverage": self.coverage,
            "paths": [
                {
                    "nodes": list(p.nodes),
                    "sectors": list(p.sectors),
                    "contribution": p.contribution,
                    "direct_intensity": p.direct_intensity,
                }
                for p in self.paths
            ],
        }

        if path is not None:
            with open(path, "w") as f:
                json.dump(data, f, **kwargs)
            return None
        return json.dumps(data, **kwargs)

    # -------------------------------------------------------------------------
    # Advanced aggregation (semantic / custom / hierarchical)
    # -------------------------------------------------------------------------

    def semantic_aggregate(
        self,
        mapping: Mapping[Union[int, str], str],
        *,
        stage: Union[int, Literal["leaf", "root"]] = "leaf",
        as_absolute: bool = False,
        unmapped: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Aggregate path contributions into user-defined semantic groups.

        This is a post-processing step: it does not change any SPA calculations.

        Args:
            mapping: Map from sector index (0-based) or sector name -> group label.
            stage:
                - "leaf" (default): aggregate by the emission source sector (leaf).
                - "root": aggregate by the target sector.
                - int: aggregate by a specific stage (0 = root, 1 = tier-1, ...).
            as_absolute: If True, report group totals in satellite units
                (fraction * total_intensity). If False, report fractions.
            unmapped: If provided, sectors not present in mapping are assigned to this
                group. If None, unmapped sectors are skipped.

        Returns:
            Dictionary mapping group label -> total contribution (fraction or absolute).
        """
        scale = self.total_intensity if as_absolute else 1.0

        # Determine whether mapping keys are indices or names
        key_type = None
        for k in mapping.keys():
            key_type = type(k)
            break

        sector_names: Optional[tuple[str, ...]] = None
        if key_type is str:
            for p in self.paths:
                if getattr(p, "_sector_names", None) is not None:
                    sector_names = p._sector_names
                    break
            if sector_names is None:
                raise ValueError("Sector names are required for name-based aggregation")

        out: dict[str, float] = {}
        for p in self.paths:
            if stage == "leaf":
                sector_idx = p.leaf
            elif stage == "root":
                sector_idx = p.root
            else:
                if stage < 0:
                    raise ValueError("stage must be >= 0")
                if len(p.nodes) <= stage:
                    continue
                sector_idx = p.nodes[stage]

            if key_type is str:
                assert sector_names is not None
                group = mapping.get(sector_names[sector_idx])
            else:
                group = mapping.get(int(sector_idx))

            if group is None:
                if unmapped is None:
                    continue
                group = unmapped

            out[group] = out.get(group, 0.0) + p.contribution * scale

        return out

    def hierarchical_aggregate(
        self,
        level_mappings: Sequence[Mapping[Union[int, str], str]],
        *,
        stage: Union[int, Literal["leaf", "root"]] = "leaf",
        as_absolute: bool = False,
        unmapped: Optional[str] = None,
    ) -> dict[tuple[str, ...], float]:
        """Aggregate path contributions into hierarchical (multi-level) groups."""
        if len(level_mappings) == 0:
            raise ValueError("level_mappings must not be empty")

        scale = self.total_intensity if as_absolute else 1.0

        key_type = None
        for k in level_mappings[0].keys():
            key_type = type(k)
            break

        sector_names: Optional[tuple[str, ...]] = None
        if key_type is str:
            for p in self.paths:
                if getattr(p, "_sector_names", None) is not None:
                    sector_names = p._sector_names
                    break
            if sector_names is None:
                raise ValueError("Sector names are required for name-based aggregation")

        out: dict[tuple[str, ...], float] = {}
        for p in self.paths:
            if stage == "leaf":
                sector_idx = p.leaf
            elif stage == "root":
                sector_idx = p.root
            else:
                if stage < 0:
                    raise ValueError("stage must be >= 0")
                if len(p.nodes) <= stage:
                    continue
                sector_idx = p.nodes[stage]

            labels: list[str] = []
            missing = False
            for lvl in level_mappings:
                if key_type is str:
                    assert sector_names is not None
                    label = lvl.get(sector_names[sector_idx])
                else:
                    label = lvl.get(int(sector_idx))

                if label is None:
                    if unmapped is None:
                        missing = True
                        break
                    label = unmapped
                labels.append(label)

            if missing:
                continue

            key = tuple(labels)
            out[key] = out.get(key, 0.0) + p.contribution * scale

        return out

    # -------------------------------------------------------------------------
    # Network science metrics (topology + betweenness + bottlenecks)
    # -------------------------------------------------------------------------

    def edge_weights(
        self,
        *,
        weight: Literal["fraction", "absolute"] = "fraction",
    ) -> dict[tuple[int, int], float]:
        """Build a directed weighted edge list from the extracted paths."""
        scale = self.total_intensity if weight == "absolute" else 1.0
        edges: dict[tuple[int, int], float] = {}

        for p in self.paths:
            if len(p.nodes) < 2:
                continue
            w = p.contribution * scale
            for u, v in zip(p.nodes[:-1], p.nodes[1:]):
                if u == v:
                    continue
                key = (int(u), int(v))
                edges[key] = edges.get(key, 0.0) + w

        return edges

    def network_topology(
        self,
        *,
        weight: Literal["fraction", "absolute"] = "fraction",
        include_isolated: bool = True,
    ) -> dict[str, Any]:
        """Compute basic topology metrics for the path-induced supply-chain graph."""
        edges = self.edge_weights(weight=weight)

        node_set: set[int] = set()
        for p in self.paths:
            node_set.update(p.nodes)

        if include_isolated:
            if self.metadata is not None:
                n_nodes = self.metadata.n_sectors
            elif node_set:
                n_nodes = max(node_set) + 1
            else:
                n_nodes = 0
        else:
            # Keep original sector indices so returned dicts are always keyed by sector index.
            n_nodes = max(node_set) + 1 if node_set else 0

        in_degree = [0] * n_nodes
        out_degree = [0] * n_nodes
        in_strength = [0.0] * n_nodes
        out_strength = [0.0] * n_nodes

        for (u, v), w in edges.items():
            if u >= n_nodes or v >= n_nodes:
                continue
            out_degree[u] += 1
            in_degree[v] += 1
            out_strength[u] += w
            in_strength[v] += w

        denom = n_nodes if include_isolated else max(1, len(node_set))
        density = (len(edges) / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0

        return {
            "n_nodes": n_nodes,
            "n_edges": len(edges),
            "density": density,
            "avg_in_degree": float(sum(in_degree) / denom) if denom else 0.0,
            "avg_out_degree": float(sum(out_degree) / denom) if denom else 0.0,
            "avg_in_strength": float(sum(in_strength) / denom) if denom else 0.0,
            "avg_out_strength": float(sum(out_strength) / denom) if denom else 0.0,
            "in_degree": {i: in_degree[i] for i in range(n_nodes)},
            "out_degree": {i: out_degree[i] for i in range(n_nodes)},
            "in_strength": {i: in_strength[i] for i in range(n_nodes)},
            "out_strength": {i: out_strength[i] for i in range(n_nodes)},
        }

    def betweenness_centrality(
        self,
        *,
        weight: Literal["fraction", "absolute"] = "fraction",
        normalized: bool = True,
        weight_transform: Literal["inverse", "unweighted", "log"] = "inverse",
        include_isolated: bool = True,
    ) -> dict[int, float]:
        """Compute node betweenness centrality on the extracted path graph."""
        edges = self.edge_weights(weight=weight)

        node_set: set[int] = set()
        for p in self.paths:
            node_set.update(p.nodes)

        if include_isolated:
            if self.metadata is not None:
                n_nodes = self.metadata.n_sectors
            elif node_set:
                n_nodes = max(node_set) + 1
            else:
                n_nodes = 0
        else:
            # Keep original sector indices so returned dicts are always keyed by sector index.
            n_nodes = max(node_set) + 1 if node_set else 0

        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
        eps = 1e-15
        for (u, v), w in edges.items():
            if u >= n_nodes or v >= n_nodes or u == v:
                continue
            if w <= 0:
                continue
            if weight_transform == "unweighted":
                length = 1.0
            elif weight_transform == "log":
                length = -float(np.log(w + eps))
            else:  # inverse
                length = 1.0 / float(w)
            adjacency[u].append((v, length))

        bc_arr = _brandes_betweenness_centrality(adjacency, normalized=normalized)
        return {i: float(bc_arr[i]) for i in range(n_nodes)}

    def bottleneck_sectors(
        self,
        top_n: int = 10,
        *,
        weight: Literal["fraction", "absolute"] = "fraction",
        exclude_target: bool = True,
    ) -> list[dict[str, Any]]:
        """Identify bottleneck sectors using betweenness centrality."""
        bc = self.betweenness_centrality(weight=weight, include_isolated=True)
        topo = self.network_topology(weight=weight, include_isolated=True)

        sector_names: Optional[tuple[str, ...]] = None
        for p in self.paths:
            if getattr(p, "_sector_names", None) is not None:
                sector_names = p._sector_names
                break

        items = []
        for i, score in bc.items():
            if exclude_target and i == self.target_sector:
                continue
            items.append((i, score))

        items.sort(key=lambda x: x[1], reverse=True)

        out: list[dict[str, Any]] = []
        for i, score in items[:top_n]:
            out.append(
                {
                    "sector": i,
                    "sector_name": (
                        sector_names[i]
                        if sector_names is not None and i < len(sector_names)
                        else None
                    ),
                    "betweenness": float(score),
                    "in_strength": topo["in_strength"].get(i, 0.0),
                    "out_strength": topo["out_strength"].get(i, 0.0),
                }
            )
        return out

    # -------------------------------------------------------------------------
    # Loop detection & analysis (circular economy / feedback loops)
    # -------------------------------------------------------------------------

    def loop_analysis(
        self,
        *,
        as_absolute: bool = False,
        min_cycle_edges: int = 1,
    ) -> LoopAnalysisResult:
        """Detect and summarize cycles (repeated sectors) in extracted paths.

        A path is considered to contain a loop if any sector index appears more than once.
        Cycles are summarized using the first repeated sector encountered when traversing
        from root to leaf.

        Args:
            as_absolute: If True, report contributions in satellite units
                (fraction * total_intensity). If False, report fractions.
            min_cycle_edges: Minimum cycle length in edges (>= 1). A cycle with
                k edges has k+1 nodes in its cycle representation.

        Returns:
            LoopAnalysisResult
        """
        if min_cycle_edges < 1:
            raise ValueError("min_cycle_edges must be >= 1")

        scale = self.total_intensity if as_absolute else 1.0

        cycles: dict[tuple[int, ...], float] = {}
        sector_participation: dict[int, float] = {}
        loop_paths: list[Path] = []
        loop_share = 0.0

        for p in self.paths:
            seen: dict[int, int] = {}
            found_cycle: Optional[tuple[int, ...]] = None
            for idx, node in enumerate(p.nodes):
                if node in seen:
                    start = seen[node]
                    cycle_nodes = p.nodes[start : idx + 1]
                    if len(cycle_nodes) - 1 >= min_cycle_edges:
                        found_cycle = tuple(int(x) for x in cycle_nodes)
                    break
                seen[node] = idx

            if found_cycle is None:
                continue

            contrib = p.contribution * scale
            loop_paths.append(p)
            loop_share += contrib

            cycles[found_cycle] = cycles.get(found_cycle, 0.0) + contrib
            for node in set(found_cycle):
                sector_participation[node] = sector_participation.get(node, 0.0) + contrib

        return LoopAnalysisResult(
            as_absolute=as_absolute,
            loop_share=loop_share,
            cycles=cycles,
            sector_participation=sector_participation,
            paths=PathCollection(
                loop_paths,
                self.target_sector,
                self.total_intensity,
                self.satellite_name,
                metadata=self.metadata,
            ),
        )

    def __repr__(self) -> str:
        return f"PathCollection({len(self.paths)} paths, coverage={self.coverage:.2%})"


@dataclass
class SPAResult:
    """
    Results for multiple satellites.

    Example:
        >>> result = spa.analyze(sector=42, depth=8)
        >>> result["ghg"].to_dataframe()
        >>> result.satellites
        ['ghg', 'water', 'energy']
    """

    results: dict[str, PathCollection]
    target_sector: int

    @property
    def satellites(self) -> list[str]:
        """List of satellite names."""
        return list(self.results.keys())

    def __getitem__(self, key: str) -> PathCollection:
        return self.results[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def items(self):
        return self.results.items()

    def to_dataframe(self):
        """Combine all satellites into a single DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas required")

        dfs = []
        for name, paths in self.results.items():
            df = paths.to_dataframe()
            df["satellite"] = name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Main SPA Class
# =============================================================================


class SPA:
    """
    Structural Path Analysis for EEIO.

    A modern, NumPy-native interface for decomposing supply chain
    environmental impacts into individual pathways.

    Supports two analysis modes:
    - Sector-specific (A-matrix): Direct supplier relationships
    - System-wide (Leontief): Total (direct + indirect) requirements

    Args:
        A: Technical coefficients matrix (n×n). Can be dense NumPy array
           or scipy sparse matrix.
        intensities: Direct intensities. Either:
            - 1D array of length n (single satellite)
            - Dict mapping satellite names to 1D arrays
            - SatelliteWithConcordance (single satellite with concordance mapping)
            - Dict mapping satellite names to SatelliteWithConcordance objects
        sectors: Optional list/array of sector names. If SatelliteWithConcordance
                objects are provided, sectors should correspond to the target
                classification (IO table sectors).
        mode: Analysis mode - "sector_specific" (default, uses A matrix) or
              "system_wide" (uses Leontief inverse for total requirements)

    Example:
        >>> spa = SPA(A_matrix, emissions)
        >>> paths = spa.analyze(sector=42, depth=8)

        >>> # Multiple satellites
        >>> spa = SPA(A_matrix, {"ghg": ghg, "water": water})
        >>> result = spa.analyze(sector=42, depth=8)
        >>> result["ghg"].top(10)

        >>> # System-wide analysis (Leontief-based)
        >>> spa = SPA(A_matrix, emissions, mode="system_wide")
        >>> paths = spa.analyze(sector=42, depth=8)

        >>> # With concordance mapping
        >>> satellite = SatelliteWithConcordance(emissions, concordance)
        >>> spa = SPA(A_matrix, satellite, sectors=target_sectors)
        >>> paths = spa.analyze(sector="Electricity", depth=8)
    """

    def __init__(
        self,
        A: ArrayLike,
        intensities: Union[
            ArrayLike,
            dict[str, ArrayLike],
            SatelliteWithConcordance,
            dict[str, SatelliteWithConcordance],
        ],
        sectors: Optional[Sequence[str]] = None,
        mode: Literal["sector_specific", "system_wide"] = "sector_specific",
    ):
        # Validate mode
        if mode not in ("sector_specific", "system_wide"):
            raise ValueError(f"mode must be 'sector_specific' or 'system_wide', got {mode}")
        self._mode = mode

        # Store A matrix
        self._A = np.asarray(A) if not (HAS_SCIPY and sparse.issparse(A)) else A
        self._n: int = int(self._A.shape[0])

        if self._A.shape[0] != self._A.shape[1]:
            raise ValueError(f"A matrix must be square, got shape {self._A.shape}")

        # Process intensities - handle SatelliteWithConcordance objects
        self._satellite_objects: Optional[dict[str, SatelliteWithConcordance]] = None

        if isinstance(intensities, SatelliteWithConcordance):
            # Single satellite with concordance
            self._satellite_objects = {intensities.name: intensities}
            self._intensities = {intensities.name: intensities.get_intensities_target()}

            # Validate that target sectors match IO table sectors
            if sectors is not None:
                if len(sectors) != len(intensities.concordance.target_sectors):
                    raise ValueError(
                        f"Sector names ({len(sectors)}) don't match concordance target sectors "
                        f"({len(intensities.concordance.target_sectors)})"
                    )
                # Check sector alignment
                if sectors != list(intensities.concordance.target_sectors):
                    raise ValueError("Sector names must match concordance target sectors exactly")

        elif isinstance(intensities, dict) and all(
            isinstance(v, SatelliteWithConcordance) for v in intensities.values()
        ):
            # Multiple satellites with concordance
            self._satellite_objects = dict(intensities)
            self._intensities = {
                name: satellite.get_intensities_target() for name, satellite in intensities.items()
            }

            # Validate all satellites have same target sectors
            target_sectors_list = [sat.concordance.target_sectors for sat in intensities.values()]
            if not all(sectors == target_sectors_list[0] for sectors in target_sectors_list):
                raise ValueError("All satellites must have the same target sector classification")

            # Validate sector alignment
            if sectors is not None:
                if sectors != list(target_sectors_list[0]):
                    raise ValueError("Sector names must match concordance target sectors exactly")

        elif isinstance(intensities, dict):
            # Regular dict of arrays
            self._intensities = {k: np.asarray(v) for k, v in intensities.items()}
        else:
            # Single array
            self._intensities = {"intensity": np.asarray(intensities)}

        # Validate intensities
        for name, arr in self._intensities.items():
            if arr.shape != (self._n,):
                raise ValueError(f"Intensity '{name}' has shape {arr.shape}, expected ({self._n},)")

        # Store sector names
        self._sectors: Optional[tuple[str, ...]]
        if sectors is not None:
            if len(sectors) != self._n:
                raise ValueError(f"Got {len(sectors)} sector names, expected {self._n}")
            self._sectors = tuple(sectors)
        else:
            self._sectors = None

        # Lazy-computed Leontief inverse
        self._L: Optional[NDArray] = None
        self._total_intensities: Optional[dict[str, NDArray]] = None
        # Lazy-computed sparse structure for faster lookups
        self._A_col_nonzero: Optional[list[NDArray]] = None
        self._A_col_values: Optional[list[NDArray]] = None

    @property
    def n_sectors(self) -> int:
        """Number of sectors in the system."""
        return self._n

    @property
    def mode(self) -> str:
        """Analysis mode: 'sector_specific' (A-matrix) or 'system_wide' (Leontief)."""
        return self._mode

    @property
    def satellites(self) -> list[str]:
        """Names of available satellites."""
        return list(self._intensities.keys())

    @property
    def sectors(self) -> Optional[tuple[str, ...]]:
        """Sector names (if provided)."""
        return self._sectors

    @property
    def satellite_objects(self) -> Optional[dict[str, SatelliteWithConcordance]]:
        """Satellite objects with concordance mappings (if provided)."""
        return self._satellite_objects

    @property
    def A(self) -> NDArray:
        """Technical coefficients matrix."""
        if HAS_SCIPY and sparse.issparse(self._A):
            return self._A.toarray()
        return self._A

    @property
    def L(self) -> NDArray:
        """
        Leontief inverse matrix (I - A)^(-1).

        Computed lazily on first access.
        """
        if self._L is None:
            self._compute_leontief()
        return self._L

    def _compute_leontief(self) -> None:
        """Compute Leontief inverse and total intensities."""
        A_dense = self.A
        identity = np.eye(self._n)
        self._L = np.linalg.inv(identity - A_dense)

        # Compute total intensities for all satellites
        self._total_intensities = {}
        for name, direct in self._intensities.items():
            self._total_intensities[name] = direct @ self._L

    def _precompute_sparse_structure(self) -> None:
        """Precompute sparse column indices and values for fast lookups."""
        if self._A_col_nonzero is not None:
            return

        A = self.A
        self._A_col_nonzero = []
        self._A_col_values = []

        for col in range(self._n):
            column = A[:, col]
            nonzero_idx = np.nonzero(column > 0)[0]
            self._A_col_nonzero.append(nonzero_idx)
            self._A_col_values.append(column[nonzero_idx])

    def total_intensity(self, satellite: Optional[str] = None) -> NDArray:
        """
        Get total intensities for a satellite.

        Args:
            satellite: Satellite name. If None and only one satellite exists,
                      uses that one.

        Returns:
            Array of total intensities for each sector
        """
        if self._total_intensities is None:
            self._compute_leontief()
        assert self._total_intensities is not None

        if satellite is None:
            if len(self._intensities) == 1:
                satellite = next(iter(self._intensities))
            else:
                raise ValueError("Multiple satellites exist, specify which one")

        return self._total_intensities[satellite]

    # -------------------------------------------------------------------------
    # Uncertainty & sensitivity
    # -------------------------------------------------------------------------

    def sensitivity(
        self,
        sector: Union[int, str],
        *,
        satellite: Optional[str] = None,
    ) -> dict[int, float]:
        """
        Sensitivity of the target sector's total intensity to direct intensity inputs.

        Returns elasticities under a multiplicative perturbation of direct intensities:

            elasticity[i] = d(ln total_intensity[target]) / d(ln direct_intensity[i])

        In Leontief form (total = direct @ L), this is:

            elasticity[i] = direct[i] * L[i, target] / total[target]

        Notes:
            - This is exact for the total intensity calculation.
            - It is independent of SPA traversal thresholds because it operates on L.
        """
        sector_idx = self._resolve_sector(sector)

        if self._L is None:
            self._compute_leontief()
        assert self._L is not None
        assert self._total_intensities is not None

        if satellite is None:
            if len(self._intensities) == 1:
                satellite = next(iter(self._intensities))
            else:
                raise ValueError("Multiple satellites exist, specify which one")

        direct = self._intensities[satellite]
        total_target = float(self._total_intensities[satellite][sector_idx])
        if total_target == 0:
            return dict.fromkeys(range(self._n), 0.0)

        elasticities = (direct * self._L[:, sector_idx]) / total_target
        return {i: float(elasticities[i]) for i in range(self._n)}

    def monte_carlo(
        self,
        sector: Union[int, str],
        depth: int = 8,
        *,
        n_samples: int = 200,
        threshold: float = 0.001,
        threshold_type: Literal["percentage", "absolute"] = "percentage",
        satellite: Optional[str] = None,
        intensity_cv: Union[float, ArrayLike] = 0.1,
        A_cv: float = 0.0,
        distribution: Literal["lognormal", "normal"] = "lognormal",
        sampling: Literal["random", "lhs"] = "random",
        ci: tuple[float, float] = (0.025, 0.975),
        seed: Optional[int] = None,
        max_paths: Optional[int] = None,
        include_stage_0: bool = True,
        as_absolute: bool = True,
    ) -> MonteCarloResult:
        """
        Monte Carlo uncertainty propagation for SPA outputs.

        This runs repeated SPA analyses under sampled (uncertain) inputs and reports
        uncertainty on:
            - total intensity of the target sector
            - contributions by emission-source sector (leaf aggregation)

        Args:
            sector: Target sector (index or name).
            depth: SPA traversal depth.
            n_samples: Number of samples.
            threshold: SPA pruning threshold.
            threshold_type: "percentage" or "absolute".
            satellite: Which satellite to analyze. Required if multiple satellites exist.
            intensity_cv: Coefficient of variation (std/mean) for direct intensities.
                Can be a scalar or a vector of length n_sectors.
            A_cv: Optional coefficient of variation for A-matrix coefficients.
                Defaults to 0.0 (no A uncertainty) to avoid unstable inverses.
            distribution: Sampling distribution for uncertain values.
                "lognormal" is recommended for non-negative intensities.
            sampling: "random" (default) or "lhs" (Latin Hypercube Sampling).
                LHS is currently supported for lognormal intensity sampling.
            ci: Quantile bounds for confidence interval reporting.
            seed: Random seed.
            max_paths: Passed through to SPA.analyze for each run.
            include_stage_0: Whether to include the Stage 0 direct path.
            as_absolute: If True, sector contributions are reported in satellite units.
                If False, they are reported as fractions of total intensity.

        Returns:
            MonteCarloResult
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")

        sector_idx = self._resolve_sector(sector)

        if satellite is None:
            if len(self._intensities) == 1:
                satellite = next(iter(self._intensities))
            else:
                raise ValueError("Multiple satellites exist, specify which one")

        base_direct = np.asarray(self._intensities[satellite], dtype=float)
        cv_arr = np.asarray(intensity_cv, dtype=float)
        if cv_arr.shape == ():
            cv_arr = np.full(self._n, float(cv_arr))
        if cv_arr.shape != (self._n,):
            raise ValueError(
                f"intensity_cv must be scalar or shape ({self._n},), got {cv_arr.shape}"
            )

        base_A = self.A

        rng = np.random.default_rng(seed)

        if sampling == "lhs":
            if distribution != "lognormal":
                raise ValueError("LHS sampling is only supported for lognormal distribution")
            u = _latin_hypercube(n_samples, self._n, rng)
            z = _norm_ppf(u)

        total_samples = np.zeros(n_samples, dtype=float)
        sector_samples = np.zeros((n_samples, self._n), dtype=float)

        for s in range(n_samples):
            if distribution == "lognormal":
                sampled_direct = _sample_lognormal_mean_cv(
                    base_direct, cv_arr, rng, z=None if sampling == "random" else z[s]
                )
            elif distribution == "normal":
                sampled_direct = _sample_trunc_normal_mean_cv(base_direct, cv_arr, rng)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

            if A_cv and A_cv > 0:
                sampled_A = _sample_lognormal_mean_cv(base_A, float(A_cv), rng)
            else:
                sampled_A = base_A

            spa_s = SPA(
                sampled_A, {satellite: sampled_direct}, sectors=self._sectors, mode=self._mode
            )
            paths_s = spa_s.analyze(
                sector=sector,
                depth=depth,
                threshold=threshold,
                threshold_type=threshold_type,
                satellite=satellite,
                max_paths=max_paths,
                include_stage_0=include_stage_0,
            )

            if isinstance(paths_s, SPAResult):
                # Should not happen because satellite is specified above, but keep it safe.
                paths_s = paths_s[satellite]

            total_i = float(paths_s.total_intensity)
            total_samples[s] = total_i

            agg = paths_s.aggregate_by_sector()
            for k, frac in agg.items():
                if 0 <= k < self._n:
                    sector_samples[s, k] += float(frac)

            if as_absolute:
                sector_samples[s, :] *= total_i

        total_stats = UncertaintyStats.from_samples(total_samples, ci=ci)
        sector_stats = {
            i: UncertaintyStats.from_samples(sector_samples[:, i], ci=ci) for i in range(self._n)
        }

        return MonteCarloResult(
            satellite=satellite,
            target_sector=sector_idx,
            as_absolute=as_absolute,
            ci=ci,
            samples_total_intensity=total_samples,
            samples_sector_contributions=sector_samples,
            total_intensity=total_stats,
            sector_contributions=sector_stats,
            sector_names=self._sectors,
        )

    def consequential(
        self,
        sector: Union[int, str],
        depth: int = 8,
        *,
        satellite: Optional[str] = None,
        intensity_multiplier: Optional[ArrayLike] = None,
        intensity_delta: Optional[ArrayLike] = None,
        A_multiplier: Optional[ArrayLike] = None,
        A_delta: Optional[ArrayLike] = None,
        threshold: float = 0.001,
        threshold_type: Literal["percentage", "absolute"] = "percentage",
        max_paths: Optional[int] = None,
        include_stage_0: bool = True,
    ) -> ConsequentialResult:
        """Consequential (perturbation) analysis for policy/scenario workflows.

        Runs SPA for a baseline and a perturbed scenario, then reports differences in:
            - total intensity of the target sector
            - absolute contributions by leaf sector

        Perturbations are applied as:
            scenario = (base + delta) * multiplier

        Args:
            sector: Target sector (1-indexed int or name).
            depth: SPA depth.
            satellite: Which satellite to analyze.
            intensity_multiplier: Optional elementwise multiplier for direct intensities.
            intensity_delta: Optional additive delta for direct intensities.
            A_multiplier: Optional elementwise multiplier for A.
            A_delta: Optional additive delta for A.

        Returns:
            ConsequentialResult
        """
        if satellite is None:
            if len(self._intensities) == 1:
                satellite = next(iter(self._intensities))
            else:
                raise ValueError("Multiple satellites exist, specify which one")

        baseline = self.analyze(
            sector,
            depth,
            threshold=threshold,
            threshold_type=threshold_type,
            satellite=satellite,
            max_paths=max_paths,
            include_stage_0=include_stage_0,
        )
        if isinstance(baseline, SPAResult):
            baseline = baseline[satellite]

        base_int = np.asarray(self._intensities[satellite], dtype=float)
        scenario_int = base_int.copy()
        if intensity_delta is not None:
            scenario_int = scenario_int + np.asarray(intensity_delta, dtype=float)
        if intensity_multiplier is not None:
            scenario_int = scenario_int * np.asarray(intensity_multiplier, dtype=float)

        base_A = self.A
        scenario_A = np.asarray(base_A, dtype=float).copy()
        if A_delta is not None:
            scenario_A = scenario_A + np.asarray(A_delta, dtype=float)
        if A_multiplier is not None:
            scenario_A = scenario_A * np.asarray(A_multiplier, dtype=float)

        scenario_spa = SPA(
            scenario_A,
            {satellite: scenario_int},
            sectors=self._sectors,
            mode=self._mode,
        )
        scenario = scenario_spa.analyze(
            sector,
            depth,
            threshold=threshold,
            threshold_type=threshold_type,
            satellite=satellite,
            max_paths=max_paths,
            include_stage_0=include_stage_0,
        )
        if isinstance(scenario, SPAResult):
            scenario = scenario[satellite]

        base_total = float(baseline.total_intensity)
        scen_total = float(scenario.total_intensity)

        base_abs = {k: float(v) * base_total for k, v in baseline.aggregate_by_sector().items()}
        scen_abs = {k: float(v) * scen_total for k, v in scenario.aggregate_by_sector().items()}

        all_keys = set(base_abs) | set(scen_abs)
        delta_sector = {k: scen_abs.get(k, 0.0) - base_abs.get(k, 0.0) for k in all_keys}

        return ConsequentialResult(
            satellite=satellite,
            baseline=baseline,
            scenario=scenario,
            delta_total_intensity=scen_total - base_total,
            delta_sector_contributions=delta_sector,
            sector_names=self._sectors,
        )

    @classmethod
    def analyze_time_series(
        cls,
        series: Mapping[
            Any,
            Union[
                SPA,
                tuple[
                    ArrayLike,
                    Union[
                        ArrayLike,
                        dict[str, ArrayLike],
                        SatelliteWithConcordance,
                        dict[str, SatelliteWithConcordance],
                    ],
                ],
            ],
        ],
        sector: Union[int, str],
        depth: int = 8,
        *,
        sectors: Optional[Sequence[str]] = None,
        mode: Literal["sector_specific", "system_wide"] = "sector_specific",
        satellite: Optional[str] = None,
        **kwargs: Any,
    ) -> TemporalSPAResult:
        """Run SPA across a time series of model snapshots.

        This provides a lightweight multi-year workflow by repeatedly running SPA for a
        set of (A, intensities) snapshots.

        Args:
            series: Mapping from time label (e.g. year) to either:
                - an SPA instance, or
                - a tuple of (A, intensities)
            sector: Target sector.
            depth: SPA depth.
            sectors: Optional sector names to use when constructing SPA instances.
            mode: Analysis mode used for constructed SPA instances.
            satellite: Satellite to analyze.
            **kwargs: Passed through to SPA.analyze.

        Returns:
            TemporalSPAResult
        """
        results: dict[Any, Union[PathCollection, SPAResult]] = {}
        for t, model in series.items():
            if isinstance(model, SPA):
                spa = model
            else:
                A, intensities = model
                spa = cls(A, intensities, sectors=sectors, mode=mode)

            results[t] = spa.analyze(
                sector,
                depth,
                satellite=satellite,
                **kwargs,
            )

        return TemporalSPAResult(
            results=results,
            sector=sector,
            depth=depth,
            satellite=satellite,
        )

    def _resolve_sector(self, sector: Union[int, str]) -> int:
        """Convert sector name to index if needed. Handles 1-indexed sector IDs."""
        if isinstance(sector, str):
            if self._sectors is None:
                raise ValueError("Sector names not provided, use integer index")
            try:
                return self._sectors.index(sector)
            except ValueError as err:
                raise ValueError(f"Unknown sector: {sector}") from err
        # Convert 1-indexed to 0-indexed
        if isinstance(sector, int):
            if sector < 1 or sector > self._n:
                raise ValueError(f"Sector index {sector} out of range [1, {self._n}]")
            return sector - 1
        raise TypeError(f"sector must be int or str, got {type(sector).__name__}")

    @overload
    def analyze(
        self,
        sector: Union[int, str],
        depth: int = 8,
        *,
        threshold: float = 0.001,
        threshold_type: Literal["percentage", "absolute"] = "percentage",
        satellite: str = ...,
        max_paths: Optional[int] = None,
        include_stage_0: bool = True,
    ) -> PathCollection: ...

    @overload
    def analyze(
        self,
        sector: Union[int, str],
        depth: int = 8,
        *,
        threshold: float = 0.001,
        threshold_type: Literal["percentage", "absolute"] = "percentage",
        satellite: None = None,
        max_paths: Optional[int] = None,
        include_stage_0: bool = True,
    ) -> Union[PathCollection, SPAResult]: ...

    def analyze(
        self,
        sector: Union[int, str],
        depth: int = 8,
        *,
        threshold: float = 0.001,
        threshold_type: Literal["percentage", "absolute"] = "percentage",
        satellite: Optional[str] = None,
        max_paths: Optional[int] = None,
        include_remainder: bool = True,
        include_stage_0: bool = True,
    ) -> Union[PathCollection, SPAResult]:
        """
        Run structural path analysis for a target sector.

        Args:
            sector: Target sector (index or name)
            depth: Maximum supply chain depth to explore
            threshold: Minimum contribution threshold
            threshold_type: Type of threshold - "percentage" (default) or "absolute"
                - "percentage": threshold as fraction of total intensity (0.001 = 0.1%)
                - "absolute": threshold as absolute value (e.g., 1e-6)
            satellite: Specific satellite to analyze (None = all)
            max_paths: Maximum number of paths to return
            include_remainder: Include a remainder path for uncovered portion
            include_stage_0: Include Stage 0 pathway representing direct requirements (default True)

        Returns:
            PathCollection if single satellite, SPAResult if multiple

        Example:
            >>> # Percentage threshold (default, conservative)
            >>> paths = spa.analyze(sector=42, depth=8, threshold=0.001)
            >>>
            >>> # Absolute threshold (permissive)
            >>> paths = spa.analyze(sector=42, depth=8, threshold=1e-6, threshold_type="absolute")
            >>>
            >>> # Without Stage 0 direct requirements
            >>> paths = spa.analyze(sector=42, depth=8, include_stage_0=False)
        """
        sector_idx = self._resolve_sector(sector)

        # Validate threshold_type
        if threshold_type not in ("percentage", "absolute"):
            raise ValueError(
                f"threshold_type must be 'percentage' or 'absolute', got {threshold_type}"
            )

        # Determine which satellites to analyze
        if satellite is not None:
            if satellite not in self._intensities:
                raise ValueError(f"Unknown satellite: {satellite}")
            satellites_to_analyze = [satellite]
        else:
            satellites_to_analyze = list(self._intensities.keys())

        # Ensure Leontief is computed
        if self._L is None:
            self._compute_leontief()
        assert self._total_intensities is not None

        results: dict[str, PathCollection] = {}
        for sat_name in satellites_to_analyze:
            direct = self._intensities[sat_name]
            total = self._total_intensities[sat_name]
            total_intensity = total[sector_idx]

            if total_intensity == 0:
                results[sat_name] = PathCollection([], sector_idx, 0.0, sat_name)
                continue

            # Run the path extraction
            paths = self._extract_paths(
                sector_idx,
                depth,
                threshold,
                threshold_type,
                direct,
                total_intensity,
                max_paths,
            )

            # Add Stage 0 pathway (direct requirements of target sector)
            if include_stage_0:
                stage_0_path = Path(
                    nodes=(sector_idx,),
                    contribution=direct[sector_idx] / total_intensity,
                    direct_intensity=direct[sector_idx],
                    cumulative_weight=1.0,
                    _sector_names=self._sectors,
                )
                paths.insert(0, stage_0_path)

            # Create metadata
            sector_names = self._sectors
            sector_name = sector_names[sector_idx] if sector_names is not None else None
            metadata = AnalysisMetadata(
                analysis_date=datetime.now(),
                sector=sector,
                sector_name=sector_name,
                threshold=threshold,
                threshold_type=threshold_type,
                max_depth=depth,
                total_intensity=total_intensity,
                coverage=sum(p.contribution for p in paths),
                n_paths=len(paths),
                mode=self._mode,
                satellite=sat_name,
                n_sectors=self._n,
            )

            results[sat_name] = PathCollection(
                paths, sector_idx, total_intensity, sat_name, metadata=metadata
            )

        # Return appropriate type
        if len(results) == 1:
            return list(results.values())[0]
        return SPAResult(results, sector_idx)

    def _extract_paths(
        self,
        target: int,
        max_depth: int,
        threshold: float,
        threshold_type: str,
        direct_intensities: NDArray,
        total_intensity: float,
        max_paths: Optional[int],
    ) -> list[Path]:
        """
        Core SPA algorithm using depth-first search with pruning.

        Supports two modes:
        - sector_specific: Uses A matrix (direct supplier relationships)
        - system_wide: Uses Leontief inverse (total requirements including circular flows)

        Args:
            threshold_type: "percentage" or "absolute"
        """
        if self._mode == "sector_specific":
            return self._extract_paths_a_matrix(
                target,
                max_depth,
                threshold,
                threshold_type,
                direct_intensities,
                total_intensity,
                max_paths,
            )
        else:  # system_wide
            return self._extract_paths_leontief(
                target,
                max_depth,
                threshold,
                threshold_type,
                direct_intensities,
                total_intensity,
                max_paths,
            )

    def _extract_paths_a_matrix(
        self,
        target: int,
        max_depth: int,
        threshold: float,
        threshold_type: str,
        direct_intensities: NDArray,
        total_intensity: float,
        max_paths: Optional[int],
    ) -> list[Path]:
        """
        SPA using A matrix (sector-specific analysis).

        Traces direct supplier relationships through the technical coefficients matrix.
        Optimized with precomputed sparse structure and deferred Path creation.

        Args:
            threshold_type: "percentage" or "absolute"
        """
        # Precompute sparse structure if not done
        self._precompute_sparse_structure()
        assert self._total_intensities is not None
        assert self._A_col_nonzero is not None
        assert self._A_col_values is not None

        # Calculate absolute threshold based on type
        if threshold_type == "percentage":
            abs_threshold = threshold * total_intensity
        else:  # absolute
            abs_threshold = threshold

        # Pre-compute and cache values used in inner loop
        first_satellite = next(iter(self._intensities))
        total_intensities_arr = self._total_intensities[first_satellite]
        inv_total = 1.0 / total_intensity if total_intensity > 0 else 0.0
        sector_names = self._sectors

        # Use precomputed sparse structure
        col_nonzero = self._A_col_nonzero
        col_values = self._A_col_values

        # Store raw path data: (nodes_list, contribution_frac, direct_int, cum_weight)
        raw_paths: list[tuple[list[int], float, float, float]] = []

        # Stack: (node_path as list, cumulative_weight)
        stack: list[tuple[list[int], float]] = [([target], 1.0)]

        while stack:
            if max_paths is not None and len(raw_paths) >= max_paths:
                break

            node_path, cum_weight = stack.pop()
            current = node_path[-1]
            current_depth = len(node_path) - 1

            # Calculate contribution of this path
            direct_int = direct_intensities[current]
            contribution_value = cum_weight * direct_int

            # Add this path if it has meaningful contribution
            if contribution_value > 0:
                raw_paths.append(
                    (
                        node_path,
                        contribution_value * inv_total,
                        direct_int,
                        cum_weight,
                    )
                )

            # Explore upstream suppliers if we haven't hit max depth
            if current_depth < max_depth:
                # Use precomputed sparse structure
                suppliers = col_nonzero[current]
                coefs = col_values[current]

                if len(suppliers) > 0:
                    # Vectorized: compute new weights for all suppliers at once
                    new_weights = cum_weight * coefs

                    # Vectorized: compute potentials and filter by threshold
                    potentials = new_weights * total_intensities_arr[suppliers]
                    valid_mask = potentials >= abs_threshold

                    # Only loop over valid suppliers
                    valid_suppliers = suppliers[valid_mask]
                    valid_weights = new_weights[valid_mask]

                    for i in range(len(valid_suppliers)):
                        new_path = node_path.copy()
                        new_path.append(int(valid_suppliers[i]))  # Convert to Python int
                        stack.append((new_path, valid_weights[i]))

        # Sort raw paths by contribution (descending)
        raw_paths.sort(key=lambda x: x[1], reverse=True)

        # Create Path objects after sorting
        paths: list[Path] = []
        for node_list, contrib, direct_int, cum_weight in raw_paths:
            paths.append(
                Path(
                    nodes=tuple(node_list),
                    contribution=contrib,
                    direct_intensity=direct_int,
                    cumulative_weight=cum_weight,
                    _sector_names=sector_names,
                )
            )

        return paths

    def _extract_paths_leontief(
        self,
        target: int,
        max_depth: int,
        threshold: float,
        threshold_type: str,
        direct_intensities: NDArray,
        total_intensity: float,
        max_paths: Optional[int],
    ) -> list[Path]:
        """
        SPA using Leontief inverse (system-wide analysis).

        Traces total requirements (direct + indirect) through the Leontief inverse matrix.
        Captures circular flows and economy-wide supply chain decomposition.
        Optimized with vectorized NumPy operations and deferred Path creation.

        Args:
            threshold_type: "percentage" or "absolute"
        """
        L = self.L

        # Calculate absolute threshold based on type
        if threshold_type == "percentage":
            abs_threshold = threshold * total_intensity
        else:  # absolute
            abs_threshold = threshold

        # Pre-compute and cache values used in inner loop
        max_direct_intensity = direct_intensities.max()
        inv_total = 1.0 / total_intensity if total_intensity > 0 else 0.0
        sector_names = self._sectors

        # Store raw path data: (nodes_list, contribution_frac, direct_int, cum_weight)
        raw_paths: list[tuple[list[int], float, float, float]] = []

        # Stack: (node_path as list, cumulative_weight)
        stack: list[tuple[list[int], float]] = [([target], 1.0)]

        while stack:
            if max_paths is not None and len(raw_paths) >= max_paths:
                break

            node_path, cum_weight = stack.pop()
            current = node_path[-1]
            current_depth = len(node_path) - 1

            # Calculate contribution of this path
            direct_int = direct_intensities[current]
            contribution_value = cum_weight * direct_int

            # Add this path if it has meaningful contribution
            if contribution_value > 0:
                raw_paths.append(
                    (
                        node_path,
                        contribution_value * inv_total,
                        direct_int,
                        cum_weight,
                    )
                )

            # Explore upstream suppliers if we haven't hit max depth
            if current_depth < max_depth:
                # Get total requirements from current sector (column of Leontief inverse)
                total_requirements = L[:, current]

                # Vectorized: find suppliers with non-zero coefficients
                nonzero_mask = total_requirements > 0
                suppliers = np.nonzero(nonzero_mask)[0]

                if len(suppliers) > 0:
                    # Vectorized: compute new weights for all suppliers at once
                    coefs = total_requirements[suppliers]
                    new_weights = cum_weight * coefs

                    # Vectorized: compute potentials and filter by threshold
                    potentials = new_weights * max_direct_intensity
                    valid_mask = potentials >= abs_threshold

                    # Only loop over valid suppliers
                    valid_suppliers = suppliers[valid_mask]
                    valid_weights = new_weights[valid_mask]

                    for i in range(len(valid_suppliers)):
                        new_path = node_path.copy()
                        new_path.append(int(valid_suppliers[i]))  # Convert to Python int
                        stack.append((new_path, valid_weights[i]))

        # Sort raw paths by contribution (descending)
        raw_paths.sort(key=lambda x: x[1], reverse=True)

        # Create Path objects after sorting
        paths: list[Path] = []
        for node_list, contrib, direct_int, cum_weight in raw_paths:
            paths.append(
                Path(
                    nodes=tuple(node_list),
                    contribution=contrib,
                    direct_intensity=direct_int,
                    cumulative_weight=cum_weight,
                    _sector_names=sector_names,
                )
            )

        return paths

    def stream(
        self,
        sector: Union[int, str],
        depth: int = 8,
        *,
        threshold: float = 0.001,
        threshold_type: Literal["percentage", "absolute"] = "percentage",
        satellite: Optional[str] = None,
    ) -> Iterator[Path]:
        """
        Stream paths as they are discovered (memory-efficient for large systems).

        Yields paths in discovery order (not sorted by contribution).
        Supports both sector-specific (A-matrix) and system-wide (Leontief) modes.

        Args:
            sector: Target sector
            depth: Maximum depth
            threshold: Minimum contribution threshold
            threshold_type: "percentage" (default) or "absolute"
            satellite: Satellite to analyze

        Yields:
            Path objects as they are discovered

        Example:
            >>> for path in spa.stream(sector=42, depth=8, threshold=0.001):
            ...     print(path)
        """
        sector_idx = self._resolve_sector(sector)

        # Validate threshold_type
        if threshold_type not in ("percentage", "absolute"):
            raise ValueError(
                f"threshold_type must be 'percentage' or 'absolute', got {threshold_type}"
            )

        if satellite is None:
            satellite = next(iter(self._intensities))

        if self._L is None:
            self._compute_leontief()
        assert self._total_intensities is not None

        direct = self._intensities[satellite]
        total_intensity = self._total_intensities[satellite][sector_idx]

        # Calculate absolute threshold based on type
        if threshold_type == "percentage":
            abs_threshold = threshold * total_intensity
        else:  # absolute
            abs_threshold = threshold

        if self._mode == "sector_specific":
            yield from self._stream_a_matrix(
                sector_idx, depth, abs_threshold, direct, total_intensity
            )
        else:  # system_wide
            yield from self._stream_leontief(
                sector_idx, depth, abs_threshold, direct, total_intensity
            )

    def _stream_a_matrix(
        self,
        sector_idx: int,
        depth: int,
        abs_threshold: float,
        direct: NDArray,
        total_intensity: float,
    ) -> Iterator[Path]:
        """Stream paths using A matrix (sector-specific). Optimized with precomputed sparse structure."""
        self._precompute_sparse_structure()
        assert self._total_intensities is not None
        assert self._A_col_nonzero is not None
        assert self._A_col_values is not None

        first_satellite = next(iter(self._intensities))
        total_intensities_arr = self._total_intensities[first_satellite]
        inv_total = 1.0 / total_intensity if total_intensity > 0 else 0.0
        sector_names = self._sectors

        col_nonzero = self._A_col_nonzero
        col_values = self._A_col_values

        stack: list[tuple[list[int], float]] = [([sector_idx], 1.0)]

        while stack:
            node_path, cum_weight = stack.pop()
            current = node_path[-1]
            current_depth = len(node_path) - 1

            direct_int = direct[current]
            contribution_value = cum_weight * direct_int

            if contribution_value > 0:
                yield Path(
                    nodes=tuple(node_path),
                    contribution=contribution_value * inv_total,
                    direct_intensity=direct_int,
                    cumulative_weight=cum_weight,
                    _sector_names=sector_names,
                )

            if current_depth < depth:
                suppliers = col_nonzero[current]
                coefs = col_values[current]

                if len(suppliers) > 0:
                    new_weights = cum_weight * coefs
                    potentials = new_weights * total_intensities_arr[suppliers]
                    valid_mask = potentials >= abs_threshold

                    valid_suppliers = suppliers[valid_mask]
                    valid_weights = new_weights[valid_mask]

                    for i in range(len(valid_suppliers)):
                        new_path = node_path.copy()
                        new_path.append(int(valid_suppliers[i]))  # Convert to Python int
                        stack.append((new_path, valid_weights[i]))

    def _stream_leontief(
        self,
        sector_idx: int,
        depth: int,
        abs_threshold: float,
        direct: NDArray,
        total_intensity: float,
    ) -> Iterator[Path]:
        """Stream paths using Leontief inverse (system-wide). Optimized with vectorization."""
        L = self.L
        max_direct_intensity = direct.max()
        inv_total = 1.0 / total_intensity if total_intensity > 0 else 0.0
        sector_names = self._sectors

        stack: list[tuple[list[int], float]] = [([sector_idx], 1.0)]

        while stack:
            node_path, cum_weight = stack.pop()
            current = node_path[-1]
            current_depth = len(node_path) - 1

            direct_int = direct[current]
            contribution_value = cum_weight * direct_int

            if contribution_value > 0:
                yield Path(
                    nodes=tuple(node_path),
                    contribution=contribution_value * inv_total,
                    direct_intensity=direct_int,
                    cumulative_weight=cum_weight,
                    _sector_names=sector_names,
                )

            if current_depth < depth:
                total_requirements = L[:, current]
                nonzero_mask = total_requirements > 0
                suppliers = np.nonzero(nonzero_mask)[0]

                if len(suppliers) > 0:
                    coefs = total_requirements[suppliers]
                    new_weights = cum_weight * coefs
                    potentials = new_weights * max_direct_intensity
                    valid_mask = potentials >= abs_threshold

                    valid_suppliers = suppliers[valid_mask]
                    valid_weights = new_weights[valid_mask]

                    for i in range(len(valid_suppliers)):
                        new_path = node_path.copy()
                        new_path.append(int(valid_suppliers[i]))  # Convert to Python int
                        stack.append((new_path, valid_weights[i]))

    def analyze_many(
        self,
        sectors: Sequence[Union[int, str]],
        depth: int = 8,
        **kwargs,
    ) -> dict[Union[int, str], Union[PathCollection, SPAResult]]:
        """
        Analyze multiple sectors.

        Args:
            sectors: List of sector indices or names
            depth: Maximum depth
            **kwargs: Additional arguments passed to analyze()

        Returns:
            Dictionary mapping sectors to their results
        """
        return {sector: self.analyze(sector, depth, **kwargs) for sector in sectors}

    def hotspots(
        self,
        sector: Union[int, str],
        depth: int = 8,
        *,
        satellite: Optional[str] = None,
        top_n: int = 10,
    ) -> list[tuple[int, float, str]]:
        """
        Identify emission hotspots (sectors contributing most to total).

        Args:
            sector: Target sector
            depth: Analysis depth
            satellite: Satellite to analyze
            top_n: Number of top hotspots to return

        Returns:
            List of (sector_index, contribution, sector_name) tuples
        """
        result = self.analyze(sector, depth, satellite=satellite)

        if isinstance(result, SPAResult):
            result = list(result.results.values())[0]

        aggregated = result.aggregate_by_sector()

        # Sort by contribution
        sorted_hotspots = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

        return [
            (idx, contrib, self._sectors[idx] if self._sectors else f"Sector_{idx}")
            for idx, contrib in sorted_hotspots[:top_n]
        ]

    def compare_sectors(
        self,
        sectors: Sequence[Union[int, str]],
        depth: int = 8,
        *,
        satellite: Optional[str] = None,
    ):
        """
        Compare SPA results across multiple sectors.

        Returns a DataFrame showing key metrics for each sector.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for compare_sectors")

        records = []
        for sector in sectors:
            result = self.analyze(sector, depth, satellite=satellite)
            if isinstance(result, SPAResult):
                result = list(result.results.values())[0]

            sector_idx = self._resolve_sector(sector)
            records.append(
                {
                    "sector": sector,
                    "sector_name": (
                        self._sectors[sector_idx] if self._sectors else f"Sector_{sector_idx}"
                    ),
                    "total_intensity": result.total_intensity,
                    "num_paths": len(result.paths),
                    "coverage": result.coverage,
                    "top_path_contribution": result.paths[0].contribution if result.paths else 0,
                }
            )

        return pd.DataFrame(records)

    def __repr__(self) -> str:
        satellites = ", ".join(self.satellites)
        return f"SPA(n_sectors={self._n}, satellites=[{satellites}], mode={self._mode})"


# =============================================================================
# Factory Functions
# =============================================================================


def from_leontief(
    L: ArrayLike,
    direct_intensities: Union[
        ArrayLike,
        dict[str, ArrayLike],
        SatelliteWithConcordance,
        dict[str, SatelliteWithConcordance],
    ],
    sectors: Optional[Sequence[str]] = None,
    mode: Literal["sector_specific", "system_wide"] = "system_wide",
) -> SPA:
    """
    Create SPA from a pre-computed Leontief inverse.

    This is more efficient if you already have L computed.
    Defaults to system_wide mode since Leontief inverse is provided.

    Args:
        L: Leontief inverse matrix (I - A)^(-1)
        direct_intensities: Direct intensity vector(s)
        sectors: Optional sector names
        mode: Analysis mode - defaults to "system_wide" for Leontief-based analysis

    Returns:
        SPA instance with pre-computed Leontief
    """
    L = np.asarray(L)
    identity = np.eye(L.shape[0])
    A = identity - np.linalg.inv(L)

    spa = SPA(A, direct_intensities, sectors, mode=mode)
    spa._L = L

    # Pre-compute total intensities
    # Note: If SatelliteWithConcordance objects were provided, intensities are already
    # in target classification, so we can use them directly
    if spa._satellite_objects is not None:
        # SatelliteWithConcordance objects already provide target intensities
        spa._total_intensities = {
            name: intensities @ L for name, intensities in spa._intensities.items()
        }
    elif isinstance(direct_intensities, dict):
        spa._total_intensities = {k: np.asarray(v) @ L for k, v in direct_intensities.items()}
    else:
        spa._total_intensities = {"intensity": np.asarray(direct_intensities) @ L}

    return spa


def from_dataframe(
    A_df,
    intensities_df,
    intensity_columns: Optional[list[str]] = None,
) -> SPA:
    """
    Create SPA from pandas DataFrames.

    Args:
        A_df: DataFrame with A matrix (index and columns are sector names)
        intensities_df: DataFrame with intensity columns, or SatelliteWithConcordance objects
        intensity_columns: Which columns to use as satellites (ignored if SatelliteWithConcordance provided)

    Returns:
        SPA instance
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for from_dataframe")

    sectors = list(A_df.index)
    A = A_df.values

    intensities: (
        ArrayLike
        | dict[str, ArrayLike]
        | SatelliteWithConcordance
        | dict[str, SatelliteWithConcordance]
    )

    # Handle SatelliteWithConcordance objects
    if isinstance(intensities_df, SatelliteWithConcordance):
        intensities = intensities_df
    elif isinstance(intensities_df, dict) and all(
        isinstance(v, SatelliteWithConcordance) for v in intensities_df.values()
    ):
        intensities = intensities_df
    else:
        # Regular DataFrame processing
        if intensity_columns is None:
            # Use all numeric columns
            intensity_columns = intensities_df.select_dtypes(include=[np.number]).columns.tolist()

        intensities = {col: intensities_df[col].values for col in intensity_columns}

    return SPA(A, intensities, sectors)


def from_csv(
    A_path: str,
    intensities_path: str,
    intensity_columns: Optional[list[str]] = None,
    sectors_path: Optional[str] = None,
    mode: Literal["sector_specific", "system_wide"] = "sector_specific",
) -> SPA:
    """
    Create SPA from CSV files.

    Loads technical coefficients matrix and intensity vectors from CSV files.
    Supports optional sector names file.

    Args:
        A_path: Path to A-matrix CSV (sectors × sectors)
        intensities_path: Path to intensities CSV (sectors × satellites)
        intensity_columns: Which columns to use as satellites (None = all numeric)
        sectors_path: Optional path to sector names (one per line)
        mode: Analysis mode ("sector_specific" or "system_wide")

    Returns:
        SPA instance

    Raises:
        ImportError: If pandas is not installed

    Example:
        >>> spa = from_csv(
        ...     "A_matrix.csv",
        ...     "intensities.csv",
        ...     intensity_columns=["ghg", "water"]
        ... )
        >>> paths = spa.analyze(sector=0, depth=8)
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for from_csv")

    # Load A matrix
    A_df = pd.read_csv(A_path, index_col=0)
    sectors = list(A_df.index)
    A = A_df.values

    # Load intensities
    intensities_df = pd.read_csv(intensities_path, index_col=0)

    if intensity_columns is None:
        # Use all numeric columns
        intensity_columns = intensities_df.select_dtypes(include=[np.number]).columns.tolist()

    intensities = {col: intensities_df[col].values for col in intensity_columns}

    # Load sector names if provided
    if sectors_path is not None:
        with open(sectors_path) as f:
            sectors = [line.strip() for line in f if line.strip()]

    return SPA(A, intensities, sectors, mode=mode)


def from_io_table(
    xlsx_path: str,
    sheet_name: str = "Table 5",
    intensities: Optional[Union[ArrayLike, dict[str, ArrayLike]]] = None,
) -> SPA:
    """
    Create SPA from an IO table in XLSX format.

    Automatically extracts the largest continuous numeric matrix as the transaction matrix,
    identifies sector names from adjacent columns, and computes technical coefficients.

    Args:
        xlsx_path: Path to the XLSX file
        sheet_name: Sheet name containing the IO table
        intensities: Direct intensities. If None, uses dummy values.

    Returns:
        SPA instance

    Raises:
        ImportError: If pandas is not installed
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for from_io_table")

    # Load the sheet
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # Find the bounding box of all numeric cells (approximates largest continuous matrix)
    rows, cols = df.shape
    numeric_positions = []
    for r in range(rows):
        for c in range(cols):
            try:
                float(df.iloc[r, c])
                numeric_positions.append((r, c))
            except (ValueError, TypeError):
                pass

    if not numeric_positions:
        raise ValueError("No numeric cells found in the sheet")

    min_row = min(r for r, c in numeric_positions)
    max_row = max(r for r, c in numeric_positions)
    min_col = min(c for r, c in numeric_positions)
    max_col = max(c for r, c in numeric_positions)

    start_row, end_row, start_col, end_col = min_row, max_row + 1, min_col, max_col + 1
    transactions = df.iloc[start_row:end_row, start_col:end_col].values.astype(float)

    # Find column headers: first non-numeric row above start_row
    for r in range(start_row - 1, -1, -1):
        row_values = [df.iloc[r, c] for c in range(start_col, end_col)]
        if not all(_is_numeric(v) for v in row_values if pd.notna(v)):
            break

    # Find row headers: first non-numeric column left of start_col
    row_headers = None
    for c in range(start_col - 1, -1, -1):
        col_values = [df.iloc[r, c] for r in range(start_row, end_row)]
        if not all(_is_numeric(v) for v in col_values if pd.notna(v)):
            row_headers = col_values
            break

    # Assume row_headers are sector names
    sectors = row_headers if row_headers else [f"Sector_{i}" for i in range(transactions.shape[0])]

    # For IO table, assume the last column of the sheet is total supply
    total_supply_col = cols - 1
    total_supply = pd.to_numeric(df.iloc[start_row:end_row, total_supply_col], errors="coerce")

    # Compute A matrix
    A = transactions / total_supply.values[:, np.newaxis]

    # Use dummy intensities if not provided
    if intensities is None:
        intensities = np.random.uniform(0.1, 2.0, len(sectors))

    return SPA(A, intensities, sectors)


def _is_numeric(val) -> bool:
    """Helper function to check if a value is numeric."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


# =============================================================================
# Uncertainty results + sampling helpers
# =============================================================================


@dataclass(frozen=True)
class UncertaintyStats:
    """Summary statistics for a Monte Carlo sample."""

    mean: float
    std: float
    var: float
    ci_low: float
    ci_high: float
    ci: tuple[float, float]
    n: int

    @classmethod
    def from_samples(
        cls, samples: ArrayLike, *, ci: tuple[float, float] = (0.025, 0.975)
    ) -> UncertaintyStats:
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            raise ValueError("samples must not be empty")

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        var = float(std * std)
        q = np.asarray(np.quantile(arr, [ci[0], ci[1]]), dtype=float)
        ci_low = float(q[0])
        ci_high = float(q[1])
        return cls(
            mean=mean,
            std=std,
            var=var,
            ci_low=ci_low,
            ci_high=ci_high,
            ci=ci,
            n=int(arr.size),
        )


@dataclass(frozen=True)
class MonteCarloResult:
    """Outputs from SPA.monte_carlo()."""

    satellite: str
    target_sector: int
    as_absolute: bool
    ci: tuple[float, float]
    samples_total_intensity: NDArray
    samples_sector_contributions: NDArray
    total_intensity: UncertaintyStats
    sector_contributions: dict[int, UncertaintyStats]
    sector_names: Optional[tuple[str, ...]] = None

    def sector_contributions_dataframe(self):
        """Return a pandas DataFrame with per-sector uncertainty stats."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for sector_contributions_dataframe()")

        records = []
        for i, stats in self.sector_contributions.items():
            records.append(
                {
                    "sector": i,
                    "sector_name": self.sector_names[i] if self.sector_names is not None else None,
                    "mean": stats.mean,
                    "std": stats.std,
                    "var": stats.var,
                    "ci_low": stats.ci_low,
                    "ci_high": stats.ci_high,
                }
            )

        return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class ConsequentialResult:
    """Outputs from SPA.consequential()."""

    satellite: str
    baseline: PathCollection
    scenario: PathCollection
    delta_total_intensity: float
    delta_sector_contributions: dict[int, float]
    sector_names: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class TemporalSPAResult:
    """Outputs from SPA.analyze_time_series()."""

    results: dict[Any, Union[PathCollection, SPAResult]]
    sector: Union[int, str]
    depth: int
    satellite: Optional[str] = None

    def total_intensity_series(self) -> dict[Any, float]:
        series: dict[Any, float] = {}
        for t, res in self.results.items():
            if isinstance(res, SPAResult):
                first = next(iter(res.results.values()))
                series[t] = float(first.total_intensity)
            else:
                series[t] = float(res.total_intensity)
        return series

    def to_dataframe(self):
        if not HAS_PANDAS:
            raise ImportError("pandas required for to_dataframe")

        records = []
        for t, res in self.results.items():
            if isinstance(res, SPAResult):
                for sat, pc in res.results.items():
                    records.append(
                        {
                            "time": t,
                            "satellite": sat,
                            "total_intensity": pc.total_intensity,
                            "coverage": pc.coverage,
                            "n_paths": len(pc.paths),
                        }
                    )
            else:
                records.append(
                    {
                        "time": t,
                        "satellite": res.satellite_name,
                        "total_intensity": res.total_intensity,
                        "coverage": res.coverage,
                        "n_paths": len(res.paths),
                    }
                )

        return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class LoopAnalysisResult:
    """Summary of detected loops in a PathCollection."""

    as_absolute: bool
    loop_share: float
    cycles: dict[tuple[int, ...], float]
    sector_participation: dict[int, float]
    paths: PathCollection

    def top_cycles(self, n: int = 10) -> list[tuple[tuple[int, ...], float]]:
        return sorted(self.cycles.items(), key=lambda x: x[1], reverse=True)[:n]


def _latin_hypercube(n_samples: int, n_dims: int, rng: np.random.Generator) -> NDArray:
    """Latin Hypercube samples in [0, 1] with independent strata per dimension."""
    if n_samples <= 0 or n_dims <= 0:
        raise ValueError("n_samples and n_dims must be > 0")

    u = np.empty((n_samples, n_dims), dtype=float)
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        u[:, j] = (perm + rng.random(n_samples)) / n_samples
    return u


def _norm_ppf(p: ArrayLike) -> NDArray:
    """Inverse standard normal CDF (Acklam approximation), vectorized."""
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p must be in (0, 1)")

    # Coefficients from Peter J. Acklam's approximation
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    plow = 0.02425
    phigh = 1 - plow

    x = np.empty_like(p)

    # Lower region
    mask = p < plow
    if np.any(mask):
        q = np.sqrt(-2 * np.log(p[mask]))
        x[mask] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )

    # Central region
    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q * q
        x[mask] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )

    # Upper region
    mask = p > phigh
    if np.any(mask):
        q = np.sqrt(-2 * np.log(1 - p[mask]))
        x[mask] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )

    return x


def _sample_lognormal_mean_cv(
    mean: ArrayLike,
    cv: Union[float, ArrayLike],
    rng: np.random.Generator,
    *,
    z: Optional[ArrayLike] = None,
) -> NDArray:
    """Sample a lognormal with specified mean and coefficient of variation."""
    mean_arr = np.asarray(mean, dtype=float)
    cv_arr = np.asarray(cv, dtype=float)

    if z is None:
        z_arr = rng.standard_normal(size=mean_arr.shape)
    else:
        z_arr = np.asarray(z, dtype=float)
        if z_arr.shape != mean_arr.shape:
            raise ValueError(f"z shape {z_arr.shape} does not match mean shape {mean_arr.shape}")

    # sigma^2 = ln(1 + cv^2)
    sigma2 = np.log1p(np.square(cv_arr))
    sigma = np.sqrt(sigma2)

    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.log(mean_arr) - sigma2 / 2
        samples = np.exp(mu + sigma * z_arr)

    # Keep exact zeros as zeros
    samples = np.where(mean_arr <= 0, 0.0, samples)
    return samples


def _sample_trunc_normal_mean_cv(
    mean: ArrayLike,
    cv: Union[float, ArrayLike],
    rng: np.random.Generator,
) -> NDArray:
    """Sample a normal with mean/std=mean*cv, truncated at 0."""
    mean_arr = np.asarray(mean, dtype=float)
    cv_arr = np.asarray(cv, dtype=float)
    std = np.abs(mean_arr) * cv_arr
    samples = mean_arr + std * rng.standard_normal(size=mean_arr.shape)
    return np.clip(samples, 0.0, None)


# =============================================================================
# Network metrics helpers
# =============================================================================


def _brandes_betweenness_centrality(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    *,
    normalized: bool = True,
) -> NDArray:
    """Brandes betweenness centrality for a directed graph with positive edge lengths."""
    import heapq

    n = len(adjacency)
    bc = np.zeros(n, dtype=float)

    if n == 0:
        return bc

    eps = 1e-12

    for s in range(n):
        # Single-source shortest paths
        S: list[int] = []
        P: list[list[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0
        dist = np.full(n, np.inf, dtype=float)
        dist[s] = 0.0

        Q: list[tuple[float, int]] = [(0.0, s)]

        while Q:
            d_v, v = heapq.heappop(Q)
            if d_v > dist[v] + eps:
                continue
            S.append(v)

            for w, length in adjacency[v]:
                d_w = d_v + length
                if d_w < dist[w] - eps:
                    dist[w] = d_w
                    heapq.heappush(Q, (d_w, w))
                    sigma[w] = sigma[v]
                    P[w] = [v]
                elif abs(d_w - dist[w]) <= eps:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # Accumulation
        delta = np.zeros(n, dtype=float)
        for w in reversed(S):
            for v in P[w]:
                if sigma[w] != 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    if normalized and n > 2:
        bc *= 1.0 / ((n - 1) * (n - 2))

    return bc
