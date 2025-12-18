"""
Sector Concordance for EEIO Models

Provides evidence-based sector mapping between classification systems
(e.g., ANZSIC to IOIG) using concordance matrices derived from supply-use tables.

Based on OECD/UN Handbook on Extended Supply and Use Tables (2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class ConcordanceMetadata:
    """Metadata documenting the provenance and quality of a concordance mapping."""
    
    source: str
    """Data source (e.g., 'ABS Supply-Use Table 2020')"""
    
    method: str
    """Derivation method (e.g., 'supply_use_table', 'entropy_minimization', 'hierarchical')"""
    
    year: Optional[int] = None
    """Reference year of the source data"""
    
    confidence: Optional[float] = None
    """Confidence score (0-1) indicating % of sectors with high-confidence mapping"""
    
    notes: str = ""
    """Additional documentation or caveats"""
    
    def __str__(self) -> str:
        lines = [
            f"Source: {self.source}",
            f"Method: {self.method}",
        ]
        if self.year:
            lines.append(f"Year: {self.year}")
        if self.confidence is not None:
            lines.append(f"Confidence: {self.confidence:.1%}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


class SectorConcordance:
    """
    Manages sector concordance matrices for EEIO models.
    
    A concordance matrix C[i,j] represents the proportion of source sector i
    that corresponds to target sector j. Derived from supply-use tables,
    employment data, or other economic sources.
    
    Attributes:
        matrix: Concordance matrix (source_sectors × target_sectors)
        source_sectors: Names of source classification sectors
        target_sectors: Names of target classification sectors
        metadata: Provenance and quality information
        
    Example:
        >>> # From supply-use table data
        >>> concordance = SectorConcordance.from_supply_use_table(
        ...     supply_use_df,
        ...     source_col="ANZSIC",
        ...     target_col="IOIG",
        ...     metadata=ConcordanceMetadata(
        ...         source="ABS Supply-Use Table 2020",
        ...         method="supply_use_table",
        ...         year=2020,
        ...         confidence=0.87
        ...     )
        ... )
        >>> 
        >>> # Allocate emissions from source to target sectors
        >>> emissions_target = concordance.allocate(emissions_source)
    """
    
    def __init__(
        self,
        matrix: ArrayLike,
        source_sectors: List[str],
        target_sectors: List[str],
        metadata: Optional[ConcordanceMetadata] = None,
    ):
        """
        Initialize a concordance matrix.
        
        Args:
            matrix: Concordance matrix (n_source × n_target)
            source_sectors: Names of source sectors
            target_sectors: Names of target sectors
            metadata: Provenance information
            
        Raises:
            ValueError: If matrix shape doesn't match sector lists
        """
        self.matrix = np.asarray(matrix, dtype=float)
        self.source_sectors = list(source_sectors)
        self.target_sectors = list(target_sectors)
        self.metadata = metadata or ConcordanceMetadata(
            source="unknown",
            method="unknown"
        )
        
        # Validate
        if self.matrix.shape != (len(self.source_sectors), len(self.target_sectors)):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} doesn't match "
                f"sectors ({len(self.source_sectors)}, {len(self.target_sectors)})"
            )
        
        # Check that rows sum to approximately 1 (each source sector fully allocated)
        row_sums = self.matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"Concordance matrix rows must sum to 1 (each source sector fully allocated). "
                f"Got row sums: {row_sums}"
            )
    
    @classmethod
    def from_supply_use_table(
        cls,
        supply_use_df: pd.DataFrame,
        source_col: str,
        target_col: str,
        value_col: str = "value",
        metadata: Optional[ConcordanceMetadata] = None,
    ) -> SectorConcordance:
        """
        Create concordance from supply-use table data.
        
        The supply-use table should have columns for source sector, target sector,
        and value (flow magnitude). The concordance is derived by normalizing
        flows so each source sector sums to 1.
        
        Args:
            supply_use_df: DataFrame with supply-use flows
            source_col: Column name for source sectors
            target_col: Column name for target sectors
            value_col: Column name for flow values
            metadata: Provenance information
            
        Returns:
            SectorConcordance instance
            
        Example:
            >>> sut = pd.DataFrame({
            ...     'ANZSIC': ['26', '26', '27', '27'],
            ...     'IOIG': ['Electricity', 'Transmission', 'Gas', 'Distribution'],
            ...     'value': [100, 50, 80, 20]
            ... })
            >>> concordance = SectorConcordance.from_supply_use_table(
            ...     sut,
            ...     source_col='ANZSIC',
            ...     target_col='IOIG',
            ...     value_col='value'
            ... )
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for from_supply_use_table")
        
        # Pivot to get matrix
        pivot = supply_use_df.pivot_table(
            index=source_col,
            columns=target_col,
            values=value_col,
            aggfunc='sum',
            fill_value=0
        )
        
        # Normalize rows to sum to 1
        row_sums = pivot.sum(axis=1)
        matrix = pivot.div(row_sums, axis=0).fillna(0)
        
        source_sectors = list(matrix.index)
        target_sectors = list(matrix.columns)
        
        if metadata is None:
            metadata = ConcordanceMetadata(
                source="supply_use_table",
                method="supply_use_table"
            )
        
        return cls(
            matrix=matrix.values,
            source_sectors=source_sectors,
            target_sectors=target_sectors,
            metadata=metadata
        )
    
    @classmethod
    def from_employment_shares(
        cls,
        employment_df: pd.DataFrame,
        source_col: str,
        target_col: str,
        employment_col: str = "employment",
        metadata: Optional[ConcordanceMetadata] = None,
    ) -> SectorConcordance:
        """
        Create concordance from employment allocation data.
        
        Assumes employment distribution reflects sector allocation.
        Less precise than supply-use but useful when detailed flow data unavailable.
        
        Args:
            employment_df: DataFrame with employment by source and target sectors
            source_col: Column name for source sectors
            target_col: Column name for target sectors
            employment_col: Column name for employment values
            metadata: Provenance information
            
        Returns:
            SectorConcordance instance
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for from_employment_shares")
        
        pivot = employment_df.pivot_table(
            index=source_col,
            columns=target_col,
            values=employment_col,
            aggfunc='sum',
            fill_value=0
        )
        
        row_sums = pivot.sum(axis=1)
        matrix = pivot.div(row_sums, axis=0).fillna(0)
        
        source_sectors = list(matrix.index)
        target_sectors = list(matrix.columns)
        
        if metadata is None:
            metadata = ConcordanceMetadata(
                source="employment_shares",
                method="employment_allocation",
                notes="Assumes employment distribution reflects sector allocation"
            )
        
        return cls(
            matrix=matrix.values,
            source_sectors=source_sectors,
            target_sectors=target_sectors,
            metadata=metadata
        )
    
    def allocate(
        self,
        source_data: Union[ArrayLike, Dict[str, float]],
    ) -> NDArray:
        """
        Allocate data from source to target sectors using concordance.
        
        Args:
            source_data: Either array of length n_source or dict mapping
                        source sector names to values
                        
        Returns:
            Array of allocated values for target sectors
            
        Example:
            >>> emissions_source = np.array([100, 50, 80])  # By ANZSIC
            >>> emissions_target = concordance.allocate(emissions_source)
            >>> # emissions_target now in IOIG classification
        """
        if isinstance(source_data, dict):
            # Convert dict to array
            source_array = np.array([
                source_data.get(sector, 0)
                for sector in self.source_sectors
            ])
        else:
            source_array = np.asarray(source_data, dtype=float)
        
        if source_array.shape[0] != len(self.source_sectors):
            raise ValueError(
                f"Source data length {source_array.shape[0]} doesn't match "
                f"number of source sectors {len(self.source_sectors)}"
            )
        
        # Matrix multiplication: target = source @ concordance
        return source_array @ self.matrix
    
    def validate(self) -> Dict[str, Union[bool, float, str]]:
        """
        Validate concordance matrix for common issues.
        
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Check row sums
        row_sums = self.matrix.sum(axis=1)
        results["rows_sum_to_one"] = np.allclose(row_sums, 1.0, atol=1e-6)
        results["row_sum_min"] = float(row_sums.min())
        results["row_sum_max"] = float(row_sums.max())
        
        # Check for negative values
        results["has_negative_values"] = bool((self.matrix < 0).any())
        results["min_value"] = float(self.matrix.min())
        
        # Check sparsity
        n_zeros = (self.matrix == 0).sum()
        total = self.matrix.size
        results["sparsity"] = float(n_zeros / total)
        
        # Check for all-zero rows/columns
        zero_rows = (self.matrix.sum(axis=1) == 0).sum()
        zero_cols = (self.matrix.sum(axis=0) == 0).sum()
        results["zero_rows"] = int(zero_rows)
        results["zero_columns"] = int(zero_cols)
        
        return results
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export concordance matrix as pandas DataFrame.
        
        Returns:
            DataFrame with source sectors as index, target sectors as columns
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for to_dataframe")
        
        return pd.DataFrame(
            self.matrix,
            index=self.source_sectors,
            columns=self.target_sectors
        )
    
    def to_csv(self, path: str) -> None:
        """Export concordance matrix to CSV file."""
        self.to_dataframe().to_csv(path)
    
    def __repr__(self) -> str:
        return (
            f"SectorConcordance("
            f"{len(self.source_sectors)} source sectors → "
            f"{len(self.target_sectors)} target sectors, "
            f"method={self.metadata.method})"
        )
    
    def __str__(self) -> str:
        lines = [
            f"Sector Concordance Matrix",
            f"{'=' * 50}",
            f"Source sectors: {len(self.source_sectors)}",
            f"Target sectors: {len(self.target_sectors)}",
            f"",
            f"Metadata:",
            str(self.metadata),
            f"",
            f"Validation:",
        ]
        
        validation = self.validate()
        for key, value in validation.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


class SatelliteWithConcordance:
    """
    Environmental satellite data with sector concordance mapping.
    
    Combines direct intensity data with a concordance matrix to enable
    flexible sector classification alignment.
    
    Example:
        >>> satellite = SatelliteWithConcordance(
        ...     intensities=emissions_by_anzsic,
        ...     concordance=concordance_matrix,
        ...     name="carbon",
        ...     unit="kg CO2-eq / AUD"
        ... )
        >>> 
        >>> # Get intensities in target classification
        >>> emissions_ioig = satellite.get_intensities_target()
    """
    
    def __init__(
        self,
        intensities: ArrayLike,
        concordance: SectorConcordance,
        name: str = "satellite",
        unit: str = "",
    ):
        """
        Initialize satellite with concordance.
        
        Args:
            intensities: Direct intensities in source classification
            concordance: SectorConcordance mapping source to target
            name: Satellite name (e.g., 'carbon', 'water')
            unit: Unit of measurement
        """
        self.intensities_source = np.asarray(intensities, dtype=float)
        self.concordance = concordance
        self.name = name
        self.unit = unit
        
        if len(self.intensities_source) != len(concordance.source_sectors):
            raise ValueError(
                f"Intensities length {len(self.intensities_source)} doesn't match "
                f"concordance source sectors {len(concordance.source_sectors)}"
            )
    
    def get_intensities_target(self) -> NDArray:
        """
        Get intensities in target sector classification.
        
        Returns:
            Array of intensities for target sectors
        """
        return self.concordance.allocate(self.intensities_source)
    
    def get_intensities_source(self) -> NDArray:
        """Get intensities in source sector classification."""
        return self.intensities_source.copy()
    
    def __repr__(self) -> str:
        return (
            f"SatelliteWithConcordance("
            f"name={self.name}, "
            f"source_sectors={len(self.concordance.source_sectors)}, "
            f"target_sectors={len(self.concordance.target_sectors)})"
        )
