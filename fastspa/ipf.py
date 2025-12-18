"""
Iterative Proportional Fitting (IPF) for Sector Adjustments

IPF is used to adjust IO tables (using the RAS method) and multi-dimensional
satellite data to match updated sector constraints while preserving the
underlying structure and cross-product ratios.

For simple 1D sector adjustments (e.g., scaling a single sector's emissions),
direct scalar multiplication is often sufficient. IPF is required when
balancing matrices against both row and column constraints (IO tables) or
when adjusting multi-dimensional arrays (e.g., Sector x Region).

Based on the ipfn package: https://github.com/Dirguis/ipfn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from ipfn.ipfn import ipfn as ipfn_core
    HAS_IPFN = True
except ImportError:
    HAS_IPFN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class IPFResult:
    """
    Results from an IPF adjustment operation.
    
    Attributes:
        adjusted_matrix: The final adjusted matrix after IPF convergence
        original_matrix: The original input matrix
        iterations: Number of iterations taken to converge
        convergence_rate: Final convergence rate
        converged: Whether the algorithm converged within max_iterations
        row_constraints: Target row sums that were applied
        col_constraints: Target column sums that were applied
        relative_entropy: Relative entropy distance from original to adjusted
    """
    adjusted_matrix: NDArray
    original_matrix: NDArray
    iterations: int
    convergence_rate: float
    converged: bool
    row_constraints: NDArray
    col_constraints: NDArray
    relative_entropy: float
    
    def __post_init__(self):
        """Validate the IPF result."""
        if self.adjusted_matrix.shape != self.original_matrix.shape:
            raise ValueError("Adjusted matrix must have same shape as original")
        
        # Check row sums (only if row_constraints are provided and matrix is 2D and shapes match)
        if (len(self.row_constraints) > 0 and
            self.adjusted_matrix.ndim >= 2 and
            self.row_constraints.shape == (self.adjusted_matrix.shape[0],)):
            adjusted_rows = self.adjusted_matrix.sum(axis=1)
            if not np.allclose(adjusted_rows, self.row_constraints, rtol=1e-5):
                raise ValueError("Row constraints not satisfied in adjusted matrix")
        
        # Check column sums (only if col_constraints are provided and matrix is 2D and shapes match)
        if (len(self.col_constraints) > 0 and
            self.adjusted_matrix.ndim >= 2 and
            self.col_constraints.shape == (self.adjusted_matrix.shape[1],)):
            adjusted_cols = self.adjusted_matrix.sum(axis=0)
            if not np.allclose(adjusted_cols, self.col_constraints, rtol=1e-5):
                raise ValueError("Column constraints not satisfied in adjusted matrix")


class IPFBalancer:
    """
    Iterative Proportional Fitting for sector adjustments.
    
    Provides high-level interface for adjusting IO tables, A matrices,
    and satellite data to match updated sector constraints.
    
    Example:
        >>> # Adjust A matrix to new sector totals
        >>> balancer = IPFBalancer()
        >>> result = balancer.adjust_io_matrix(
        ...     A_matrix,
        ...     new_row_totals=new_output_totals,
        ...     new_col_totals=new_input_totals
        ... )
        >>> adjusted_A = result.adjusted_matrix
    """
    
    def __init__(self):
        """Initialize the IPF balancer."""
        if not HAS_IPFN:
            raise ImportError(
                "ipfn package required for IPF functionality. "
                "Install with: pip install ipfn"
            )
    
    def adjust_io_matrix(
        self,
        io_matrix: ArrayLike,
        new_row_totals: ArrayLike,
        new_col_totals: ArrayLike,
        *,
        max_iterations: int = 1000,
        convergence_rate: float = 1e-6,
        rate_tolerance: float = 0.0,
        verbose: int = 0,
    ) -> IPFResult:
        """
        Adjust an IO transaction matrix to match new row and column totals.
        
        This implements the RAS method (biproportional scaling), which is the
        standard technique for updating IO tables to new sector outputs and uses.
        
        Args:
            io_matrix: Original transaction matrix (n×n)
            new_row_totals: Target row sums (outputs by sector)
            new_col_totals: Target column sums (inputs by sector)
            max_iterations: Maximum number of iterations
            convergence_rate: Convergence tolerance
            rate_tolerance: Tolerance for early stopping based on rate change
            verbose: Verbosity level (0=silent, 1=status, 2=detailed)
            
        Returns:
            IPFResult with adjusted matrix and convergence information
            
        Example:
            >>> # Update IO table to reflect new economic structure
            >>> io_matrix = np.array([[100, 50], [30, 200]])
            >>> new_outputs = np.array([180, 250])  # New sector outputs
            >>> new_inputs = np.array([150, 280])   # New sector inputs
            >>> result = balancer.adjust_io_matrix(io_matrix, new_outputs, new_inputs)
        """
        io_matrix = np.asarray(io_matrix, dtype=float)
        new_row_totals = np.asarray(new_row_totals, dtype=float)
        new_col_totals = np.asarray(new_col_totals, dtype=float)
        
        # Validate inputs
        if io_matrix.ndim != 2:
            raise ValueError("IO matrix must be 2D")
        
        if io_matrix.shape[0] != io_matrix.shape[1]:
            raise ValueError("IO matrix must be square")
        
        n = io_matrix.shape[0]
        
        if new_row_totals.shape != (n,):
            raise ValueError(f"Row totals must have length {n}")
        
        if new_col_totals.shape != (n,):
            raise ValueError(f"Column totals must have length {n}")
        
        if np.any(new_row_totals <= 0):
            raise ValueError("Row totals must be positive")
        
        if np.any(new_col_totals <= 0):
            raise ValueError("Column totals must be positive")
        
        # Apply IPF
        aggregates = [new_row_totals, new_col_totals]
        dimensions = [[0], [1]]  # Sum over rows (dim 0), then columns (dim 1)
        
        IPF = ipfn_core(
            io_matrix,
            aggregates,
            dimensions,
            convergence_rate=convergence_rate,
            max_iteration=max_iterations,
            verbose=verbose,
            rate_tolerance=rate_tolerance
        )
        
        adjusted_matrix = IPF.iteration()
        
        # Calculate convergence metrics
        final_rate = self._calculate_convergence_rate(
            io_matrix, adjusted_matrix, new_row_totals, new_col_totals
        )
        
        # Check convergence
        converged = final_rate <= convergence_rate
        
        # Calculate relative entropy (Kullback-Leibler divergence)
        rel_entropy = self._calculate_relative_entropy(io_matrix, adjusted_matrix)
        
        return IPFResult(
            adjusted_matrix=adjusted_matrix,
            original_matrix=io_matrix.copy(),
            iterations=IPF.iteration_counter if hasattr(IPF, 'iteration_counter') else 0,
            convergence_rate=final_rate,
            converged=converged,
            row_constraints=new_row_totals.copy(),
            col_constraints=new_col_totals.copy(),
            relative_entropy=rel_entropy,
        )
    
    def adjust_a_matrix(
        self,
        A_matrix: ArrayLike,
        new_row_totals: ArrayLike,
        new_col_totals: Optional[ArrayLike] = None,
        *,
        preserve_structure: bool = True,
        max_iterations: int = 1000,
        convergence_rate: float = 1e-6,
        verbose: int = 0,
    ) -> IPFResult:
        """
        Adjust technical coefficients matrix to new sector totals.
        
        Args:
            A_matrix: Technical coefficients matrix (n×n)
            new_row_totals: Target sector outputs (typically gross output)
            new_col_totals: Target sector inputs (if None, use row totals)
            preserve_structure: Whether to preserve zero structure
            max_iterations: Maximum iterations
            convergence_rate: Convergence tolerance
            verbose: Verbosity level
            
        Returns:
            IPFResult with adjusted A matrix
            
        Example:
            >>> # Update technical coefficients to new economic structure
            >>> result = balancer.adjust_a_matrix(
            ...     A_matrix,
            ...     new_row_totals=new_gross_outputs
            ... )
        """
        A_matrix = np.asarray(A_matrix, dtype=float)
        new_row_totals = np.asarray(new_row_totals, dtype=float)
        
        if A_matrix.ndim != 2:
            raise ValueError("A matrix must be 2D")
        
        n = A_matrix.shape[0]
        
        if new_row_totals.shape != (n,):
            raise ValueError(f"Row totals must have length {n}")
        
        if new_col_totals is None:
            new_col_totals = new_row_totals.copy()
        else:
            new_col_totals = np.asarray(new_col_totals, dtype=float)
            if new_col_totals.shape != (n,):
                raise ValueError(f"Column totals must have length {n}")
        
        # If preserving structure, set zero elements to small positive values
        if preserve_structure:
            adjusted_A = A_matrix.copy()
            zero_mask = (A_matrix == 0)
            # Use minimum positive value to avoid division by zero
            min_positive = A_matrix[A_matrix > 0].min() if np.any(A_matrix > 0) else 1e-12
            adjusted_A[zero_mask] = min_positive * 1e-6
        else:
            adjusted_A = A_matrix.copy()
        
        # Apply IPF to get scaling factors
        result = self.adjust_io_matrix(
            adjusted_A,
            new_row_totals,
            new_col_totals,
            max_iterations=max_iterations,
            convergence_rate=convergence_rate,
            verbose=verbose
        )
        
        # Restore zero structure if requested
        if preserve_structure:
            result.adjusted_matrix[zero_mask] = 0.0
        
        return result
    
    def adjust_satellite_data(
        self,
        satellite_data: ArrayLike,
        new_totals: ArrayLike,
        *,
        dimensions: Optional[List[int]] = None,
        max_iterations: int = 1000,
        convergence_rate: float = 1e-6,
    ) -> IPFResult:
        """
        Adjust satellite data (emissions, resource use) to new sector totals.
        
        Uses generalized IPF to handle multi-dimensional constraints (e.g.,
        Sector x Region). For simple 1D satellite vectors, this is equivalent
        to scalar scaling but can be used for consistency.
        
        Args:
            satellite_data: Satellite data array (can be 1D, 2D, or 3D)
            new_totals: Target totals along specified dimensions
            dimensions: Which dimensions to constrain (default: first dimension)
            max_iterations: Maximum iterations
            convergence_rate: Convergence tolerance
            
        Returns:
            IPFResult with adjusted satellite data
        """
        satellite_data = np.asarray(satellite_data, dtype=float)
        new_totals = np.asarray(new_totals, dtype=float)
        
        if dimensions is None:
            dimensions = [0]  # Default: constrain first dimension
        
        # Validate dimensions
        if any(dim < 0 or dim >= satellite_data.ndim for dim in dimensions):
            raise ValueError("Invalid dimension specified")
        
        if len(dimensions) != len(new_totals):
            raise ValueError("Number of dimension constraints must match new_totals")
        
        # Validate new_totals shapes
        for dim, total in zip(dimensions, new_totals):
            expected_shape = satellite_data.shape[dim]
            if total.shape != (expected_shape,):
                raise ValueError(
                    f"Total for dimension {dim} must have length {expected_shape}"
                )
        
        # Prepare aggregates and dimensions for IPF
        aggregates = [np.asarray(total, dtype=float) for total in new_totals]
        ipf_dimensions = [[dim] for dim in dimensions]
        
        # Apply IPF
        IPF = ipfn_core(
            satellite_data,
            aggregates,
            ipf_dimensions,
            convergence_rate=convergence_rate,
            max_iteration=max_iterations,
            verbose=0
        )
        
        adjusted_data = IPF.iteration()
        
        # Calculate metrics
        final_rate = self._calculate_convergence_rate_satellite(
            satellite_data, adjusted_data, new_totals, dimensions
        )
        
        converged = final_rate <= convergence_rate
        rel_entropy = self._calculate_relative_entropy(satellite_data, adjusted_data)
        
        return IPFResult(
            adjusted_matrix=adjusted_data,
            original_matrix=satellite_data.copy(),
            iterations=IPF.iteration_counter if hasattr(IPF, 'iteration_counter') else 0,
            convergence_rate=final_rate,
            converged=converged,
            row_constraints=new_totals[0] if len(new_totals) > 0 else np.array([]),
            col_constraints=np.array([]),  # Not applicable for satellite data
            relative_entropy=rel_entropy,
        )
    
    def balance_multi_sector_data(
        self,
        data_matrices: Dict[str, ArrayLike],
        constraints: Dict[str, Dict[str, ArrayLike]],
        *,
        max_iterations: int = 1000,
        convergence_rate: float = 1e-6,
    ) -> Dict[str, IPFResult]:
        """
        Balance multiple related data matrices simultaneously.
        
        Useful for balancing a set of matrices (e.g., different satellite accounts)
        that share the same sector structure but have different constraints.
        
        Args:
            data_matrices: Dict mapping names to matrices to adjust
            constraints: Dict mapping matrix names to constraint dicts:
                        {'matrix_name': {'rows': row_totals, 'cols': col_totals}}
            max_iterations: Maximum iterations for each matrix
            convergence_rate: Convergence tolerance
            
        Returns:
            Dict mapping matrix names to their IPFResult
            
        Example:
            >>> matrices = {'ghg': ghg_matrix, 'water': water_matrix}
            >>> constraints = {
            ...     'ghg': {'rows': new_ghg_totals, 'cols': new_ghg_inputs},
            ...     'water': {'rows': new_water_totals, 'cols': new_water_inputs}
            ... }
            >>> results = balancer.balance_multi_sector_data(matrices, constraints)
        """
        results = {}
        
        for name, matrix in data_matrices.items():
            if name not in constraints:
                raise ValueError(f"No constraints provided for matrix '{name}'")
            
            matrix_constraints = constraints[name]
            
            if 'rows' in matrix_constraints and 'cols' in matrix_constraints:
                # 2D matrix with both row and column constraints
                result = self.adjust_io_matrix(
                    matrix,
                    matrix_constraints['rows'],
                    matrix_constraints['cols'],
                    max_iterations=max_iterations,
                    convergence_rate=convergence_rate
                )
            elif 'rows' in matrix_constraints:
                # 1D or multi-D with only row constraints
                result = self.adjust_satellite_data(
                    matrix,
                    [matrix_constraints['rows']],
                    dimensions=[0],
                    max_iterations=max_iterations,
                    convergence_rate=convergence_rate
                )
            else:
                raise ValueError(
                    f"Matrix '{name}' must have 'rows' constraint at minimum"
                )
            
            results[name] = result
        
        return results
    
    def _calculate_convergence_rate(
        self,
        original: NDArray,
        adjusted: NDArray,
        target_rows: NDArray,
        target_cols: NDArray,
    ) -> float:
        """Calculate convergence rate for IO matrix adjustment."""
        # Calculate achieved row and column sums
        achieved_rows = adjusted.sum(axis=1)
        achieved_cols = adjusted.sum(axis=0)
        
        # Calculate relative errors
        row_errors = np.abs(achieved_rows - target_rows) / (target_rows + 1e-12)
        col_errors = np.abs(achieved_cols - target_cols) / (target_cols + 1e-12)
        
        # Return maximum relative error
        return max(row_errors.max(), col_errors.max())
    
    def _calculate_convergence_rate_satellite(
        self,
        original: NDArray,
        adjusted: NDArray,
        targets: List[NDArray],
        dimensions: List[int],
    ) -> float:
        """Calculate convergence rate for satellite data adjustment."""
        max_error = 0.0
        
        for target, dim in zip(targets, dimensions):
            # Sum along the specified dimension
            achieved = adjusted.sum(axis=dim)
            
            # Convert to numpy array to handle both scalars and arrays
            achieved = np.asarray(achieved)
            target = np.asarray(target)
            
            # Ensure target and achieved have the same shape
            if achieved.shape != target.shape:
                # Take the minimum length to avoid index errors
                min_len = min(achieved.size, target.size)
                achieved = achieved.flat[:min_len]
                target = target.flat[:min_len]
            
            # Calculate relative errors
            errors = np.abs(achieved - target) / (target + 1e-12)
            max_error = max(max_error, errors.max())
        
        return max_error
    
    def _calculate_relative_entropy(self, original: NDArray, adjusted: NDArray) -> float:
        """Calculate relative entropy (Kullback-Leibler divergence)."""
        # Avoid log(0) by adding small constant
        epsilon = 1e-12
        original_safe = original + epsilon
        adjusted_safe = adjusted + epsilon
        
        # Normalize to make it a proper probability distribution
        original_norm = original_safe / original_safe.sum()
        adjusted_norm = adjusted_safe / adjusted_safe.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(original_norm * np.log(original_norm / adjusted_norm))
        
        return float(kl_div)


def adjust_io_table_with_concordance(
    io_matrix: ArrayLike,
    concordance_matrix: ArrayLike,
    target_sector_totals: ArrayLike,
    *,
    max_iterations: int = 1000,
    convergence_rate: float = 1e-6,
) -> Tuple[NDArray, IPFResult]:
    """
    Adjust IO table using sector concordance to target totals.
    
    This function combines IPF adjustment with sector concordance mapping
    to adjust an IO table from one sector classification to match totals
    in a different classification.
    
    Args:
        io_matrix: Original IO matrix in source classification
        concordance_matrix: Mapping from source to target sectors
        target_sector_totals: Target totals in target classification
        max_iterations: Maximum IPF iterations
        convergence_rate: Convergence tolerance
        
    Returns:
        Tuple of (adjusted_io_matrix, ipf_result)
        
    Example:
        >>> # Adjust IO table from ANZSIC to IOIG classification
        >>> adjusted_io, result = adjust_io_table_with_concordance(
        ...     io_anzsic,
        ...     concordance_matrix,
        ...     target_ioig_totals
        ... )
    """
    io_matrix = np.asarray(io_matrix, dtype=float)
    concordance_matrix = np.asarray(concordance_matrix, dtype=float)
    target_sector_totals = np.asarray(target_sector_totals, dtype=float)
    
    # Validate concordance matrix
    if concordance_matrix.ndim != 2:
        raise ValueError("Concordance matrix must be 2D")
    
    n_source, n_target = concordance_matrix.shape
    
    # Validate dimensions
    if io_matrix.shape != (n_source, n_source):
        raise ValueError(
            f"IO matrix must be {n_source}×{n_source} to match concordance"
        )
    
    if target_sector_totals.shape != (n_target,):
        raise ValueError(
            f"Target totals must have length {n_target} to match concordance"
        )
    
    # Check concordance matrix is properly normalized
    row_sums = concordance_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Concordance matrix rows must sum to 1")
    
    # Transform IO matrix to target classification
    io_target = concordance_matrix.T @ io_matrix @ concordance_matrix
    
    # Apply IPF to match target totals
    balancer = IPFBalancer()
    result = balancer.adjust_io_matrix(
        io_target,
        target_sector_totals,
        target_sector_totals,  # Assume symmetric constraints
        max_iterations=max_iterations,
        convergence_rate=convergence_rate
    )
    
    # Transform back to source classification if needed
    adjusted_target = result.adjusted_matrix
    adjusted_source = concordance_matrix @ adjusted_target @ concordance_matrix.T
    
    # Create result with source classification matrix
    adjusted_result = IPFResult(
        adjusted_matrix=adjusted_source,
        original_matrix=io_matrix,
        iterations=result.iterations,
        convergence_rate=result.convergence_rate,
        converged=result.converged,
        row_constraints=np.array([]),  # Empty for transformed data
        col_constraints=np.array([]),  # Empty for transformed data
        relative_entropy=result.relative_entropy,
    )
    
    return adjusted_source, adjusted_result