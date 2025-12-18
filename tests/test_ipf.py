"""
Tests for IPF (Iterative Proportional Fitting) functionality.
"""

import numpy as np
import pytest

try:
    from fastspa import IPFBalancer, IPFResult, adjust_io_table_with_concordance
    HAS_IPF = True
except ImportError:
    HAS_IPF = False


@pytest.mark.skipif(not HAS_IPF, reason="IPF functionality not available")
class TestIPFBalancer:
    """Test the IPFBalancer class."""
    
    def test_init(self):
        """Test IPFBalancer initialization."""
        balancer = IPFBalancer()
        assert balancer is not None
    
    def test_adjust_io_matrix_basic(self):
        """Test basic IO matrix adjustment."""
        balancer = IPFBalancer()
        
        # Simple 2x2 example
        io_matrix = np.array([[100, 50], [30, 200]])
        new_row_totals = np.array([180, 250])
        new_col_totals = np.array([150, 280])
        
        result = balancer.adjust_io_matrix(io_matrix, new_row_totals, new_col_totals)
        
        assert isinstance(result, IPFResult)
        assert result.adjusted_matrix.shape == io_matrix.shape
        assert result.original_matrix.shape == io_matrix.shape
        
        # Check row constraints
        adjusted_rows = result.adjusted_matrix.sum(axis=1)
        np.testing.assert_allclose(adjusted_rows, new_row_totals, rtol=1e-5)
        
        # Check column constraints
        adjusted_cols = result.adjusted_matrix.sum(axis=0)
        np.testing.assert_allclose(adjusted_cols, new_col_totals, rtol=1e-5)
        
        # Check convergence
        assert result.converged in [True, False]  # Check it's boolean-like
        assert isinstance(result.convergence_rate, (float, int))
        assert isinstance(result.iterations, int)
        assert isinstance(result.relative_entropy, (float, int))
    
    def test_adjust_io_matrix_validation(self):
        """Test input validation for IO matrix adjustment."""
        balancer = IPFBalancer()
        
        # Non-square matrix
        with pytest.raises(ValueError, match="must be square"):
            balancer.adjust_io_matrix(
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([3, 6]),
                np.array([5, 7])
            )
        
        # Wrong total dimensions
        with pytest.raises(ValueError, match="Row totals must have length"):
            balancer.adjust_io_matrix(
                np.array([[1, 2], [3, 4]]),
                np.array([3, 6, 9]),  # Wrong length
                np.array([5, 7])
            )
        
        # Negative totals
        with pytest.raises(ValueError, match="totals must be positive"):
            balancer.adjust_io_matrix(
                np.array([[1, 2], [3, 4]]),
                np.array([3, -6]),  # Negative
                np.array([5, 7])
            )
    
    def test_adjust_a_matrix(self):
        """Test A matrix adjustment."""
        balancer = IPFBalancer()
        
        A_matrix = np.array([[0.1, 0.05], [0.03, 0.15]])
        new_row_totals = np.array([1.2, 1.8])
        
        result = balancer.adjust_a_matrix(A_matrix, new_row_totals)
        
        assert isinstance(result, IPFResult)
        assert result.adjusted_matrix.shape == A_matrix.shape
        
        # Check that matrix is still valid (non-negative, reasonable values)
        assert np.all(result.adjusted_matrix >= 0)
        assert np.all(result.adjusted_matrix <= 2.0)  # Reasonable upper bound
    
    def test_adjust_satellite_data_1d(self):
        """Test 1D satellite data adjustment."""
        balancer = IPFBalancer()
        
        satellite_data = np.array([100, 50, 80, 30])
        new_totals = np.array([120, 60, 95, 35])
        
        result = balancer.adjust_satellite_data(satellite_data, [new_totals])
        
        assert isinstance(result, IPFResult)
        assert result.adjusted_matrix.shape == satellite_data.shape
        
        # Check constraint satisfaction
        adjusted_sum = result.adjusted_matrix.sum()
        np.testing.assert_allclose(adjusted_sum, new_totals.sum(), rtol=1e-5)
    
    def test_adjust_satellite_data_2d(self):
        """Test 2D satellite data adjustment."""
        balancer = IPFBalancer()
        
        # 2D satellite data (e.g., emissions by sector and region)
        satellite_data = np.array([
            [100, 50],
            [30, 80],
            [60, 40]
        ])
        
        # New totals for each region (sum over sectors)
        region_totals = np.array([190, 170])  # Two regions
        
        result = balancer.adjust_satellite_data(
            satellite_data,
            [region_totals],
            dimensions=[1]  # Sum over second dimension (regions)
        )
        
        assert isinstance(result, IPFResult)
        assert result.adjusted_matrix.shape == satellite_data.shape
        
        # Check region totals (sum over sectors, which is dimension 1)
        achieved_region_totals = result.adjusted_matrix.sum(axis=0)
        np.testing.assert_allclose(achieved_region_totals, region_totals, rtol=1e-5)
    
    def test_balance_multi_sector_data(self):
        """Test balancing multiple matrices simultaneously."""
        balancer = IPFBalancer()
        
        matrices = {
            'ghg': np.array([[100, 50], [30, 200]]),
            'water': np.array([[80, 40], [25, 160]])
        }
        
        constraints = {
            'ghg': {
                'rows': np.array([180, 250]),
                'cols': np.array([150, 280])
            },
            'water': {
                'rows': np.array([140, 200]),
                'cols': np.array([120, 220])
            }
        }
        
        results = balancer.balance_multi_sector_data(matrices, constraints)
        
        assert len(results) == 2
        assert 'ghg' in results
        assert 'water' in results
        
        # Check both results
        for name, result in results.items():
            assert isinstance(result, IPFResult)
            assert result.adjusted_matrix.shape == matrices[name].shape


@pytest.mark.skipif(not HAS_IPF, reason="IPF functionality not available")
class TestAdjustWithConcordance:
    """Test the adjust_io_table_with_concordance function."""
    
    def test_basic_concordance_adjustment(self):
        """Test basic IO table adjustment with concordance."""
        # Simple 2x2 example
        io_matrix = np.array([[100, 50], [30, 200]])
        
        # Concordance matrix (2 source sectors to 2 target sectors)
        concordance = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        # Target totals in target classification
        target_totals = np.array([200, 300])
        
        adjusted_io, result = adjust_io_table_with_concordance(
            io_matrix, concordance, target_totals
        )
        
        assert adjusted_io.shape == io_matrix.shape
        assert isinstance(result, IPFResult)
        
        # Check that concordance rows sum to 1
        concordance_rows = concordance.sum(axis=1)
        np.testing.assert_allclose(concordance_rows, 1.0, atol=1e-6)
    
    def test_concordance_validation(self):
        """Test input validation for concordance adjustment."""
        # Non-square IO matrix
        with pytest.raises(ValueError, match="must be"):
            adjust_io_table_with_concordance(
                np.array([[1, 2, 3], [4, 5, 6]]),  # 2x3, not square
                np.array([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]),
                np.array([100, 200])
            )
        
        # Concordance rows don't sum to 1
        with pytest.raises(ValueError, match="rows must sum to 1"):
            adjust_io_table_with_concordance(
                np.array([[100, 50], [30, 200]]),
                np.array([[0.5, 0.3], [0.4, 0.8]]),  # Rows sum to 0.8 and 1.2
                np.array([200, 300])
            )


@pytest.mark.skipif(not HAS_IPF, reason="IPF functionality not available")
class TestIPFResult:
    """Test the IPFResult dataclass."""
    
    def test_ipf_result_creation(self):
        """Test creating IPFResult objects."""
        original = np.array([[100, 50], [30, 200]])
        adjusted = np.array([[120, 80], [40, 190]])
        
        result = IPFResult(
            adjusted_matrix=adjusted,
            original_matrix=original,
            iterations=10,
            convergence_rate=1e-6,
            converged=True,
            row_constraints=np.array([200, 230]),
            col_constraints=np.array([160, 270]),
            relative_entropy=0.05
        )
        
        assert result.adjusted_matrix is adjusted
        assert result.original_matrix is original
        assert result.iterations == 10
        assert result.convergence_rate == 1e-6
        assert result.converged is True
        np.testing.assert_array_equal(result.row_constraints, np.array([200, 230]))
        np.testing.assert_array_equal(result.col_constraints, np.array([160, 270]))
        assert result.relative_entropy == 0.05
    
    def test_ipf_result_validation(self):
        """Test IPFResult validation."""
        original = np.array([[100, 50], [30, 200]])
        adjusted = np.array([[120, 80, 10], [40, 190, 20]])  # Different shape
        
        with pytest.raises(ValueError, match="must have same shape"):
            IPFResult(
                adjusted_matrix=adjusted,
                original_matrix=original,
                iterations=10,
                convergence_rate=1e-6,
                converged=True,
                row_constraints=np.array([200, 230]),
                col_constraints=np.array([160, 270]),
                relative_entropy=0.05
            )


@pytest.mark.skipif(HAS_IPF, reason="Testing behavior when IPF is not available")
class TestIPFImportError:
    """Test behavior when IPF is not available."""
    
    def test_import_error(self):
        """Test that importing IPF classes raises appropriate error."""
        with pytest.raises(ImportError, match="ipfn package required"):
            IPFBalancer()