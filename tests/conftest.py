"""
Test fixtures for fastspa testing.
"""
import numpy as np
import pytest

from fastspa import SPA, SectorConcordance, SatelliteWithConcordance
from fastspa.concordance import ConcordanceMetadata


@pytest.fixture
def sample_a_matrix():
    """A simple 3x3 A-matrix for testing."""
    # Simple economy: Agriculture, Manufacturing, Services
    return np.array([
        [0.1, 0.2, 0.1],  # Agriculture inputs
        [0.3, 0.1, 0.4],  # Manufacturing inputs
        [0.2, 0.3, 0.2],  # Services inputs
    ])


@pytest.fixture
def sample_intensities():
    """Sample direct intensities for 3 sectors."""
    return np.array([0.5, 1.0, 0.8])  # kg CO2 per unit output


@pytest.fixture
def sample_sectors():
    """Sample sector names."""
    return ["Agriculture", "Manufacturing", "Services"]


@pytest.fixture
def sample_spa(sample_a_matrix, sample_intensities, sample_sectors):
    """Basic SPA instance for testing."""
    return SPA(sample_a_matrix, sample_intensities, sample_sectors)


@pytest.fixture
def sample_multi_satellite(sample_a_matrix, sample_sectors):
    """SPA with multiple satellites."""
    intensities = {
        "ghg": np.array([0.5, 1.0, 0.8]),
        "water": np.array([10.0, 5.0, 15.0]),
        "energy": np.array([2.0, 3.0, 1.5])
    }
    return SPA(sample_a_matrix, intensities, sample_sectors)


@pytest.fixture
def sample_concordance():
    """Sample sector concordance for testing."""
    # Source sectors (e.g., NIBES) -> Target sectors (IO)
    source_sectors = ["Crop Farming", "Livestock", "Food Processing"]
    target_sectors = ["Agriculture", "Manufacturing", "Services"]

    # Concordance matrix: rows=source, cols=target
    # Each row must sum to 1.0 (full allocation)
    concordance_matrix = np.array([
        [0.8, 0.1, 0.1],  # Crop Farming -> Agriculture (80%), Manufacturing (10%), Services (10%)
        [0.9, 0.05, 0.05],  # Livestock -> Agriculture (90%), Manufacturing (5%), Services (5%)
        [0.0, 0.9, 0.1],  # Food Processing -> Manufacturing (90%), Services (10%)
    ])

    return SectorConcordance(
        source_sectors=source_sectors,
        target_sectors=target_sectors,
        matrix=concordance_matrix,
        metadata=ConcordanceMetadata(
            source="Test Data",
            method="manual",
            year=2023
        )
    )


@pytest.fixture
def sample_satellite_with_concordance(sample_concordance):
    """Satellite with concordance mapping."""
    # Source intensities (NIBES sectors)
    source_intensities = np.array([0.6, 0.4, 1.2])

    return SatelliteWithConcordance(
        intensities=source_intensities,
        concordance=sample_concordance,
        name="ghg_concordance",
        unit="kg CO2-e"
    )


@pytest.fixture
def sample_leontief(sample_a_matrix):
    """Pre-computed Leontief inverse."""
    I = np.eye(3)
    L = np.linalg.inv(I - sample_a_matrix)
    return L