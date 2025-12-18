# fastspa Test Suite

This directory contains comprehensive tests for the fastspa package, covering all implemented features described in the main README.

## Test Structure

- `conftest.py`: Shared fixtures for test data (A-matrices, intensities, sectors, etc.)
- `test_basic_spa.py`: Tests for basic SPA creation and analysis
- `test_multi_satellite.py`: Tests for multi-satellite analysis
- `test_concordance.py`: Tests for sector concordance features
- `test_factory_functions.py`: Tests for factory functions (from_leontief, from_dataframe, etc.)
- `test_path_collection.py`: Tests for PathCollection methods and Path properties
- `test_spa_methods.py`: Tests for advanced SPA methods (analyze_many, hotspots, etc.)

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=fastspa --cov-report=html

# Run specific test file
pytest test_basic_spa.py -v
```

## Test Coverage

The test suite covers:

- ✅ Basic SPA functionality (sector-specific and system-wide modes)
- ✅ Multi-satellite analysis
- ✅ Sector concordance with SatelliteWithConcordance
- ✅ All factory functions (from_leontief, from_dataframe, from_csv, from_io_table)
- ✅ PathCollection filtering, aggregation, and export methods
- ✅ Path object properties
- ✅ Advanced SPA methods (analyze_many, hotspots, compare_sectors, stream)

## Notes

- Visualization tests are not included as the `fastspa.viz` module is not yet implemented
- Tests use pytest fixtures for reusable test data
- All tests are designed to be fast and focused on functionality rather than exhaustive edge cases