# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-18

### Added
- Initial release of fastspa - Modern Structural Path Analysis for EEIO
- NumPy-native, functional interface for Structural Path Analysis
- Two analysis modes:
  - Sector-specific analysis (A-matrix mode): Direct supplier relationships
  - System-wide analysis (Leontief mode): Total requirements including circular flows
- Support for multiple environmental satellites
- Comprehensive path extraction and analysis capabilities
- Advanced features:
  - Uncertainty quantification with Monte Carlo methods
  - Network analysis and bottleneck identification
  - Temporal analysis for multi-year workflows
  - Loop detection for circular economy analysis
  - Semantic aggregation for stakeholder-friendly results
- Factory functions for various data input formats:
  - `from_leontief()`: From pre-computed Leontief inverse
  - `from_dataframe()`: From pandas DataFrames
  - `from_csv()`: From CSV files
  - `from_io_table()`: From Excel IO tables
- Interactive visualization capabilities with plotly
- Evidence-based sector concordance for national inventory mapping
- Comprehensive documentation and examples
- Full test suite with pytest
- Code quality tools: black, ruff, mypy

### Dependencies
- Core: numpy>=1.20, ipfn>=2.1.0
- Optional: pandas>=1.3, scipy>=1.7, matplotlib>=3.4, plotly>=5.0
- Development: pytest>=7.0, pytest-cov>=4.0, black>=23.0, ruff>=0.1, mypy>=1.0

### Package Information
- Python >= 3.9
- License: MIT
- Status: Beta