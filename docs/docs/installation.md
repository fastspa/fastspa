---
sidebar_position: 2
---

# Installation

## Requirements

- Python >= 3.9
- NumPy >= 1.20

## Install from PyPI

```bash
pip install fastspa
```

## Optional Dependencies

FastSPA has optional dependencies for extended functionality:

### pandas (DataFrame export)

```bash
pip install fastspa[pandas]
# or
pip install pandas
```

Required for:
- `paths.to_dataframe()`
- `from_dataframe()` factory function
- `spa.compare_sectors()`

### scipy (Sparse matrices)

```bash
pip install fastspa[scipy]
# or
pip install scipy
```

Required for:
- Working with sparse A-matrices
- Loading MATLAB `.mat` files

### Visualization (Plotly)

```bash
pip install fastspa[viz]
# or
pip install plotly
```

Required for:
- Interactive icicle plots (hierarchical supply chain visualization)
- Sunburst charts (radial hierarchical view)
- Sector contribution bar charts
- Multi-satellite visualization

## Install All Optional Dependencies

```bash
pip install fastspa[all]
```

## Development Installation

For contributing to FastSPA:

```bash
git clone https://docs.fastspa.net.git
cd fastspa
pip install -e ".[dev]"
```

## Verify Installation

```python
import fastspa
print(fastspa.__version__)
```
