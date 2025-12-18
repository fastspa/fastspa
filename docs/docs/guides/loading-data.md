---
sidebar_position: 4
---

# Loading Data

FastSPA provides multiple ways to load data from different sources.

## From Arrays (Primary Interface)

The most direct approach:

```python
import numpy as np
from fastspa import SPA

# Load your data
A = np.load("A_matrix.npy")
intensities = np.load("intensities.npy")
sectors = ["Agriculture", "Mining", "Manufacturing", ...]

spa = SPA(A, intensities, sectors)
```

## From DataFrames

```python
import pandas as pd
from fastspa import from_dataframe

# A-matrix as DataFrame (index and columns are sector names)
A_df = pd.read_csv("A_matrix.csv", index_col=0)

# Intensities with sector index
intensities_df = pd.read_csv("intensities.csv", index_col=0)

spa = from_dataframe(
    A_df,
    intensities_df,
    intensity_columns=["CO2", "Water", "Energy"]  # Optional: specify columns
)
```

## From CSV Files

```python
from fastspa import from_csv

spa = from_csv(
    "A_matrix.csv",           # A-matrix file
    "intensities.csv",        # Intensities file
    intensity_columns=["ghg", "water"],  # Which columns to use
    sectors_path="sectors.txt",  # Optional: sector names file
    mode="sector_specific"       # Analysis mode
)
```

## From Pre-computed Leontief Inverse

If you already have the Leontief inverse:

```python
from fastspa import from_leontief

# Skip L computation (saves time for large matrices)
spa = from_leontief(
    L_matrix,       # Pre-computed (I - A)^(-1)
    intensities,
    sectors,
    mode="system_wide"  # Default for Leontief-based analysis
)
```

## From IO Tables (Experimental)

Parse IO tables from Excel files:

```python
from fastspa import from_io_table

spa = from_io_table(
    "io_table.xlsx",
    sheet_name="Table 5",
    intensities=emissions_vector  # Optional: provide intensities
)
```

## Working with Sparse Matrices

FastSPA supports scipy sparse matrices:

```python
from scipy import sparse

# Load or create sparse A-matrix
A_sparse = sparse.load_npz("A_sparse.npz")

# Use directly
spa = SPA(A_sparse, intensities, sectors)
```

## Data Format Requirements

### A-Matrix (Technical Coefficients)

- Square matrix (n × n)
- Values typically between 0 and 1
- Rows: inputs to sectors
- Columns: sector production

### Intensities

- Vector of length n
- Direct intensity per unit output
- Units depend on your analysis (e.g., kg CO2-e / $)

### Sector Names

- List/tuple of n strings
- Must match A-matrix dimensions
- Used for readable output

## Example: Complete Data Loading

```python
import numpy as np
import pandas as pd
from fastspa import SPA

# Load A-matrix
A_df = pd.read_csv("A_matrix.csv", index_col=0)
A = A_df.values
sectors = list(A_df.index)

# Load multiple intensity vectors
emissions = pd.read_csv("emissions.csv", index_col=0)
intensities = {
    "carbon": emissions["CO2"].values,
    "water": emissions["Water"].values,
}

# Create SPA instance
spa = SPA(A, intensities, sectors, mode="sector_specific")

# Ready for analysis
paths = spa.analyze(sector="Manufacturing", depth=8)
```
