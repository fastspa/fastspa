---
sidebar_position: 2
---

# Factory Functions

Convenience functions for creating SPA instances from different data sources.

## from_leontief()

Create SPA from a pre-computed Leontief inverse.

```python
from_leontief(
    L: ArrayLike,
    direct_intensities: Union[ArrayLike, Dict[str, ArrayLike]],
    sectors: Optional[Sequence[str]] = None,
    mode: Literal["sector_specific", "system_wide"] = "system_wide",
) -> SPA
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | `ArrayLike` | - | Leontief inverse matrix (I - A)^(-1) |
| `direct_intensities` | `ArrayLike` or `Dict` | - | Direct intensity vector(s) |
| `sectors` | `Sequence[str]` | None | Optional sector names |
| `mode` | `str` | `"system_wide"` | Analysis mode |

### Example

```python
from fastspa import from_leontief

# If you already have L computed
spa = from_leontief(L_matrix, intensities, sectors)
```

## from_dataframe()

Create SPA from pandas DataFrames.

```python
from_dataframe(
    A_df: pd.DataFrame,
    intensities_df: pd.DataFrame,
    intensity_columns: Optional[List[str]] = None,
) -> SPA
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A_df` | `DataFrame` | A-matrix with sector names as index and columns |
| `intensities_df` | `DataFrame` | Intensities with sector index |
| `intensity_columns` | `List[str]` | Which columns to use (None = all numeric) |

### Example

```python
from fastspa import from_dataframe
import pandas as pd

A_df = pd.read_csv("A.csv", index_col=0)
intensities_df = pd.read_csv("intensities.csv", index_col=0)

spa = from_dataframe(A_df, intensities_df, intensity_columns=["CO2", "Water"])
```

## from_csv()

Create SPA from CSV files.

```python
from_csv(
    A_path: str,
    intensities_path: str,
    intensity_columns: Optional[List[str]] = None,
    sectors_path: Optional[str] = None,
    mode: Literal["sector_specific", "system_wide"] = "sector_specific",
) -> SPA
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A_path` | `str` | - | Path to A-matrix CSV |
| `intensities_path` | `str` | - | Path to intensities CSV |
| `intensity_columns` | `List[str]` | None | Columns to use (None = all numeric) |
| `sectors_path` | `str` | None | Path to sector names file |
| `mode` | `str` | `"sector_specific"` | Analysis mode |

### Example

```python
from fastspa import from_csv

spa = from_csv(
    "A_matrix.csv",
    "intensities.csv",
    intensity_columns=["ghg", "water"],
)
```

## from_io_table()

Create SPA from an IO table in XLSX format (experimental).

```python
from_io_table(
    xlsx_path: str,
    sheet_name: str = "Table 5",
    intensities: Optional[ArrayLike] = None,
) -> SPA
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xlsx_path` | `str` | - | Path to XLSX file |
| `sheet_name` | `str` | `"Table 5"` | Sheet containing IO table |
| `intensities` | `ArrayLike` | None | Direct intensities (random if not provided) |

### Notes

This function automatically:
- Extracts the largest continuous numeric matrix as transactions
- Identifies sector names from adjacent rows/columns
- Computes technical coefficients

### Example

```python
from fastspa import from_io_table

spa = from_io_table(
    "national_io_table.xlsx",
    sheet_name="Transactions",
    intensities=my_emissions_vector
)
```
