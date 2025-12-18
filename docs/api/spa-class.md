---
sidebar_position: 1
---

# SPA Class

The main class for Structural Path Analysis.

## Constructor

```python
SPA(
    A: ArrayLike,
    intensities: Union[ArrayLike, Dict[str, ArrayLike], SatelliteWithConcordance],
    sectors: Optional[Sequence[str]] = None,
    mode: Literal["sector_specific", "system_wide"] = "sector_specific",
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `ArrayLike` | Technical coefficients matrix (n×n). Can be dense NumPy array or scipy sparse matrix. |
| `intensities` | `ArrayLike` or `Dict` | Direct intensities. Either a 1D array (single satellite) or dict mapping names to arrays. |
| `sectors` | `Sequence[str]` | Optional list of sector names for readable output. |
| `mode` | `str` | Analysis mode: `"sector_specific"` (default) or `"system_wide"`. |

### Example

```python
from fastspa import SPA

spa = SPA(
    A_matrix,
    {"ghg": ghg_vector, "water": water_vector},
    sectors=["Agriculture", "Mining", "Manufacturing"],
    mode="system_wide",
)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_sectors` | `int` | Number of sectors in the system |
| `mode` | `str` | Analysis mode (`"sector_specific"` or `"system_wide"`) |
| `satellites` | `List[str]` | Names of available satellites |
| `sectors` | `Tuple[str, ...]` | Sector names (if provided) |
| `A` | `NDArray` | Technical coefficients matrix |
| `L` | `NDArray` | Leontief inverse (computed lazily) |

## Methods

### analyze()

Run structural path analysis for a target sector.

```python
analyze(
    sector: Union[int, str],
    depth: int = 8,
    *,
    threshold: float = 0.001,
    threshold_type: Literal["percentage", "absolute"] = "percentage",
    satellite: Optional[str] = None,
    max_paths: Optional[int] = None,
    include_stage_0: bool = True,
) -> Union[PathCollection, SPAResult]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sector` | `int` or `str` | - | Target sector (1-indexed integer or name) |
| `depth` | `int` | 8 | Maximum supply chain depth to explore |
| `threshold` | `float` | 0.001 | Minimum contribution threshold |
| `threshold_type` | `str` | `"percentage"` | Type of threshold: `"percentage"` or `"absolute"` |
| `satellite` | `str` | None | Specific satellite to analyze (None = all) |
| `max_paths` | `int` | None | Maximum number of paths to return |
| `include_stage_0` | `bool` | True | Include Stage 0 (direct requirements) |

#### Returns

- `PathCollection` if single satellite
- `SPAResult` if multiple satellites

#### Example

```python
# Basic analysis
paths = spa.analyze(sector=42, depth=8)

# By sector name
paths = spa.analyze(sector="Electricity", depth=8)

# With custom threshold
paths = spa.analyze(sector=42, threshold=0.0001, threshold_type="absolute")

# Limit results
paths = spa.analyze(sector=42, max_paths=100)
```

### stream()

Stream paths as they are discovered (memory-efficient).

```python
stream(
    sector: Union[int, str],
    depth: int = 8,
    *,
    threshold: float = 0.001,
    threshold_type: Literal["percentage", "absolute"] = "percentage",
    satellite: Optional[str] = None,
) -> Iterator[Path]
```

Returns paths in discovery order (not sorted by contribution).

### analyze_many()

Analyze multiple sectors at once.

```python
analyze_many(
    sectors: Sequence[Union[int, str]],
    depth: int = 8,
    **kwargs,
) -> Dict[Union[int, str], Union[PathCollection, SPAResult]]
```

### hotspots()

Identify emission hotspots (sectors contributing most to total).

```python
hotspots(
    sector: Union[int, str],
    depth: int = 8,
    *,
    satellite: Optional[str] = None,
    top_n: int = 10,
) -> List[Tuple[int, float, str]]
```

Returns list of `(sector_index, contribution, sector_name)` tuples.

### compare_sectors()

Compare SPA results across multiple sectors.

```python
compare_sectors(
    sectors: Sequence[Union[int, str]],
    depth: int = 8,
    *,
    satellite: Optional[str] = None,
) -> pd.DataFrame
```

Returns DataFrame with columns: `sector`, `sector_name`, `total_intensity`, `num_paths`, `coverage`, `top_path_contribution`.

### total_intensity()

Get total intensities for a satellite.

```python
total_intensity(satellite: Optional[str] = None) -> NDArray
```

Returns array of total intensities for each sector.

### sensitivity()

Elasticities of a target sector's total intensity to each direct intensity input.

```python
sensitivity(sector: Union[int, str], *, satellite: Optional[str] = None) -> Dict[int, float]
```

### monte_carlo()

Monte Carlo (or Latin Hypercube) uncertainty propagation.

```python
monte_carlo(
    sector: Union[int, str],
    depth: int = 8,
    *,
    n_samples: int = 200,
    intensity_cv: Union[float, ArrayLike] = 0.1,
    sampling: Literal["random", "lhs"] = "random",
    distribution: Literal["lognormal", "normal"] = "lognormal",
    seed: Optional[int] = None,
) -> MonteCarloResult
```

See: `guides/uncertainty` for interpretation and examples.

### consequential()

Consequential (perturbation / policy scenario) analysis.

```python
consequential(
    sector: Union[int, str],
    depth: int = 8,
    *,
    satellite: Optional[str] = None,
    intensity_multiplier: Optional[ArrayLike] = None,
    intensity_delta: Optional[ArrayLike] = None,
    A_multiplier: Optional[ArrayLike] = None,
    A_delta: Optional[ArrayLike] = None,
) -> ConsequentialResult
```

### analyze_time_series()

Multi-year / dynamic SPA.

```python
analyze_time_series(
    series: Mapping[Any, Union[SPA, Tuple[ArrayLike, Any]]],
    sector: Union[int, str],
    depth: int = 8,
    *,
    satellite: Optional[str] = None,
) -> TemporalSPAResult
```
