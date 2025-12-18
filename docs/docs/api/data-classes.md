---
sidebar_position: 3
---

# Data Classes

Core data structures used in FastSPA.

## Path

A single supply chain pathway.

```python
@dataclass(frozen=True, slots=True)
class Path:
    nodes: Tuple[int, ...]           # Sector indices (root to leaf)
    contribution: float               # Fraction of total intensity
    direct_intensity: float           # Direct intensity at leaf node
    cumulative_weight: float          # Product of A-matrix coefficients
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `nodes` | `Tuple[int, ...]` | Sector indices in the path |
| `sectors` | `Tuple[str, ...]` | Sector names (if available) |
| `contribution` | `float` | Fraction of total intensity |
| `direct_intensity` | `float` | Intensity at emission source |
| `cumulative_weight` | `float` | Product of path coefficients |
| `depth` | `int` | Number of upstream stages |
| `root` | `int` | Target sector (first node) |
| `leaf` | `int` | Final upstream sector |

### Example

```python
path = paths[0]
print(path.nodes)        # (42, 15, 3)
print(path.sectors)      # ('Electricity', 'Coal', 'Mining')
print(path.contribution) # 0.0523
print(path.depth)        # 2
print(path)              # "  5.23% | Electricity → Coal → Mining"
```

## PathCollection

A collection of paths with analysis methods.

```python
@dataclass
class PathCollection:
    paths: List[Path]
    target_sector: int
    total_intensity: float
    satellite_name: str = "intensity"
    metadata: Optional[AnalysisMetadata] = None
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `paths` | `List[Path]` | List of Path objects |
| `target_sector` | `int` | Index of analyzed sector |
| `total_intensity` | `float` | Total intensity for target |
| `satellite_name` | `str` | Name of satellite analyzed |
| `metadata` | `AnalysisMetadata` | Analysis parameters |
| `coverage` | `float` | Sum of all contributions |

### Methods

See [PathCollection](../concepts/pathcollection) for full documentation.

## SPAResult

Results for multiple satellites.

```python
@dataclass
class SPAResult:
    results: Dict[str, PathCollection]
    target_sector: int
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `satellites` | `List[str]` | Names of satellites |

### Methods

| Method | Description |
|--------|-------------|
| `__getitem__(key)` | Get PathCollection by satellite name |
| `__iter__()` | Iterate over satellite names |
| `items()` | Get (name, PathCollection) pairs |
| `to_dataframe()` | Combine all to DataFrame |

### Example

```python
result = spa.analyze(sector=42, depth=8)

# Access by name
ghg_paths = result["ghg"]

# Iterate
for name in result:
    print(f"{name}: {len(result[name])} paths")

# Combined DataFrame
df = result.to_dataframe()
```

## AnalysisMetadata

Metadata for reproducibility.

```python
@dataclass(frozen=True)
class AnalysisMetadata:
    analysis_date: datetime
    sector: Union[int, str]
    sector_name: Optional[str]
    threshold: float
    threshold_type: str
    max_depth: int
    total_intensity: float
    coverage: float
    n_paths: int
    mode: str
    satellite: str
    n_sectors: int
```

### Methods

| Method | Description |
|--------|-------------|
| `to_dict()` | Convert to dictionary |

### Example

```python
meta = paths.metadata
print(f"Analyzed: {meta.sector_name}")
print(f"Mode: {meta.mode}")
print(f"Coverage: {meta.coverage:.1%}")
print(f"Date: {meta.analysis_date}")
```

## SectorConcordance

Maps between sector classifications.

```python
@dataclass
class SectorConcordance:
    matrix: NDArray                    # Allocation matrix
    source_sectors: Sequence[str]      # Source classification
    target_sectors: Sequence[str]      # Target classification
    metadata: Optional[ConcordanceMetadata] = None
```

### Class Methods

| Method | Description |
|--------|-------------|
| `from_supply_use_table()` | Create from supply-use data |

### Instance Methods

| Method | Description |
|--------|-------------|
| `allocate(source_values)` | Allocate values to target sectors |

## SatelliteWithConcordance

Satellite data with sector mapping.

```python
@dataclass
class SatelliteWithConcordance:
    intensities: NDArray              # Source classification intensities
    concordance: SectorConcordance    # Mapping to target
    name: str                         # Satellite name
    unit: Optional[str] = None        # Unit description
```

### Methods

| Method | Description |
|--------|-------------|
| `get_intensities_target()` | Get intensities in target classification |

### Example

```python
from fastspa import SatelliteWithConcordance, SectorConcordance

satellite = SatelliteWithConcordance(
    intensities=source_intensities,
    concordance=concordance,
    name="carbon",
    unit="kg CO2-e / AUD"
)

# Use with SPA
spa = SPA(A, satellite, sectors=concordance.target_sectors)
```
