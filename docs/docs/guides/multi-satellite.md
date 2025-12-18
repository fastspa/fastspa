---
sidebar_position: 1
---

# Multi-Satellite Analysis

Analyze multiple environmental flows (satellites) simultaneously.

## Setting Up Multiple Satellites

Pass a dictionary of intensity vectors to the SPA constructor:

```python
from fastspa import SPA

spa = SPA(
    A_matrix,
    {
        "ghg": ghg_intensities,      # kg CO2-e per unit output
        "water": water_intensities,  # m³ per unit output
        "energy": energy_intensities # MJ per unit output
    },
    sectors=sector_names
)
```

## Analyzing All Satellites

When you analyze without specifying a satellite, you get an `SPAResult` containing all satellites:

```python
result = spa.analyze(sector="Manufacturing", depth=8)

# SPAResult contains results for all satellites
print(result.satellites)  # ['ghg', 'water', 'energy']

# Access individual satellite results
ghg_paths = result["ghg"]
water_paths = result["water"]

# Each is a PathCollection
print(ghg_paths.summary())
```

## Analyzing a Specific Satellite

To analyze only one satellite:

```python
# Specify the satellite
water_paths = spa.analyze(
    sector="Manufacturing",
    depth=8,
    satellite="water"
)

# Returns a PathCollection directly (not SPAResult)
print(water_paths.summary())
```

## Combining Results

Export all satellites to a single DataFrame:

```python
# Combined DataFrame with a 'satellite' column
df = result.to_dataframe()
print(df.columns)
# ['contribution', 'contribution_pct', 'direct_intensity',
#  'cumulative_weight', 'depth', 'stage_0', 'stage_0_name', ..., 'satellite']

# Filter by satellite
ghg_df = df[df['satellite'] == 'ghg']
```

## Example: Comparing Environmental Impacts

```python
from fastspa import SPA

spa = SPA(A, {
    "carbon": carbon_intensities,
    "water": water_intensities,
}, sectors=sectors)

result = spa.analyze(sector="Electricity", depth=8)

# Compare top emission sources
for sat_name in result.satellites:
    paths = result[sat_name]
    print(f"\n{sat_name.upper()} - Top 5 Sources:")
    hotspots = paths.aggregate_by_sector()
    top_5 = sorted(hotspots.items(), key=lambda x: x[1], reverse=True)[:5]
    for idx, contrib in top_5:
        sector_name = sectors[idx]
        print(f"  {sector_name}: {contrib:.2%}")
```

## Iterating Over Results

```python
# Iterate over satellite names
for sat_name in result:
    print(f"Satellite: {sat_name}")

# Iterate over name-paths pairs
for sat_name, paths in result.items():
    print(f"{sat_name}: {len(paths)} paths, {paths.coverage:.1%} coverage")
```
