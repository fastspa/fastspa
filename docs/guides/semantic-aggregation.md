---
title: Semantic Aggregation (Custom & Hierarchical)
---

Structural paths are often too granular to communicate directly.
fastspa supports post-processing aggregation of results into semantic groups.

## Custom sector grouping

Aggregate by leaf (emission source sector) into user-defined groups:

```python
paths = spa.analyze(sector=42, depth=8)

mapping = {
    0: "Primary",
    1: "Secondary",
    2: "Services",
}

by_group = paths.semantic_aggregate(mapping)
```

If sector names were provided, mappings can also be name-based:

```python
mapping = {
    "Agriculture": "Primary",
    "Manufacturing": "Secondary",
    "Services": "Services",
}
by_group = paths.semantic_aggregate(mapping)
```

## Hierarchical aggregation

Provide multiple mappings to build multi-level group keys:

```python
level_1 = {0: "Goods", 1: "Goods", 2: "Services"}
level_2 = {0: "Agriculture", 1: "Manufacturing", 2: "Services"}

hier = paths.hierarchical_aggregate([level_1, level_2])
# keys look like: ("Goods", "Manufacturing")
```

## Notes

- Aggregation is a *post-processing* step: it does not change SPA traversal.
- By default, aggregation uses the **leaf sector**; you can also aggregate by other stages.
