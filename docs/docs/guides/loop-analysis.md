---
title: Loop Detection & Circular Economy Analysis
---

In system-wide (Leontief) mode, circular flows are represented implicitly.
fastspa also supports explicit loop detection on the extracted path set.

## Detect loops in extracted paths

```python
paths = spa.analyze(sector=42, depth=8)

loops = paths.loop_analysis()

print("Loop share:", loops.loop_share)
print("Top cycles:", loops.top_cycles(10))
```

## Interpretation

- A *loop* is detected when a sector index repeats within a single path.
- The analysis summarises each looping path using the first repeated sector encountered.

## Feedback loop metrics

`loops.sector_participation` provides a per-sector measure of participation in looping paths.
This can be useful as a proxy for self-reinforcing circular dynamics.
