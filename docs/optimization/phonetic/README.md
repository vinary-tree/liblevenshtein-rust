# Phonetic Rules Optimization — Records

Append-only optimization journal for the phonetic-rules hot path: a phased
investigation (baseline → code analysis → optimization → iteration/cache/slice
analysis → algorithmic position-skipping) with profiling data, hypotheses, and
results. The `flamegraphs/` subdirectory holds the supporting `perf` flamegraph
artifacts. Preserved as a dated experimental record.

## Records

- [00-investigation-log.md](00-investigation-log.md) — Phonetic Rules Performance Investigation Log
- [01-baseline-investigation.md](01-baseline-investigation.md) — Phase 1: Baseline Investigation
- [02-code-analysis.md](02-code-analysis.md) — Phase 2: Code Analysis — Allocation Patterns
- [03-optimization-results.md](03-optimization-results.md) — Phase 3: Optimization Results — `can_apply_at()` Helper
- [04-iteration-analysis.md](04-iteration-analysis.md) — Phase 4: Iteration Count Analysis
- [05-h3-cache-analysis.md](05-h3-cache-analysis.md) — Phase 5: H3 Cache Inefficiency Analysis
- [06-h4-slice-analysis.md](06-h4-slice-analysis.md) — Phase 6: H4 Slice Copying Overhead Analysis
- [07-algorithmic-optimization-analysis.md](07-algorithmic-optimization-analysis.md) — Phase 7: Algorithmic Optimization Analysis (Position Skipping)
- [flamegraphs/](flamegraphs/) — Supporting `perf` flamegraph artifacts

**Status: Historical — append-only scientific record; indexed, not edited.**

[← Documentation Index](../../README.md)
