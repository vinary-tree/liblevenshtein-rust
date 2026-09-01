# Research and Analysis

Performance research, optimization analysis, and experimental features.

## Early planning docs (archived)

> An early (2025-11-11) never-executed planning pair —
> [`RESEARCH_INITIATIVES.md`](../archive/research/RESEARCH_INITIATIVES.md) and
> [`RESEARCH_TRACKING.md`](../archive/research/RESEARCH_TRACKING.md) — is preserved under
> `docs/archive/research/`.
> Its premise (an `OptimizedDawg` deprecation analysis) has since been resolved: `OptimizedDawg`
> was removed from the source tree, so the proposals it framed are superseded.

## Contents

### [Future Enhancements](future-enhancements.md)
Exploration of potential future enhancements:
- Proposed features
- Technical feasibility
- Performance impact analysis
- Implementation considerations

### Research Areas

#### [Lowrance–Wagner unrestricted Damerau distance](lowrance-wagner/PAPER_SUMMARY.md)
Scientific and engineering reading of the 1975 recurrence:
- last-occurrence transposition macro
- budget-bounded streaming refinement
- cost-equivalence equation
- implementation, proof, and resource obligations

#### [Edit distance with Real Penalty](erp/PAPER_SUMMARY.md)
Scientific analysis of Chen and Ng's ERP measure:
- exact recurrence and source examples
- raw-sequence pseudometric versus $`g`$-quotient identity
- gap-mass and interval lower-bound derivations
- implementation, test, and formal-proof mapping

#### [Discrete Fréchet distance](frechet/PAPER_SUMMARY.md)
Scientific analysis of Eiter and Mannila's coupling distance:
- coupling semantics and the complete Table 1 recurrence
- raw vectors versus identity modulo consecutive-duplicate collapse
- pinned-endpoint, one-sided-Hausdorff, and interval-bound derivations
- bottleneck-monoid, implementation, test, and formal-proof mapping

#### [Time Warp Edit Distance](twed/PAPER_SUMMARY.md)
Scientific analysis of Marteau's timestamp-aware segment edit distance:
- unit-spaced recurrence and accumulated empty boundaries
- carry-aware interval match/deletion minima and length lower bound
- strict metric-domain correction and zero-parameter counterexample
- API, testing, security, and heterogeneous formal-proof mapping

#### [SIMD Optimization](simd-optimization/README.md)
Research and implementation of SIMD optimizations:
- Vectorization opportunities
- Performance measurements
- Implementation phases
- Results and conclusions

#### [Distance Optimization](distance-optimization/README.md)
Levenshtein distance computation optimization research:
- Algorithm improvements
- Implementation techniques
- Benchmark results
- Roadmap for future work

#### [Comparative Analysis](comparative-analysis/README.md)
Comparative analysis of different approaches and implementations:
- Algorithm comparisons
- Backend performance analysis
- Trade-off evaluations

#### [Eviction Wrapper](eviction-wrapper/README.md)
Cache eviction strategy research and design:
- Eviction policy evaluation
- Implementation architecture
- Performance characteristics

## Related Documentation

- [Design Documents](../design/README.md) - Specifications derived from research
- [Benchmarks](../benchmarks/README.md) - Performance measurement results
- [Developer Guide](../developer-guide/performance.md) - Performance optimization guide
