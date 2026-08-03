# Research roadmap

This living roadmap records open research directions. Executable work is
tracked in pgmcp under root epic
`extending-liblevenshtein-automaton-families-4bb97598`; pgmcp owns task status,
dependencies, and acceptance gates. This page explains why the remaining work
belongs in the library and directs readers to its design evidence.

## Shipped foundations

The following items appeared as proposals in the former version of this page
but are already implemented. They are listed here only to prevent duplicate
projects:

- ordered and priority query iterators;
- WallBreaker search and its split bound;
- byte and Unicode double-array tries, dynamic DAWGs, suffix automata, and
  persistent dictionary backends through `libdictenstein`;
- dictionary serialization surfaces and precomputed universal automata;
- SIMD helpers, parallel traversal features, property tests, fuzz targets, and
  corpus benchmarks.

Their current behavior is documented in the [algorithm index](../algorithms/README.md),
[developer guide](../developer-guide/README.md), and
[formal-verification index](../verification/README.md).

## Active automaton-family program

The program classifies a proposed measure before adding machinery:

| Class | Defining feature | Planned seam | Representative work |
|---|---|---|---|
| Alignment | A minimum over bounded consuming operations | `OperationSet` presets and repaired generalized acceptance | Hamming, indel, bounded-skip subsequence |
| Stateful | A continuation depends on bounded edit history | `PositionKind` and a specialized transition kernel | affine gaps, bounded true Damerau–Levenshtein |
| Cost algebra | Path costs combine by an operation other than addition | `CostMonoid` | discrete Fréchet bottleneck cost |
| Input domain | Transitions compare non-character observations or a language state | `ElasticKernel` or `LanguageProduct` | ERP, TWED, DTW, fuzzy regular languages |

The classification prevents two recurring mistakes: calling a restricted
optimal-string-alignment recurrence unrestricted Damerau–Levenshtein, and
adding a new automaton type when an exact `OperationSet` configuration suffices.
See the metric terminology in the [glossary](../GLOSSARY.md).

## Search and ranking surfaces

Three extensions remain useful because they can prune before materializing a
candidate:

- a stateful `PrefixPruner` visitor with balanced enter/leave events;
- lazy value-aware ranking by `(distance, score)` for mapped dictionaries;
- structural subsequence traversal, including the library half of an fzf-style
  matcher whose gain-valued scoring stays in downstream WFST crates.

A post-result `MatchFilter` is not planned: the existing generic filtered
iterators already monomorphize closures, and filtering after materialization
cannot reduce dictionary traversal. A minimum-distance range is API sugar only;
its lower bound cannot prune a subtree because extending an exact prefix may
raise the final distance.

## Experimental decision gates

Optimizations land only after a pre-registered benchmark passes its stated
decision rule:

- specialized Hamming/indel walkers must beat the honest standard-automaton
  candidate-generation baseline by at least `2×` on two of three dictionary
  sizes for budgets 1 and 2, while enumerating at least `4×` fewer edges;
- transition-kernel specialization must keep the Standard path below a `1.5%`
  mean regression, with a 95% confidence interval excluding `3%`;
- any float-engine unification must satisfy the same zero-cost gate and preserve
  all weighted-distance results.

Failed gates are results, not unfinished features. They belong in
`docs/scientific-ledger/` with the workload, environment, confidence interval,
and disposition.

## Deliberate boundaries

The library accepts a cost model only when non-negative prefix extension gives
a sound subtree lower bound. Scoring systems with positive gains, weight
pushing, closure (`star`), or division belong in the sibling WFST stack rather
than in `CostMonoid`. Multi-kind bounded-depth Dyck recognition is also outside
scope: its finite-state representation grows exponentially with stack depth;
the pushdown implementation belongs in `lling-llang`.

## How to propose another enhancement

1. State the semantic function and a reference algorithm.
2. Classify it by alignment, history, cost combination, and input domain.
3. Prove or falsify the pruning invariant before designing a public API.
4. Add a differential oracle, property invariants, resource guards, and a
   pre-registered performance decision rule.
5. Create the task under the pgmcp root epic above and link its scientific
   ledger record.

[← Research index](README.md)
