# Distance Calculation — Algorithms

**Direct edit-distance algorithms (Layer 4), independent of the automaton path.**

These documents cover *direct* string-to-string distance computation: the
classical dynamic-programming and memoized-recursive methods used when you have
two strings to compare without a dictionary. They serve direct comparison,
validation of automaton results, and benchmarking. For dictionary-driven
matching, see [Layer 2](../../02-levenshtein-automata/) instead.

## Algorithms

| Document | Purpose |
|----------|---------|
| [iterative-dp.md](iterative-dp.md) | Iterative dynamic-programming edit distance with the 2-row optimization: `𝒪(mn)` time, `𝒪(min(m,n))` space. |
| [recursive-memoization.md](recursive-memoization.md) | Recursive edit distance with memoization (C++-style caching) and its trade-offs versus the iterative form. |
| [optimizations.md](optimizations.md) | Distance-calculation optimizations: common prefix/suffix stripping, early termination, and bounded-`k` cutoffs. |

**Status: Living reference.**

[← Documentation Index](../../../README.md)
