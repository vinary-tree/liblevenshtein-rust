# Lexicographic kNN stopping model report

The finite [model](tla/LexicographicKnn.tla) checks original coverage,
top-k retention, and ordered completion under nondeterministic region
splitting, exact verification, complete or unknown tie summaries, and
lexicographic pruning. Candidate names differ from tie rank, so encounter
order cannot stand in for the contract's tie key.

From the repository root, run the configurations separately:

```sh
tlc -config docs/verification/tla/LexicographicKnn.cfg docs/verification/tla/LexicographicKnn.tla
tlc -config docs/verification/tla/LexicographicKnnCostOnlyEquality.cfg docs/verification/tla/LexicographicKnn.tla
```

With TLC 2.19 on 2026-09-29, the clean configuration passed after 456
distinct states at depth 11. The checked-in cost-only equality mutant exited
12 and violated `NoLostTopK` at depth 6. Its trace verifies `a` and `b`,
then discards the region containing `c` and `d` because that region's cost
floor equals the current worst verified cost. Candidate `c` has a better tie
key than `a`, so it belongs to the true top two and the discard is unsound.

The [Rocq order proof](temporal_automata/theories/LazyProductOperations.v)
establishes generic lexicographic pruning and top-k preservation under
admissible cost and tie bounds. The TLC instance is a finite negative control
for an omitted tie premise. Neither establishes machine-bound admissibility,
captured-original coverage, Rust heap order, or production correspondence.
