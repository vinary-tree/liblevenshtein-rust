# Subsumption in liblevenshtein-java and liblevenshtein-rust

## Status and scope

This document replaces the earlier unmeasured claim that the Java and Rust
implementations perform equivalent subsumption work. That claim was false for
the benchmarked path.

The measurements below were collected on 2026-08-14 from:

- the published Java artifact `com.github.universal-automata:liblevenshtein:3.0.0`;
- the current Rust `DynamicDawg<u8>` query implementation;
- the committed 79,343-term ASCII dictionary and 1,000-query `std-d2` set;
- standard Levenshtein distance with maximum distance two; and
- one CPU in the same AMD Ryzen 9 7950X3D cache domain.

Subsumption is the dominance relation used to remove a position whose future
language is already represented by another position. A *subsumption
comparison* is one invocation of that relation on a pair of positions. It is
not merely a loop iteration or an equality test.

## Measured result

All arms produced exactly 18,514 matches, 82,131 UTF-8 term bytes, and a
distance sum of 36,201. The Rust checksum was
`6be8b7274d8277d8`. Therefore the work counts compare semantically equivalent
passes.

| Implementation or experiment | Subsumption comparisons | Median pass time | Decision |
|---|---:|---:|---|
| Published Java 3.0.0 | 0 | not used for the Rust treatment decision | Measured reference |
| Rust retained control | 5,576,499 | experiment-local controls below | Production baseline |
| H-O20: cost-order guard | 5,484,525 | 330.515 ms → 314.907 ms | Reverted: 4.722% missed the pre-registered 5% gate |
| H-O21: batch cost-order normalization | 5,313,364 | 317.033 ms → 352.388 ms | Rejected: 11.15% slower |

Medians from different experiment rows must not be compared with each other.
Each arrow is a separately built, pinned, 51-sample control/treatment pair.

The direct answer to H-O6 is therefore **no**: Rust does not match Java's
pairwise subsumption work on this workload. Rust dispatches millions of
comparisons that Java does not execute. The experiments also show that
comparison count alone is not a sufficient optimization objective: reducing
it by sorting and moving tiny states made the end-to-end query slower.

## Measurement method

### Java

The published JAR does not expose work counters. The probe places a
measurement-only class ahead of the JAR on the Java class path. That class is
a control-flow transcription of the released `UnsubsumeFunction` bytecode and
adds counters immediately around `SubsumesFunction.at`. The rest of the query,
including state generation, merging, candidate traversal, and the dictionary,
comes from the unmodified published artifact.

The reproducible entry point is:

```bash
benchmarks/causal/run-legacy-subsumption-probe.sh \
  benchmarks/cross-language/workload/dictionary.txt \
  benchmarks/cross-language/workload/queries/std-d2.txt \
  standard 2 /tmp/java-subsumption
```

The probe reported:

| Java counter | Count |
|---|---:|
| `unsubsume_calls` | 6,996,242 |
| `outer_positions` | 3,443,733 |
| `subsumption_comparisons` | 0 |
| `removed_positions` | 0 |

The call count is important: zero comparisons are not evidence that the
shadow class was bypassed. Its `at` method ran once for every attempted
dictionary-edge transition. Exact result aggregates independently validate
the query.

### Rust

The `perf-instrumentation` feature increments a counter immediately before
each monomorphized `AutomatonVariant::subsumes` dispatch. The reproducible
entry point is:

```bash
cargo build --release --features perf-instrumentation \
  --bin causal_query_profile
taskset -c 3 target/release/causal_query_profile \
  --dictionary benchmarks/cross-language/workload/dictionary.txt \
  --queries benchmarks/cross-language/workload/queries/std-d2.txt \
  --domain byte --constructor from_sorted_terms \
  --algorithm standard --max-distance 2 --passes 1
```

The retained Rust path reported 4,133,255 state-insert attempts, 3,375,478
retained inserts, and 5,576,499 subsumption comparisons. State-copy traffic is
zero after H-O16, so these counts are not aliases for the formerly dominant
copy path.

## Why the old analysis was wrong

The earlier document described a `BTreeSet` ordered by error count and a Rust
`take_while` early exit. Current production Rust uses
`SmallVec<[Position; 8]>`, and `Position::cmp` orders by term index first, then
error count and continuation metadata. `State::insert_with` scans the retained
antichain twice: existing positions may dominate the new position, then the
new position may remove existing positions.

The earlier Java description was incomplete as well. Released bytecode calls
`UnsubsumeFunction.at` before the final `State.sort` in
`StateTransitionFunction.of`. The ordering seen by unsubsumption is therefore
the order established by the merge step, not the later
`StandardPositionComparator` sort. On this workload, the early scan consumes
all remaining positions without reaching a higher-error suffix, so it never
calls the dominance predicate.

The relevant control flows are:

```text
Java 3.0.0
  generate per-position successor states
  → merge into one linked state
  → run unsubsumption scan
  → sort the retained state

Rust retained path
  for each generated successor
    → test it against the current SmallVec antichain
    → remove representatives it dominates
    → insert it in term-index-first order
```

For a state containing `n` raw candidates and a retained antichain of size
`k`, Rust's online path performs work proportional to roughly
$`\mathcal{O}(n k)`$. Here `k` is small, so contiguous scans are cheap even
when their aggregate count is large.

## Optimization experiments

### H-O20: necessary cost-order guard

Every supported dominance relation requires the potential subsumer's
accumulated cost to be no greater than the target's cost. H-O20 checked this
condition before variant dispatch in both directions.

```text
if potential_subsumer.cost > target.cost
  skip variant-specific subsumption
else
  dispatch variant-specific subsumption
```

This was semantically exact and statistically faster, but it eliminated only
91,974 comparisons, or 1.65%. The pre-registered production gate required at
least a 5% median reduction; the measured reduction was 4.722%. The treatment
was therefore reverted. pgmcp experiment 196 retains the samples and records
that its generic statistical criterion passed while the explicit engineering
threshold did not.

### H-O21: Java-like batch normalization

H-O21 accumulated all raw successor positions, stably ordered them by
accumulated cost, then computed one minimal antichain before restoring the
public term-index-first order.

```text
generate every raw successor
stable-sort successors by accumulated cost
for each successor from cheapest to most expensive
  discard it if a retained position dominates it
  otherwise retain it and remove any equal-cost position it dominates
sort the final antichain by Position order
```

The treatment reduced comparisons by 263,135, or 4.72%, but regressed the
51-sample median by 11.15%. Stable sorting, moving the `SmallVec`, and restoring
canonical order cost more than comparisons over the tiny retained antichain.
pgmcp experiment 197 rejected the treatment, and it was reverted.

## Retained design and next hypothesis

The retained implementation remains the monomorphized online `SmallVec`
antichain. This is not because its work matches Java—it does not—but because
the two direct attempts to remove that work failed their end-to-end production
gates.

The next principled optimization should bypass the position antichain for the
common standard-distance-zero-to-two case instead of making its pairwise loop
more elaborate. A generated parametric automaton can encode the normalized
state as a compact state identifier and table transition:

```text
(state_id, characteristic_mask) → next_state_id
```

That design removes successor generation, online insertion, and subsumption
together. The generic positional engine remains the correctness fallback for
larger distances, restricted substitution policies, and extended algorithms.
Any such specialization requires exact cross-algorithm and cross-domain gates;
comparison-count reduction by itself is not an acceptance criterion.

## Reproducibility artifacts

- Java shadow counter:
  `benchmarks/causal/java-shadow/com/github/liblevenshtein/transducer/UnsubsumeFunction.java`
- Java workload driver: `benchmarks/causal/LegacySubsumptionProbe.java`
- Java runner: `benchmarks/causal/run-legacy-subsumption-probe.sh`
- Rust counter driver: `src/bin/causal_query_profile.rs`
- Accepted/rejected experiment ledger: pgmcp experiments 196 and 197
