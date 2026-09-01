# TLA+ Specifications for liblevenshtein

This directory contains TLA+ specifications for verifying state machine properties of liblevenshtein components.

## Overview

TLA+ is used for model checking concurrent and stateful algorithms where exhaustive state space exploration is valuable. These specifications complement the Rocq/Coq proofs by verifying operational properties that are easier to express as state machines.

## Model-checking status

All registered models model-check with **no errors** under the bounds in their
`.cfg` files (`scripts/verify-formal.sh tla`). Captured transcripts live in
[`states/tlc-results-2026-05-26.txt`](states/) and
[`states/tlc-results-2026-05-27.txt`](states/) (the latter includes
`ValueYieldingQuery`). Coverage was tightened so the configs check the
properties the modules define:

- **OnlineScanner** now checks `NoMissedMatches` (reference-free form: no
  reachable final state is dropped without being recorded) and
  `MatchesRecordedCorrectly`, in addition to the structural invariants.
- **Subsumption** mirrors the three executable Rust branches and checks their
  distinct algebraic contracts: Standard and OSA are reflexive, merge/split is
  irreflexive, cross-algorithm and OSA normal/pending variants are separated,
  the coverage relation is antisymmetric and transitive, and its distinct-state
  restriction is a strict order. The lifecycle model also checks that every
  removed representative retains a current cover, removal is bounded, and an
  antichain is eventually reached.
- **ProductAutomaton** now uses a finite non-total witness NFA transition
  relation and checks `PatternPositionValid`.

One model-vs-implementation limitation remains, reconciled against the Rust
tests that cover the gap:

- **PriorityQuery** models an *idealized admissible* A*. The Rust
  `PriorityQueryIterator` actually uses an **inadmissible** heuristic
  (`query_len - max_consumed`) and only guarantees an approximate, fast-first-k
  ordering; the model's optimal/ordered guarantees are realized by
  `OrderedQueryIterator` (`Transducer::query_ordered`), as verified by
  `tests/proptest_priority_query.rs`.

`tlc_finite_state_exhaustiveness` (TLC checks the configured finite bounds, not
the unbounded algorithm) remains the one acknowledged TLA assumption in
`docs/verification/ASSUMPTIONS.tsv`.

## Specifications

### OperationSetDecode.tla

Exhaustively selects header fields, declared/available payload sizes, decoded
operation counts, and semantic-validation results. Acceptance is reachable
only after valid magic, version 1, zero flags, exact payload consumption,
configured resource limits, and semantic validation. Separate invariants keep
truncated and trailing payloads out of the accepted phase.

**Corresponds to:** `src/transducer/operation_set_binary.rs` and
`core/theories/Conformance/OperationSetSerialization.v`.

### OperationSetPortableDecode.tla

Models the two supported inner formats and optional gzip as a staged admission
lifecycle. For protobuf, the collection-allocation phase is unreachable until
wire validity, schema version, payload bytes, operation count, and pair count
pass preflight. For compressed input, acceptance requires a valid checksum,
both byte ceilings, and exact consumption of one gzip member. The model checks
that unsupported protobuf versions, over-limit counts, concatenated/trailing
compressed bytes, and invalid semantic models never reach `Done`.

**Corresponds to:** `src/transducer/operation_set_protobuf.rs`,
`src/transducer/operation_set_gzip.rs`, and
`core/theories/Conformance/OperationSetSerialization.v`.

### ClassAPresets.tla

Explores the Hamming, indel, and bounded-skip operation grids together. The
finite model checks operation progress and declared-consumption policy for
every preset, equal coordinates for Hamming, both indel length lower bounds,
and the exact directional equation $`\text{source}=\text{target}+\text{cost}`$
for bounded skip. With source, target, and cost capped at 4, TLC explores all
72 reachable states with no violation.

**Corresponds to:** `src/transducer/presets.rs`,
`src/transducer/operation_set.rs`, and the direct Class-A distance references.

### DamerauStreaming.tla

Specifies one finite unrestricted-Damerau macro from entry through zero or
more interior extensions to resolution. The $`k=3`$ model checks that a
pending continuation carries a positive delta, entry plus extensions preserve
`errors = delta + between`, every pending state has consumed a dictionary
unit, and resolution advances the query endpoint by exactly $`\delta+1`$
without changing the macro charge.

**Corresponds to:** `src/transducer/variants/damerau.rs` and the
Lowrance–Wagner refinement in
`docs/verification/damerau/theories/DamerauStreaming.v`.

### ElasticTrieSearch.tla

Specifies the generic `ElasticTransducer` range traversal with concrete tables
that make the prefix K1, interval-column K1, and K4 bounds executable. The
model contains a final root representing an indexed empty series, a subtree
rejected by the prefix gate before column construction, a separate
column-pruned subtree, a candidate-bound rejection, an exact-rescore rejection,
and two emitted non-root leaves. It checks both prune stages, absence of false
positives, root-terminal and full terminal completeness, termination, and the
`PrefixGatePrecedesColumn` order invariant without an abstract admissibility
assumption.

The transition system is cost-carrier agnostic: `ElasticTrieSearch` requires
only ordered lower bounds and exact rescoring. Discrete Fréchet uses its
bottleneck-compatible branch. Banded DTW adds the generic optional prefix
stage; its cumulative LB_Keogh premise is discharged by
`dtw/theories/Indexing/DtwProperties.v`, the Verus model, both SMT solvers, and
Rust properties. Kernels that return the default zero prefix bound retain the
previous transition behavior.

**Corresponds to:** `src/time_series/elastic/walker.rs` and the assumption-free
Rocq/Verus/SMT K1–K4 artifacts.

### ElasticSnapshotPublication.tla

Models the complete-snapshot protocol independently of the search recurrence.
The decode machine cannot reach a semantic or accepted phase without a matching
checksum. The publication machine creates an unreferenced staging generation,
seals it, atomically publishes the generation, and only then makes its identity
visible through the manifest; crash removes only staging state.

The finite `ElasticSnapshotPublication.cfg` instance checks `TypeOK`, checksum-
before-semantics, manifest-to-sealed-generation closure, and permanent rejection
after checksum failure. The model abstracts digests as identities and rename as
an atomic action; it does not prove SHA-256, filesystem, persistent-trie, or Rust
implementation correctness.

**Corresponds to:** `src/time_series/elastic/walker/snapshot.rs`,
`docs/verification/temporal_automata/theories/ElasticSnapshot.v`, and
`docs/design/complete-elastic-snapshots.md`.

### MsmTrieSearch.tla

Specifies the original MSM-specialized operational model. Its abstract
admissibility table is discharged by the MSM interval-column Rocq proofs. The
generic `ElasticTrieSearch` model complements it by making a representative
K1/K4 table executable and adding candidate-level pruning.

**Corresponds to:** `src/time_series/msm_kernel.rs` and
`src/time_series/msm_interval.rs`.

### OnlineScanner.tla

Specifies the online scanner that processes input character-by-character while tracking multiple concurrent matches.

**Key Properties:**
- `ActiveMatchesBounded`: Number of active matches stays within bounds
- `NoMissedMatches`: All valid matches are eventually recorded
- `PositionMonotonicity`: Input position only advances
- `ErrorBoundRespected`: All matches have errors <= max_errors

**Corresponds to:** `src/phonetic/online_scanner.rs`

### ProductAutomaton.tla

Specifies the product construction between a phonetic NFA and Levenshtein automaton.
The TLC model uses a finite witness NFA whose initial state consumes checked
characters into final states and whose final states self-loop. Arbitrary Rust
NFA product correctness and `min_distance` agreement remain cross-checked by
`tests/proptest_product_automaton.rs`.

**Key Properties:**
- `ProductCorrectness`: Acceptance iff NFA accepts AND cost <= max_cost
- `TransitionDeterminism`: Each (state, char) pair has well-defined transitions
- `CostMonotonicity`: Cost only increases along paths
- `StateSpaceBounded`: Product state space is polynomially bounded

**Corresponds to:** `src/phonetic/nfa/product.rs`

### Subsumption.tla

Verifies the bounded executable subsumption relation used to prune redundant
states. This is a coverage relation rather than one uniformly strict order:
Standard and OSA representatives cover themselves, while merge/split requires
a strict error improvement. `StrictDominates` adds representative inequality
for the removal lifecycle.

**Key Properties:**
- `AlgebraicInv`: algorithm-specific reflexivity, merge/split irreflexivity,
  variant separation, antisymmetry, and transitivity
- `RemovedHasCurrentCover`: every removed representative is still covered by
  a retained distinct representative
- `IterationBound`: each removal consumes one member of the finite sample
- `EventuallyAntichain`: weak fairness drives removal to a canonical frontier

**Corresponds to:** `src/transducer/position.rs`, `src/transducer/state.rs`, and
their float-weighted twins. The unbounded arithmetic and insertion contracts
are proved in `core/theories/Conformance/RustSubsumption.v`; TLC exhausts only
the finite constants in `Subsumption.cfg`.

### PriorityQuery.tla

Specifies the A* priority queue search for efficient fuzzy matching.

**Key Properties:**
- `HeapInvariant`: Priority queue maintains min-heap property
- `AdmissibleHeuristic`: h(n) <= actual_cost(n) for all nodes
- `Optimality`: First match found has minimum edit distance
- `Completeness`: All matches within bound eventually found

**Corresponds to:** `src/transducer/priority_query.rs`

### ValueYieldingQuery.tla

Specifies the value-yielding transducer query (`Transducer::query_values`): a BFS over the
dictionary×automaton intersection that yields `(term, distance, value)` for each match within the
edit-distance threshold, reading the value during traversal and skipping valueless finals. The model
runs over a concrete dictionary that includes valued and valueless finals, in-range and out-of-range
finals, and a shared-term (dedup) case; the nondeterministic processing order makes TLC verify the
invariants are order-independent.

**Key Properties:**
- `ValueCorrectness`: every yielded value equals the dictionary's stored value for that term
- `Soundness`: every yielded distance is within the threshold
- `NoValuelessYielded`: valueless finals are never emitted
- `DedupInv`: each term is yielded at most once
- `CompletenessInv`: every processed in-range valued final has its term in the results
- `EventuallyTerminates`: the traversal terminates

**Corresponds to:** `src/transducer/value_filtered_query.rs`, `src/transducer/mod.rs`
(cross-validated on the real dictionary by `tests/proptest_value_yielding_query.rs`).

## Running TLC Model Checker

### Prerequisites

Install TLA+ Toolbox or use command-line TLC:
```bash
# Download TLC
wget https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

# Or use the TLA+ VSCode extension
```

### Model Checking Commands

```bash
# Check OnlineScanner
java -jar tla2tools.jar -config OnlineScanner.cfg OnlineScanner.tla

# Check with specific bounds
java -jar tla2tools.jar \
  -config OnlineScanner.cfg \
  -workers 8 \
  OnlineScanner.tla

# Check Subsumption (fast - small state space)
java -jar tla2tools.jar Subsumption.tla

# Check ProductAutomaton
java -jar tla2tools.jar -config ProductAutomaton.cfg ProductAutomaton.tla
```

### Recommended Model Checking Bounds

| Specification | Constants | Estimated States |
|---------------|-----------|------------------|
| OnlineScanner | MAX_ERRORS=2, PATTERN_LENGTH=3, INPUT_LENGTH=5 | ~10^4 |
| ProductAutomaton | MAX_COST=2, INPUT_LENGTH=4 | ~10^5 |
| Subsumption | MAX_POSITION=2, MAX_ERRORS=1, QUERY_LENGTH=2 | 1,024 |
| PriorityQuery | MAX_COST=2, WORD_LENGTH=3, DICT_SIZE=4 | ~10^4 |
| ValueYieldingQuery | MaxDistance=1, NoVal=999 (7-node dictionary) | ~30 |
| ElasticTrieSearch | Tau=3 (9-node dictionary with prefix and column prunes) | 69 |
| VariantDispatch | three algorithms × three positions | small finite model |
| DamerauStreaming | K=3 | 13 generated states, depth 5 |
| MsmTrieSearch | Tau=3 (7-node dictionary) | small finite model |

## Configuration Files

Each specification can use a `.cfg` file for TLC configuration:

```
CONSTANTS
    MAX_ERRORS = 2
    MAX_ACTIVE_MATCHES = 50
    PATTERN_LENGTH = 4
    INPUT_LENGTH = 6
    ALPHABET = {"a", "b", "c"}

SPECIFICATION Spec

INVARIANTS
    TypeInvariant
    ActiveMatchesBounded
    ErrorBoundValid

PROPERTIES
    EventuallyComplete
```

## Relationship to Rocq Proofs

| Property Type | TLA+ | Rocq |
|---------------|------|------|
| State machine invariants | Primary | Secondary |
| Temporal properties | Primary | Via coinduction |
| Mathematical proofs | Secondary | Primary |
| Metric properties | Not applicable | Primary |
| Inductive properties | Model checking | Proof by induction |

## Adding New Specifications

1. Create a new `.tla` file following the module structure
2. Define CONSTANTS, VARIABLES, and types
3. Define Init and Next relations
4. Add INVARIANTS for safety properties
5. Add PROPERTIES for liveness properties
6. Create a `.cfg` file with appropriate bounds
7. Test with small bounds first, then increase

## Notes

- TLA+ specifications are abstractions; they capture essential behavior but may simplify implementation details
- Model checking is exhaustive within bounds; properties hold for all checked states
- Use small bounds for initial development, larger bounds for final verification
- Consider using symmetry reduction for symmetric state spaces
