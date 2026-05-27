# TLA+ Specifications for liblevenshtein

This directory contains TLA+ specifications for verifying state machine properties of liblevenshtein components.

## Overview

TLA+ is used for model checking concurrent and stateful algorithms where exhaustive state space exploration is valuable. These specifications complement the Rocq/Coq proofs by verifying operational properties that are easier to express as state machines.

## Model-checking status

All five models model-check with **no errors** under the bounds in their `.cfg`
files (`scripts/verify-formal.sh tla`). Captured transcripts live in
[`states/tlc-results-2026-05-26.txt`](states/) and
[`states/tlc-results-2026-05-27.txt`](states/) (the latter includes
`ValueYieldingQuery`). Coverage was tightened so the configs check the
properties the modules define:

- **OnlineScanner** now checks `NoMissedMatches` (reference-free form: no
  reachable final state is dropped without being recorded) and
  `MatchesRecordedCorrectly`, in addition to the structural invariants.
- **Subsumption** now checks `Irreflexive`, `Asymmetric`, `Transitive`,
  `CompletionPreservationInv`, and `NoFalseRemoval` explicitly (previously only
  `TypeInv`, which folded in the order properties).
- **ProductAutomaton** now checks `PatternPositionValid`.

Two known limitations remain, each reconciled against the Rust tests that cover
the gap:

- **ProductAutomaton** uses a *placeholder* (total) NFA transition relation, so
  `ProductCorrectness` and `CostMonotonicity` cannot be model-checked against a
  concrete NFA. They are verified on the real construction by
  `tests/proptest_product_automaton.rs` (acceptance / `min_distance` vs the exact
  edit-distance oracle, plus cost monotonicity).
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

**Key Properties:**
- `ProductCorrectness`: Acceptance iff NFA accepts AND cost <= max_cost
- `TransitionDeterminism`: Each (state, char) pair has well-defined transitions
- `CostMonotonicity`: Cost only increases along paths
- `StateSpaceBounded`: Product state space is polynomially bounded

**Corresponds to:** `src/phonetic/nfa/product.rs`

### Subsumption.tla

Verifies the subsumption relation used to prune redundant states.

**Key Properties:**
- `Irreflexive`: ~Subsumes(p, p) for all positions
- `Asymmetric`: Subsumes(p, q) => ~Subsumes(q, p)
- `Transitive`: Subsumes(p, q) /\ Subsumes(q, r) => Subsumes(p, r)
- `CompletionPreservation`: Subsuming state covers all completions

**Corresponds to:** `src/transducer/universal/subsumption.rs`

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
| Subsumption | MAX_POSITION=3, MAX_ERRORS=2 | ~10^3 |
| PriorityQuery | MAX_COST=2, WORD_LENGTH=3, DICT_SIZE=4 | ~10^4 |
| ValueYieldingQuery | MaxDistance=1, NoVal=999 (7-node dictionary) | ~30 |

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
