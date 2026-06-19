# Liblevenshtein Core Verification Library

**Namespace**: `Liblevenshtein.Core.Verification`

## Overview

This directory contains the reusable Rocq proof library for liblevenshtein's
edit-distance, automaton, trace, composition, lower-bound, and optimal-trace
contracts. Domain-specific verification work, including grammar and phonetic
automata, imports these modules instead of duplicating low-level distance and
transition reasoning.

The current active proof sources under `docs/verification/core/theories/` have
no top-level `Admitted.`, `admit.`, `Axiom`, `Parameter`, `Conjecture`, or
`Hypothesis` lines in the maintained `.v` files.

## Structure

```text
docs/verification/core/
|-- _CoqProject
|-- Makefile
|-- README.md
|-- ADMITTED_LEMMAS_STATUS.md
|-- COMPLETION_SUMMARY.md
`-- theories/
    |-- Automaton/          universal automaton states, transitions, soundness
    |-- Cardinality/        NoDup and inclusion preservation
    |-- Composition/        edit-sequence composition and cost bounds
    |-- Core/               distance definitions and metric properties
    |-- DPMatrix/           matrix operations and DP correctness support
    |-- LowerBound/         pigeonhole and shift-trace lower bounds
    |-- OptimalTrace/       optimal trace construction and validity
    |-- Trace/              Damerau, merge/split, and generic traces
    |-- Triangle/           triangle-inequality proof support
    |-- Distance.v          historical aggregate distance proof file
    |-- MainTheorems.v
    `-- TraceLowerBound.v
```

## Proof Families

| Family | Representative modules | Contract |
|--------|------------------------|----------|
| Edit distance | `Core/LevDistance.v`, `Core/MetricProperties.v`, `Distance.v` | Levenshtein distance, identity, symmetry, triangle structure, and trace equivalence |
| Transposition and merge/split | `Core/DamerauLevDistance.v`, `Core/MergeSplitDistance.v`, `Trace/DamerauTrace.v`, `Trace/MergeSplitTrace.v` | Operation-specific traces and cost accounting |
| Automata | `Automaton/*.v` | Universal automaton states, transitions, antichains, acceptance, soundness, and completeness |
| Composition | `Composition/*.v` | Edit-sequence composition, witness preservation, cardinality, and cost bounds |
| Lower bounds | `LowerBound/*.v`, `TraceLowerBound.v` | Pigeonhole and trace-derived pruning bounds |
| Optimal traces | `OptimalTrace/*.v` | Construction, validity, and cost equality for optimal edit traces |
| Triangle support | `Triangle/*.v` | Local cost lemmas used by triangle-inequality proofs |

## Build

Use memory-capped builds for agent runs and CI-like local verification. The
complete core proof suite is large: a 2 GiB capped build reaches
`DPMatrix/SnocLemmas.v` and is killed by the unit memory cap; a 4 GiB capped
build compiles past `DPMatrix/SnocLemmas.v` and reaches the trace layer; an
8 GiB capped build reaches `Trace/DamerauTrace.v` and is killed by that cap
while checking the trace proof. These are proof-compilation memory results from
`systemd-run --user --scope`, not runtime benchmark failures.

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
  make -C docs/verification/core/theories -j1
```

For focused changes, compile the touched dependency slice directly and pick a
cap appropriate for that slice. Example:

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  rocq c -Q docs/verification/core/theories Liblevenshtein.Core.Verification \
  docs/verification/core/theories/Triangle/TriangleInequality.v
```

## Maintenance Gates

Before committing changes in this directory:

1. Compile the touched Rocq slice under `systemd-run --user --scope`.
2. Run:

   ```bash
   rg -n "^\s*(Admitted\.|admit\.|Axiom |Parameter |Conjecture |Hypothesis )" \
     docs/verification/core/theories -g '*.v'
   ```

3. For full-suite status changes, record the `systemd-run` memory cap, unit
   result, peak memory, and last compiled Rocq file.
4. Run a stale-marker scan for changed docs and proof files.
5. Remove generated proof artifacts unless they are intentionally tracked by
   this repository.

## Status Documents

- `ADMITTED_LEMMAS_STATUS.md` records the Distance.v axiom-elimination audit.
- `COMPLETION_SUMMARY.md` records the major trace and triangle proof milestones.
- `PHASE*_*.md` files are historical session reports. Prefer this README and the
  active `.v` sources for current build status.

## Design Rationale

The library separates reusable mathematics from domain-specific correction
pipelines:

- Distance and trace proofs are independent of any dictionary implementation.
- Automaton proofs expose generic state and transition contracts.
- Composition proofs isolate the algebra needed to combine operation traces.
- Lower-bound proofs support pruning without coupling to a particular search
  engine.

This separation lets grammar, phonetic, and WFST-oriented work reuse the same
proof obligations while keeping implementation-specific concerns in Rust tests
and integration benchmarks.
