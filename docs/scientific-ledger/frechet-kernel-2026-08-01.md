---
title: Discrete Fréchet kernel correctness and bottleneck-monoid gate
date: 2026-08-01
project: liblevenshtein-rust
kind: feature_evaluation
status: decided
verdict: correctness_accepted_performance_recorded
plan: /home/dylon/.claude/plans/liblevenshtein-currently-supports-levens-generic-sunrise.md
root_epic: extending-liblevenshtein-automaton-families-4bb97598
---

# Discrete Fréchet kernel correctness and bottleneck-monoid gate

## Pre-registration provenance

The 2026-07-25 root plan fixed the recurrence, interval relaxation, metric
quotient, candidate bounds, proof obligations, differential gate, and kernel
order before implementation. This ledger transcribes those pre-existing
hypotheses without changing their thresholds. Focused development tests run
before this transcription remain confirmatory because the acceptance rules
come from the dated plan rather than their outcomes. No performance benchmark
has been measured.

## Hypotheses

| ID | Pre-registered hypothesis | Acceptance rule | Verdict |
|---|---|---|---|
| `frechet-001-differential` | The optimized two-row recurrence equals an independent Table 1 full matrix. | Exact equality on boundaries, branch examples, and at least 2,000 generated sequence pairs. | **accepted** — every source recurrence branch and 2,000 generated sequence/cutoff cases agree exactly. |
| `frechet-002-k1` | Exact interval-link minima lift through min/max to an admissible column, with point bins exactly reproducing scalar columns. | Cellwise admissibility and degenerate-bin equality on at least 2,000 generated interval paths, plus proof obligations. | **accepted** — 2,000 generated interval paths are cellwise admissible and point-exact; another 2,000 integer boxes establish exact leaf minima; all proof tools pass. |
| `frechet-003-bottleneck` | `BottleneckCost` exercises the unchanged generic walker soundly end to end. | Range and kNN differential agreement on at least 2,000 generated databases each; no Phase 4b walker edit. | **accepted** — range and kNN each pass 2,000 generated databases, with public examples, collisions, empty boundaries, mutations, determinism, and no walker-behavior change. |
| `frechet-004-k4` | Pinned endpoints and one-sided Hausdorff are admissible candidate bounds, and their maximum remains admissible. | Generated comparisons against exact distance plus Rocq and independent Verus/SMT obligations. | **accepted** — 2,000 generated metric triples compare each component and their maximum with exact distance; Rocq, Verus, Z3, and cvc5 independently discharge the bound structure. |
| `frechet-005-quotient` | Raw vectors form a pseudometric whose identity is run-length collapse. | Generated non-negativity, symmetry, triangle, and zero iff run-collapsed vectors agree; zero-link formal obligations. | **accepted** — 2,000 generated triples satisfy the laws and exact quotient identity; Rocq proves zero-link and non-negative zero-bottleneck consequences, mirrored by Verus/SMT. |
| `frechet-006-pruning-economics` | Interval and candidate bounds reduce exact scoring on the shared UCR slice. | Report visited edges, bound prunes, exact evaluations, and cutoff abandons; no favorable direction is assumed. | **accepted as descriptive evidence** — the flat candidate cascade pruned 995,028 of 1,306,949 candidates; the trie visited 27,413,343 edges, pruned 137,544 built columns and 635,332 queued subtrees, candidate-pruned 1,192 finals, attempted 29,818 exact evaluations, and cutoff-abandoned 3,069. |

## Confirmatory evidence

The focused suite exercised 12,000 generated cases: 8,000 unit/property cases
for the reference recurrence, interval columns, exact interval leaves, bounds,
metric laws, and quotient identity; plus 4,000 integration databases for range
and k-nearest-neighbour differential agreement. The all-feature repository run
then passed 3,800 library tests and all integration and documentation tests.

The formal evidence uses complementary semantics. Rocq proves recurrence
monotonicity, interval lifting, candidate bounds, bottleneck composition, and
zero-link consequences over mathematical reals without axioms or admitted
goals. Verus verifies 9/9 Rust-facing integer obligations. Z3 and cvc5 each
report `UNSAT` for all nine negated bounded obligations. The unchanged TLA+
generic walker model rechecks subtree/candidate pruning, exact emission,
terminal completeness, and termination. The complete repository formal
manifest passes with the three new artifacts registered as trusted.

Supporting gates also pass: all-target/all-feature clippy with warnings denied,
strict private-item rustdoc, format check, Criterion benchmark compilation, all
56 reproducible diagrams, and MathJax conformance across 264 living documents.
Principal captured logs are `/tmp/liblevenshtein_phase4_frechet_unit_2.log`,
`/tmp/liblevenshtein_phase4_frechet_integration_2.log`,
`/tmp/liblevenshtein_phase4_frechet_all_tests_1.log`, and
`/tmp/liblevenshtein_phase4_frechet_formal_all_1.log`.

The common UCR run decided `frechet-006` without changing any correctness
claim. Discrete Fréchet classified 10,696 of 13,754 cases correctly
(0.777664679366); summed per-dataset elapsed time was 29,063.445362 ms, peak
resident memory was 169,912 KiB, and the native-distance checksum was
88,981.171150006616. The
[shared ledger](elastic-ucr-harness-2026-08-01.md) retains every raw counter,
artifact digest, and paired-binary result.

## Fixed design decisions

| Decision | Rationale |
|---|---|
| `FrechetConfig` is a named unit kernel. | Scalar absolute point distance has no runtime parameter, while a named type keeps construction and future point-metric variants explicit. |
| One empty side maps to `TOP`. | The paper's coupling must cover both endpoint sets and is undefined when exactly one set is empty. |
| K4 takes endpoint bound `max` one-sided Hausdorff. | Both are independently admissible under every coupling; maximum preserves admissibility. |
| Identity is stated modulo consecutive-duplicate collapse. | Stutters trace the same polygonal curve and have zero bottleneck cost. |
| The walker remains unchanged. | The phase exists to demonstrate that its contract depends on ordered inflation, not additive accumulation. |

## Measurement protocol

Correctness commands capture complete output under
`/tmp/liblevenshtein_phase4_frechet_*`. Performance measurement, when run, uses
the registered Criterion benchmark and common UCR harness. No correctness
verdict depends on wall time, and no favorable performance direction is assumed.
The common measurement is now recorded above.
