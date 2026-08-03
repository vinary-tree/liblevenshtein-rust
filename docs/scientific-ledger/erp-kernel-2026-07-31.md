---
title: ERP kernel correctness and pruning gate
date: 2026-07-31
project: liblevenshtein-rust
kind: feature_evaluation
status: decided
verdict: correctness_accepted_performance_recorded
plan: /home/dylon/.claude/plans/liblevenshtein-currently-supports-levens-generic-sunrise.md
root_epic: extending-liblevenshtein-automaton-families-4bb97598
---

# ERP kernel correctness and pruning gate

## Pre-registration provenance

The 2026-07-25 implementation plan fixes the ERP recurrence, its gap-mass
candidate bound, its quotient metric status, and the differential/property
gates before implementation. This ledger preserves those hypotheses before
any performance benchmark is measured. Correctness tests that implement the
pre-existing plan are confirmatory, not exploratory.

## Hypotheses

| ID | Pre-registered hypothesis | Acceptance rule | Verdict |
|---|---|---|---|
| `erp-001-differential` | The optimized two-row DP equals an independent full-matrix ERP recurrence. | Exact equality on worked examples, boundaries, and at least 2,000 generated triples `(x,y,g)`. | **accepted** — Chen–Ng worked examples, empty boundaries, cutoff behavior, and 2,000 generated triples agree exactly. |
| `erp-002-k1` | Interval columns lower-bound every represented concrete column. | At least 2,000 generated columns satisfy cellwise admissibility; point bins are exactly equal. | **accepted** — 2,000 generated interval columns are cellwise admissible for center/lower/upper realizations; point bins and 2,000 generated leaves are exact. |
| `erp-003-k4` | The absolute gap-mass difference lower-bounds every exact ERP distance. | Rocq proof over arbitrary edit scripts, independent SMT/Verus checks, and generated differential properties. | **accepted** — Rocq proves the arbitrary-edit-script potential bound; Verus verifies 8/8 obligations; Z3 and cvc5 each refute all 8 negated obligations; generated Rust cases agree. |
| `erp-004-quotient` | ERP is a pseudometric on raw sequences and has identity only after deleting occurrences of the fixed gap value `$`g`$`. | Property tests cover non-negativity, symmetry, triangle inequality, and zero iff quotient normal forms agree; Rocq proves zero-cost alignments imply quotient identity. | **accepted** — 2,000 generated triples satisfy the stated pseudometric laws and quotient identity; Rocq proves quotient equivalence, gap removal, and zero-cost alignment identity. |
| `erp-005-search` | The generic trie walker returns exact ERP range and kNN answers. | Range sets/scores and kNN distance multisets equal brute force for examples, saved regressions, and at least 2,000 generated databases. | **accepted** — range and kNN each pass 2,000 generated databases plus the saved empty-root regression; exact rescoring preserves brute-force scores and distance multisets. |
| `erp-006-pruning-economics` | ERP's gap-mass and interval bounds reduce exact candidate evaluations on the common UCR slice. | Report visited edges, candidate-bound prunes, exact evaluations, and cutoff abandons; no favorable direction is assumed. | **accepted as descriptive evidence** — the flat gap-mass bound pruned 368,186 of 1,306,949 candidates; the trie visited 71,097,379 edges, pruned 1,071,152 built columns and 88,753 queued subtrees, candidate-pruned 209 finals, attempted 39,665 exact evaluations, and cutoff-abandoned 13,557. |

## Confirmatory evidence

The correctness decision used the pre-registered acceptance rules without
changing their thresholds. Focused ERP tests exercised 10,000 generated cases:
6,000 unit/property cases for the scalar recurrence, interval relaxation, leaf
exactness, gap-mass bound, and quotient laws; and 4,000 integration cases for
range and k-nearest-neighbor differential agreement. The full all-feature suite
then passed 3,794 library tests and all integration and documentation tests.

Formal evidence is deliberately heterogeneous. Rocq establishes the unbounded
mathematical statements over arbitrary edit scripts; Verus connects arithmetic
invariants to executable-style integer specifications; Z3 and cvc5 independently
search for counterexamples to eight quantifier-free obligations; and TLC checks
root-terminal completeness in the generic walker state machine. The repository-
wide formal manifest passed after the ERP obligations were registered.

The exact captured commands are recorded in pgmcp. The principal logs are
`/tmp/liblevenshtein_phase4_erp_unit_2.log`,
`/tmp/liblevenshtein_phase4_erp_integration_3.log`,
`/tmp/liblevenshtein_phase4_erp_all_tests_1.log`, and
`/tmp/liblevenshtein_phase4_erp_formal_all_1.log`.

The common UCR run decided `erp-006` without changing the correctness gate.
ERP classified 11,858 of 13,754 cases correctly (0.862149192962); summed
per-dataset elapsed time was 107,069.394305 ms, peak resident memory was
169,136 KiB, and the native-distance checksum was 4,287,337.931012114510. The
[shared ledger](elastic-ucr-harness-2026-08-01.md) records the complete flat,
trie, artifact, and paired-binary evidence.

## Fixed design decisions

| Decision | Rationale |
|---|---|
| `ErpConfig` is the kernel; `ErpKernel` is an alias. | The gap value is ERP's only runtime state, so a second wrapper adds no invariant. |
| Non-finite direct-field `g` normalizes to zero. | The public field cannot be made private compatibly; normalization keeps ordered arithmetic total. |
| Non-finite samples are outside the exact domain. | NaN has no lawful absolute-distance or heap order semantics. |
| The K4 bound is gap-mass difference, never length difference. | ERP gap cost depends on value; length alone can imply no positive cost. |
| Exact kNN admits a final trie root. | ERP can assign finite or zero distance between a nonempty query and an empty candidate. |

## Measurement protocol

Correctness commands capture complete output under `/tmp/liblevenshtein_phase4_erp_*`.
Performance measurement, when run, uses the registered Criterion benchmark and
the common UCR harness; correctness results never depend on timing. That
measurement is now recorded above and in the shared ledger.
