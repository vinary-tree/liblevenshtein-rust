---
title: Banded DTW, LB_Keogh, and prefix-first pruning gate
date: 2026-08-01
project: liblevenshtein-rust
kind: feature_evaluation
status: decided
verdict: correctness_accepted_performance_recorded
plan: /home/dylon/.claude/plans/liblevenshtein-currently-supports-levens-generic-sunrise.md
root_epic: extending-liblevenshtein-automaton-families-4bb97598
---

# Banded DTW, LB_Keogh, and prefix-first pruning gate

## Pre-registration provenance

The 2026-07-25 root plan fixed the required symmetric band, squared recurrence,
root-valued API, LB_Keogh envelopes, prefix-before-column gate, explicit
non-metric label, triangle counterexample, proof toolbox, differential tests,
and pruning-economics measurement before implementation. This ledger preserves
those hypotheses and acceptance rules. Focused correctness runs are
confirmatory because their thresholds predate the implementation. No
performance result is inferred from benchmark compilation.

## Hypotheses

| ID | Pre-registered hypothesis | Acceptance rule | Verdict |
|---|---|---|---|
| `dtw-001-differential` | The optimized two-row banded recurrence equals an independent full matrix. | Exact equality on boundaries and at least 2,000 generated sequence/band/cutoff cases. | **accepted** — boundary examples and 2,000 generated cases agree exactly. |
| `dtw-002-band` | The public API cannot silently omit the band, and endpoint reachability uses the same symmetric constraint. | No `Default` or unbanded public constructor; narrow/wide regression; formal length-gap obligation. | **accepted** — `DtwConfig::new(band)` is required, compile-fail docs reject `Default`, the same pair changes from `TOP` to finite when widened, and all formal tools cover endpoint reachability. |
| `dtw-003-k1` | Squared interval minima lift to admissible banded columns; point bins reproduce scalar columns. | At least 2,000 generated interval paths and 2,000 leaf boxes plus independent proof obligations. | **accepted** — generated cellwise and leaf tests pass; Rocq, Verus, Z3, and cvc5 discharge the corresponding invariants. |
| `dtw-004-keogh` | Candidate and interval-prefix LB_Keogh never exceed exact squared banded DTW. | At least 2,000 generated exact comparisons; prefix-sum induction and first-gate prune proof. | **accepted** — candidate and every generated prefix remain admissible; all formal layers prove the arithmetic and prune implication. |
| `dtw-005-walker` | Evaluating prefix LB before the child column preserves exact range and kNN results. | At least 2,000 generated range databases and 2,000 kNN databases; model-check prefix gate order and completeness. | **accepted** — 4,000 generated public searches match brute force; TLC exhausts 69 states and checks `PrefixGatePrecedesColumn`, no false positives, completeness, and termination. |
| `dtw-006-labelling` | DTW is symmetric/non-negative but not metric, and triangle-dependent structures cannot accept it through the checked kernel boundary. | Generated symmetry/non-negativity; fixed triangle counterexample in Rust and Rocq; `IS_METRIC=false`; compile-time marker rejection. | **accepted** — 2,000 generated pairs pass, the band-one witness is pinned in Rust and assumption-free Rocq, and `DtwConfig` does not implement `MetricElasticKernel`. |
| `dtw-007-pruning-economics` | Prefix LB_Keogh reduces band-column work on the common UCR slice. | Report total edges, prefix prunes, columns built, column prunes, candidate prunes, exact evaluations, and cutoff abandons; no favorable direction assumed. | **accepted as descriptive evidence** — over 44,855,191 visited edges, prefix LB_Keogh pruned 93,727 before constructing 44,761,464 columns; 641,224 columns and 345,595 queued subtrees were later pruned; the candidate gate pruned 136 finals before 52,446 exact evaluations, of which 11,409 cutoff-abandoned. |

## Confirmatory evidence

The focused suites exercise 12,000 generated cases: 8,000 kernel cases for the
reference recurrence, interval columns, leaf exactness, LB_Keogh, symmetry,
and non-negativity; plus 4,000 public range and k-nearest-neighbour databases.
Examples cover quantization collisions, upsert/removal, empty sides, non-finite
samples, root/squared unit conversion, cutoff inclusivity, and band
reachability.

Rocq compiles the interval, recurrence, prefix-sum, prune, reachability, and
non-metric theorems without axioms or admitted goals. Verus verifies 16 of 16
Rust-facing obligations. Z3 and cvc5 each report `UNSAT` for all 10 negated SMT
obligations. TLC generates 203 states, finds 69 distinct states, exhausts its
queue, and reports no invariant or temporal-property violation.

Principal captured logs are `/tmp/liblevenshtein_phase4c_dtw_unit.log`,
`/tmp/liblevenshtein_phase4c_dtw_integration.log`,
`/tmp/liblevenshtein_phase4c_rocq.log`,
`/tmp/liblevenshtein_phase4c_verus.log`,
`/tmp/liblevenshtein_phase4c_z3.log`,
`/tmp/liblevenshtein_phase4c_cvc5.log`, and
`/tmp/liblevenshtein_phase4c_tla.log`.

The fixed common-UCR band rule `$`w=\max(1,\lceil0.1L\rceil)`$` classified
11,581 of 13,754 cases correctly (0.842009597208), inside the preregistered
contextual interval. Summed per-dataset elapsed time was 17,421.721074 ms, peak
resident memory was 169,652 KiB, and the native squared-distance checksum was
78,364,758.759925141931. The
[shared ledger](elastic-ucr-harness-2026-08-01.md) records the complete
cross-kernel and paired-binary analysis.

## Fixed design decisions

| Decision | Rationale |
|---|---|
| Require `band` and provide no default. | The band changes reachability, complexity, and pruning behavior; silently choosing it would silently choose another distance. |
| Accumulate squared costs and expose root units. | LB_Keogh and DP remain additive in one domain; callers receive the conventional Euclidean-root scale. |
| Build envelopes with monotonic deques. | Every query index is processed a constant number of times, giving linear query planning. |
| Carry cumulative interval LB_Keogh. | The constant-time prefix gate can run before allocating or computing an `$`\mathcal{O}(w)`$` column. |
| Label metricity in the trait and marker boundary. | A code-queryable and compile-time contract is harder to misuse than a prose warning. |
| Keep exact originals behind quantized bins. | Quantization affects pruning only; every emitted distance remains full precision. |

## Measurement protocol

Correctness commands capture complete output under
`/tmp/liblevenshtein_phase4c_*`. The pending common UCR experiment must use a
fixed dataset manifest, query order, band policy, cutoff policy, warm-up, and
instrumentation schema. It must report counts even if the prefix gate regresses
runtime or prunes nothing. The correctness verdict does not depend on a
favorable performance direction. The registered measurement is now complete
and recorded above.
