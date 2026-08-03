---
title: TWED kernel correctness, metric contract, and pruning gate
date: 2026-08-01
project: liblevenshtein-rust
kind: feature_evaluation
status: decided
verdict: correctness_accepted_performance_recorded
plan: /home/dylon/.claude/plans/liblevenshtein-currently-supports-levens-generic-sunrise.md
root_epic: extending-liblevenshtein-automaton-families-4bb97598
---

# TWED kernel correctness, metric contract, and pruning gate

## Pre-registration provenance

The 2026-07-25 implementation plan selected Time Warp Edit Distance (TWED),
fixed the implementation order, required an independent quadratic reference
recurrence, and specified the differential, metric, interval, integration, and
formal gates before implementation. This ledger freezes the measurement
protocol before any TWED benchmark or UCR-archive run. Correctness checks that
instantiate those prior requirements are confirmatory.

The plan stated the metric domain too broadly as `$`\nu\ge0`$` and
`$`\lambda\ge0`$` while separately requiring the known
`$`\nu=\lambda=0`$` degeneracy. The primary-source proposition requires the
coefficient of timestamp displacement to be **strictly positive**. The Rust API
therefore separates the complete parameter family from a validated metric
witness. This correction was made before performance measurement and is not an
outcome-dependent reinterpretation.

## Hypotheses

| ID | Pre-registered hypothesis | Acceptance rule | Verdict |
|---|---|---|---|
| `twed-001-differential` | The optimized two-row recurrence equals an independent full-matrix recurrence. | Exact equality on source-derived examples, empty boundaries, cutoffs, saved regressions, and at least 2,000 generated inputs. | **accepted** — all six focused unit/property tests pass, including 2,000 optimized/reference/cutoff cases. |
| `twed-002-k1-match` | Carrying the previous target interval makes the two-sample TWED match leaf separable and exact over an interval box. | Closed-form leaf equals exhaustive integer-grid minimization; singleton bins equal scalar leaves; at least 2,000 generated boxes. | **accepted** — 2,000 generated interval paths are admissible and point-exact; 2,000 generated boxes establish separability and exactness. |
| `twed-003-k1-delete` | The target-deletion leaf is interval gap plus stiffness and gap penalty. | Closed form equals exhaustive integer-grid minimization and is no greater than every represented scalar deletion. | **accepted** — the same 2,000-case leaf suite establishes exact box minima. |
| `twed-004-k4-length` | `$`\lvert m-n\rvert\lambda`$` is a lower bound on exact TWED. | Arbitrary-script Rocq proof, independent SMT/Verus checks, and at least 2,000 generated exact comparisons. | **accepted** — Rocq proves the arbitrary-script bound without axioms or admissions, Verus and both SMT solvers discharge independent arithmetic encodings, and 2,000 implementation comparisons pass. |
| `twed-005-metric-domain` | TWED satisfies all four metric axioms only under the validated domain `$`\nu>0`$`, `$`\lambda\ge0`$` for finite unit-spaced series. | API prevents construction outside the premise; generated symmetry, non-negativity, identity, and triangle tests pass; formal artifacts pin the local and composition obligations. | **accepted for the declared API domain** — construction enforces the source premise, 2,000 generated triples pass all four axioms, the zero-stiffness counterexample remains executable, and the trusted artifacts prove the registered local/composition obligations. The Rocq island does not claim a new mechanization of Marteau's complete metric theorem. |
| `twed-006-zero-degeneracy` | The unrestricted `$`\nu=\lambda=0`$` family is non-metric. | A fixed unequal-series zero-distance witness remains executable and the raw type never implements `MetricElasticKernel`. | **accepted** — `[0,1]` and `[1]` have distance zero; compile-time contract tests distinguish raw and validated types. |
| `twed-007-search` | The generic elastic trie returns exact TWED range and k-nearest-neighbor answers despite quantization collisions. | Range sets/scores and kNN distance multisets equal brute force for examples and at least 2,000 generated databases per search mode. | **accepted** — 4,000 generated databases plus empty, mutation, collision, and invalid-input examples pass. |
| `twed-008-pruning-economics` | Carry-aware interval columns and the length lower bound reduce exact evaluations on the common UCR slice. | Report visited edges, candidate-bound prunes, column prunes, exact evaluations, cutoff abandons, memory, and wall time under the shared harness; no favorable direction is assumed. | **accepted as descriptive evidence** — the flat length bound pruned 85,642 of 1,306,949 candidates; the trie visited 86,278,155 edges, pruned 934,679 built columns and 127,358 queued subtrees, attempted 55,976 exact evaluations, and cutoff-abandoned 33,297. |

## Fixed experimental protocol

Correctness uses integer-valued generated samples converted exactly to `f64`.
This isolates recurrence logic from irrelevant representation noise while the
runtime still accepts arbitrary finite `f64` samples. Each proptest family runs
2,000 cases. Saved regression seeds remain committed even when the discovered
error was in a test oracle, because rerunning the exact witness protects the
oracle as well as production code.

Performance, when measured, uses the registered Criterion arms and the common
UCR harness. Every run must capture:

- kernel name and parameters `$`(\nu,\lambda)`$`;
- query and reference lengths and quantization configuration;
- visited edges, lower-bound prunes, exact evaluations, and cutoff abandons;
- elapsed time, peak resident memory, and result checksum;
- the brute-force result checksum for exactness.

Correctness does not depend on a favorable timing result. The shared protocol
has now run; its observed result is recorded below without changing any
correctness claim.

## Technical decisions fixed before measurement

| Decision | Rationale |
|---|---|
| `TwedConfig` represents every finite non-negative parameter pair and has `IS_METRIC = false`. | The family intentionally includes `$`\nu=0`$`; a static metric promise would be unsound for some values of the public type. |
| `MetricTwedConfig::try_new` requires finite `$`\nu>0`$` and finite `$`\lambda\ge0`$`. | This makes the primary-source metric premise unrepresentable by construction and is the only TWED type implementing `MetricElasticKernel`. |
| Samples have unit-spaced timestamps and share the sentinel `$`x_0=y_0=0`$`. | It gives a deterministic scalar API without introducing a second timestamp array; the recurrence remains the paper's timestamp-aware recurrence under this specialization. |
| Empty/nonempty distance is the accumulated segment-deletion boundary. | It is the recurrence's natural boundary and preserves finite edit paths, symmetry, and exact trie-root results. |
| The target carry is its preceding quantization interval. | TWED's match and deletion leaves depend on adjacent target samples; one interval pair is the minimal state that makes the relaxation Markovian. |
| The candidate bound is `$`\lvert m-n\rvert\lambda`$`. | Every length-changing path contains at least that many deletions and every deletion pays `$`\lambda`$` in addition to non-negative terms. |
| Non-finite samples are outside the exact search domain. | NaN and infinity do not provide a lawful total order or finite absolute-distance semantics for pruning. |

## Recorded evidence and result

The focused unit/property and public-surface integration suites passed. The
complete executable gate then passed 3,814 library tests and every integration
and example target selected by `cargo test --all-features --lib --tests
--examples`. Strict Clippy, formatting, rustdoc, doctest, documentation-math,
benchmark-compilation, example-compilation, and 57-diagram synchronization
gates also passed. The final pgmcp bug gate found no open bug anchored to any of
the 125 changed files.

The focused property families use 2,000 cases each. In particular, they cover
the optimized/reference/cutoff equivalence, point-exact and admissible interval
relaxations, exhaustive local box minima, four metric axioms for validated
configurations, the length lower bound, and exact range and k-nearest-neighbor
search against brute force. The search properties therefore contribute 4,000
generated databases in total.

Formal verification deliberately uses independent representations. Rocq
targets arbitrary-script length and local interval theorems; Verus connects the
arithmetic to Rust-shaped integer specifications; Z3 and cvc5 search for bounded
counterexamples; the generic `ElasticTrieSearch.tla` model covers K1–K4 traversal
once the kernel-specific arithmetic premises are discharged; and Rust
properties exercise the actual floating-point implementation. The trusted
formal gate passed in full: Rocq compiled the TWED island with no axioms or
admissions, Verus verified 13 obligations, Z3 and cvc5 each returned `unsat` for
13 negated obligations, and all seven repository TLA+ models completed. TLC was
run outside the filesystem sandbox because its worker infrastructure requires a
loopback RMI socket; that environment change does not alter the model or
configuration files.

Correctness is therefore accepted independently of performance. The common
UCR run classified 11,000 of 13,754 cases correctly (0.799767340410); summed
per-dataset elapsed time was 170,872.895160 ms, peak resident memory was
169,632 KiB, and the native-distance checksum was 3,960,614.403140308335. The
[shared ledger](elastic-ucr-harness-2026-08-01.md) records all flat/trie
counters, artifact digests, and pgmcp-computed paired-binary evidence.

## Sources

- P.-F. Marteau, “Time Warp Edit Distance with Stiffness Adjustment for Time
  Series Matching,” *IEEE Transactions on Pattern Analysis and Machine
  Intelligence* 31(2), 2009. DOI:
  [10.1109/TPAMI.2008.76](https://doi.org/10.1109/TPAMI.2008.76).
- The author's revised technical-report record and manuscript:
  [HAL hal-00135473v5](https://data.hal.science/document/hal-00135473v5).
