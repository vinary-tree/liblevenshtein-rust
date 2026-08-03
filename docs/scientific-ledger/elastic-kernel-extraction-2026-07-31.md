---
title: ElasticKernel extraction and MSM compatibility gate
date: 2026-07-31
project: liblevenshtein-rust
kind: feature_refactor
status: accepted
verdict: accepted
plan: /home/dylon/.claude/plans/liblevenshtein-currently-supports-levens-generic-sunrise.md
root_epic: extending-liblevenshtein-automaton-families-4bb97598
---

# ElasticKernel extraction and MSM compatibility gate

## Pre-registration provenance

The implementation plan dated 2026-07-25 pre-registered Phase 3 before this
extraction. Its criterion is semantic rather than statistical: every existing
MSM unit, integration, property, saved-regression, empty/non-finite, collision,
mutation, range, and kNN test must pass without changing the `MsmTransducer`
public API. This ledger does not retroactively invent a performance claim.

## Hypotheses and decisions

| ID | Hypothesis | Verdict | Evidence and interpretation |
|---|---|---|---|
| `elastic-001-msm-equivalence` | Replacing the concrete walker by `ElasticTransducer<MsmKernel,V>` preserves all observable MSM behavior. | accepted | Unchanged MSM unit/proptests: 19 passed with 2,000 property cases; public integration suites: 41 passed. Range and kNN remain differential-equal to brute force. |
| `elastic-002-no-hidden-msm-dependency` | A second kernel can use the public seam without modifying walker code. | accepted | `PointwiseL1` integration kernel exercises range, kNN, empty/length mismatch, threshold monotonicity, deterministic ordering, and 2,000-case differential equality. |
| `elastic-003-pruning-contract` | K1–K4 suffice for subtree pruning, candidate pruning, exact emission, and best-first termination for additive and bottleneck accumulation. | accepted | Rocq proof passes; Verus verifies 8/8; Z3 and cvc5 each return 7/7 expected `UNSAT`; TLC explores all 35 states with no error. |
| `elastic-004-interval-geometry` | The two-interval gap is symmetric, non-negative, zero exactly on intersection, and exact on point bins. | accepted | Rust 2,000-case property, Verus proof, and cross-solver SMT counterexample checks pass. |
| `elastic-005-performance` | Extraction is faster or slower than the prior MSM implementation. | not tested | No performance hypothesis was pre-registered for Phase 3. Benchmark claims are reserved for the per-kernel Phase 4 pruning-economics experiments. |

## Technical decisions

| Decision | Rationale | Consequence |
|---|---|---|
| Query plans are borrowed by column and candidate-bound methods. | DTW envelopes must be computed once per query, not `$`\mathcal{O}(m)`$` per edge. | `QueryPlan` is live shared metadata rather than a decorative associated type. |
| Empty/nonempty cost receives the nonempty series. | ERP empty-side cost is a running sum of value-dependent `$`\lvert x_i-g\rvert`$` gaps. | The trait is not constrained to MSM's constant `TOP` boundary. |
| Invalid interval queries use deterministic exact scan. | NaN in DP columns or priority queues would violate the ordered-cost domain. | Range behavior remains exact and stable; kNN emits only finite exact scores. |
| `TOP` is excluded from kNN result heaps. | An unreachable/infinite candidate is not a finite nearest neighbor. | Legacy MSM empty and non-finite kNN behavior is preserved. |
| MSM candidate K4 uses `length_lb`. | The bound is already proved and inexpensive. | Some survivors are rejected before exact MSM without changing results. |

## Reproducibility

Captured validation logs:

- `/tmp/liblevenshtein_phase3_elastic_unit.log`
- `/tmp/liblevenshtein_phase3_msm_integration.log`
- `/tmp/liblevenshtein_phase3_generic_kernel_properties_2.log`
- `/tmp/liblevenshtein_phase3_coq_2.log`
- `/tmp/liblevenshtein_phase3_verus_2.log`
- `/tmp/liblevenshtein_phase3_smt.log`
- `/tmp/liblevenshtein_phase3_tla_3_unsandboxed.log`

These logs are working-session evidence; the committed sources, tests, and
formal manifest are the durable reproducibility record.
