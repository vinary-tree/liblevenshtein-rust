---
pgmcp_experiment: degenerate-metric-walkers
title: Degenerate Hamming and indel dictionary walkers
date: 2026-07-31
project: liblevenshtein-rust
kind: optimization
status: decided
verdict: rejected
p_value: n/a
plan: /home/dylon/.claude/plans/liblevenshtein-currently-supports-levens-generic-sunrise.md
git_ref: 3f362b4
---

# Degenerate Hamming and indel dictionary walkers

**Kind:** optimization  |  **Status:** decided  |  **Correction:** benjamini_hochberg

## Method

**Question:** Do the candidate Hamming and indel walkers satisfy the Phase-8 compound shipping
gate of at least two-times faster at k in {1,2} on at least two dictionary sizes and at least
four-times fewer enumerated edges?

The compound ship rule was locked in the external plan dated 2026-07-25. The first benchmark
run was invalidated after it exposed sibling DoubleArrayTrieChar base-allocation corruption;
D14 fixed that defect and this experiment ingests only the complete post-fix rerun.

## Hypotheses

**H1.** Across the pre-registered 3 dictionary sizes × 3 query lengths × k in {1,2} matrix
for both metrics, the candidate walker achieves the runtime clause and an aggregate
edge-enumeration reduction of at least 4x relative to the honest baseline. — *❌ rejected*

- metric: `edge_reduction_factor` (ratio) · predicted: increase · planned n/arm: —
- pre-registered criterion (locked 2026-07-31 20:10:56Z):
  `{"type": "absolute_threshold", "params": {"op": "ge", "arm": "treatment", "stat": "mean", "value": 4.0}}`

## Measurements & Decisions

| Metric | Test | Statistic | p | Effect | 95% CI | Verdict |
|--------|------|-----------|---|--------|--------|--------|
| `edge_reduction_factor` | absolute_threshold | 1.228961 | n/a | n/a | — | rejected |

**Decision on `edge_reduction_factor`:**

REJECTED (criterion: absolute_threshold, correction: BenjaminiHochberg)
[0] AbsoluteThreshold: statistic=1.2290, p=n/a

Operator note: REJECT shipping the specialized walkers. The runtime clause passed
comfortably: both Hamming and indel candidates were more than 2x faster at k=1 and k=2 for
all three dictionary sizes and all three query lengths. The required structural clause
failed: aggregate edge reduction was only 322,794/262,656 = 1.229x, and no measured
configuration approached 4x. Per the locked compound rule, the walkers remain benchmark-only
and Phase 8 must use the repaired GeneralizedAutomaton/reference path.

## What did NOT work

- `edge_reduction_factor`: rejected (test=absolute_threshold, p=n/a)

## Reproducibility

- git ref: `3f362b4`
- See each hypothesis's pre-registered criterion above; raw samples are retained in
  `experiment_samples`.

## Timeline

- 2026-07-31 20:10:56Z — **opened**: Degenerate Hamming and indel dictionary walkers
- 2026-07-31 20:10:56Z — **criterion_locked**: Across the pre-registered 3 dictionary sizes ×
  3 query lengths × k in {1,2} matrix for both metrics, the candidate walker achieves the
  runtime clause and an aggregate edge-enumeration reduction of at least 4x relative to the
  honest baseline.
- 2026-07-31 20:11:10Z — **run**: treatment (treatment)
- 2026-07-31 20:11:19Z — **decided**: rejected on edge_reduction_factor (absolute_threshold)

---
_Rendered from the pgmcp experiment record (the structured source of truth). Edit the
experiment, not this file._
