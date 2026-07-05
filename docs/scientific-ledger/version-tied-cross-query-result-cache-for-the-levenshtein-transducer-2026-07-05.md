---
pgmcp_experiment: version-tied-cross-query-result-cache-for-the-levenshtein-transducer
title: Version-tied cross-query result cache for the Levenshtein transducer
date: 2026-07-05
project: liblevenshtein-rust
kind: optimization
status: decided
verdict: accepted
p_value: 0.000000
git_ref: 82cce83
---

# Version-tied cross-query result cache for the Levenshtein transducer

**Kind:** optimization  |  **Status:** decided  |  **Correction:** benjamini_hochberg

## Method

**Question:** AMD uProf (tbp) shows transducer::transition::transition_state_pooled_ref dominates query CPU at ~80% (100.9s). Does a dictionary-version-tied query-result cache that skips the entire transition loop on repeated queries reduce mean per-query latency on a repeated-query workload, without regressing unique-query workloads unacceptably?

Phase 6 novel option #4 (cross-query cache with eviction tied to dictionary versioning). Profiled with AMD uProf tbp; transition_state_pooled_ref is the sole hot function. A cache hit skips it entirely.

## Hypotheses

**H1.** A VersionedQueryCache memoizing (query, max_distance) -> results and invalidating on a dictionary version bump reduces mean per-query latency on a repeated-query workload by a large margin versus the uncached transducer, while returning identical results. — *✅ accepted*

- metric: `mean_query_latency_ns` (ns) · predicted: decrease · planned n/arm: 30
- pre-registered criterion (locked 2026-07-05 00:13:14Z): `{"type": "welch_t", "params": {"tail": "less", "alpha": 0.05, "min_effect": {"kind": "cohens_d", "threshold": 0.5}}}`

## Measurements & Decisions

| Metric | Test | Statistic | p | Effect | 95% CI | Verdict |
|--------|------|-----------|---|--------|--------|--------|
| `mean_query_latency_ns` | welch_t | -619.411566 | 0.000000 | -80.866202 | [-285238.0267, -283414.9544] | accepted |

**Decision on `mean_query_latency_ns`:**

ACCEPTED (criterion: welch_t, correction: BenjaminiHochberg)
  [0] WelchT: statistic=-619.4116, p=0.000000, effect=-80.8662, 95% CI=[-285238.0267, -283414.9544]

Operator note: Version-tied cross-query cache vs uncached transducer on a repeated-query workload (50 queries × replicates, dict 1000 terms, max_distance 2), CPU-pinned core 2. Cache hits skip the AMD-uProf-dominant transition_state_pooled_ref (~80% CPU). Correctness gate passed (cached results identical to uncached for all 50 queries).

## What did NOT work

_Nothing rejected (or no decisions yet)._

## Reproducibility

- git ref: `82cce83`
- See each hypothesis's pre-registered criterion above; raw samples are retained in `experiment_samples`.

## Timeline

- 2026-07-05 00:13:14Z — **opened**: Version-tied cross-query result cache for the Levenshtein transducer
- 2026-07-05 00:13:14Z — **criterion_locked**: A VersionedQueryCache memoizing (query, max_distance) -> results and invalidating on a dictionary version bump reduces mean per-query latency on a repeated-query workload by a large margin versus the uncached transducer, while returning identical results.
- 2026-07-05 00:19:55Z — **run**: control (control)
- 2026-07-05 00:21:21Z — **run**: treatment (treatment)
- 2026-07-05 00:21:48Z — **decided**: accepted on mean_query_latency_ns (welch_t)

---
_Rendered from the pgmcp experiment record (the structured source of truth). Edit the experiment, not this file._
