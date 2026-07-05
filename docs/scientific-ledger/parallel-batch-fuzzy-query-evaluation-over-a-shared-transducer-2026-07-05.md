---
pgmcp_experiment: parallel-batch-fuzzy-query-evaluation-over-a-shared-transducer
title: Parallel batch fuzzy-query evaluation over a shared transducer
date: 2026-07-05
project: liblevenshtein-rust
kind: optimization
status: decided
verdict: accepted
p_value: 0.000000
git_ref: cd385e3
---

# Parallel batch fuzzy-query evaluation over a shared transducer

**Kind:** optimization  |  **Status:** decided  |  **Correction:** benjamini_hochberg

## Method

**Question:** Phase 6 novel option #3 (specialize for observed query distributions) / SOTA parallel batch evaluation: the Transducer is Send+Sync (PathMapDictionary = Arc<ArcSwap>), so a batch (distribution) of independent fuzzy queries can be evaluated with rayon over a shared read-only transducer. Does parallel batch evaluation reduce whole-batch wall time versus sequential on a 512-query batch?

Transducer holds dictionary: D; PathMapDictionary is Arc<ArcSwap>-backed (Sync). query() takes &self, so a shared transducer can be queried across threads.

## Hypotheses

**H1.** Evaluating a 512-query batch with rayon par_iter over a shared &Transducer reduces whole-batch wall time versus sequential iteration by a large margin (multi-core, embarrassingly parallel), returning identical per-query result counts. — *✅ accepted*

- metric: `batch_wall_time_ms` (ms) · predicted: decrease · planned n/arm: 30
- pre-registered criterion (locked 2026-07-05 00:42:32Z): `{"type": "welch_t", "params": {"tail": "less", "alpha": 0.05, "min_effect": {"kind": "cohens_d", "threshold": 0.5}}}`

## Measurements & Decisions

| Metric | Test | Statistic | p | Effect | 95% CI | Verdict |
|--------|------|-----------|---|--------|--------|--------|
| `batch_wall_time_ms` | welch_t | -3663.538190 | 0.000000 | -529.084858 | [-161.9787, -161.8040] | accepted |

**Decision on `batch_wall_time_ms`:**

ACCEPTED (criterion: welch_t, correction: BenjaminiHochberg)
  [0] WelchT: statistic=-3663.5382, p=0.000000, effect=-529.0849, 95% CI=[-161.9787, -161.8040]

Operator note: Parallel batch fuzzy-query evaluation (512-query batch, dict 2000, d=2) via rayon par_iter over a shared read-only &Transducer, pinned to one 8-core CCD. Control (sequential) 185.98 ms vs treatment (parallel) 23.81 ms — ~7.8x on 8 cores (near-linear). Correctness gate passed: parallel per-query result counts identical to sequential. Transducer is Send+Sync (PathMapDictionary = Arc<ArcSwap>), so no code change needed — the win is an opt-in usage pattern (par_iter over queries).

## What did NOT work

_Nothing rejected (or no decisions yet)._

## Reproducibility

- git ref: `cd385e3`
- See each hypothesis's pre-registered criterion above; raw samples are retained in `experiment_samples`.

## Timeline

- 2026-07-05 00:42:32Z — **opened**: Parallel batch fuzzy-query evaluation over a shared transducer
- 2026-07-05 00:42:32Z — **criterion_locked**: Evaluating a 512-query batch with rayon par_iter over a shared &Transducer reduces whole-batch wall time versus sequential iteration by a large margin (multi-core, embarrassingly parallel), returning identical per-query result counts.
- 2026-07-05 00:44:54Z — **run**: treatment (treatment)
- 2026-07-05 00:46:00Z — **run**: control (control)
- 2026-07-05 00:46:13Z — **decided**: accepted on batch_wall_time_ms (welch_t)

---
_Rendered from the pgmcp experiment record (the structured source of truth). Edit the experiment, not this file._
