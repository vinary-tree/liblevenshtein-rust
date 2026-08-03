---
pgmcp_experiment: language-product-cost-indexed-frontier
title: LanguageProduct cost-indexed frontier versus legacy cost-distinct frontier
date: 2026-07-31
project: liblevenshtein-rust
kind: feature_refactor
status: decided
verdict: accepted
p_value: 0.000000
plan: /home/dylon/.claude/plans/liblevenshtein-currently-supports-levens-generic-sunrise.md
git_ref: 3f362b4
---

# LanguageProduct cost-indexed frontier versus legacy cost-distinct frontier

**Kind:** feature_refactor  |  **Status:** decided  |  **Correction:** benjamini_hochberg

## Method

**Question:** Does the generic cost-indexed LanguageProduct frontier reduce mean per-input
transition latency across the pre-registered literal, alternation, and branching cases while
preserving exact accepting distance?

Criterion was locked in the external implementation plan dated 2026-07-25 before measurement.
This database registration precedes sample ingestion; correctness is a separate
property/differential gate.

## Hypotheses

**H1.** Across the fixed Phase-0 pattern/input corpus, the canonical minimum-cost
LanguageProduct frontier decreases mean per-input transition latency relative to the legacy
cost-distinct frontier while returning the same accepting distance. — *✅ accepted*

- metric: `transition_latency_ns` (ns) · predicted: decrease · planned n/arm: 51
- pre-registered criterion (locked 2026-07-31 20:08:36Z):
  `{"type": "welch_t", "params": {"tail": "less", "alpha": 0.05, "min_effect": {"kind": "cohens_d", "threshold": 0.5}}}`

## Measurements & Decisions

| Metric | Test | Statistic | p | Effect | 95% CI | Verdict |
|--------|------|-----------|---|--------|--------|--------|
| `transition_latency_ns` | welch_t | -11.866418 | 0.000000 | -0.968889 | [-497970.2678, -356308.5597] | accepted |

**Decision on `transition_latency_ns`:**

ACCEPTED (criterion: welch_t, correction: BenjaminiHochberg)
[0] WelchT: statistic=-11.8664, p=0.000000, effect=-0.9689, 95% CI=[-497970.2678, -356308.5597]

Operator note: ACCEPT for the implementation plan's runtime hypothesis: exact
accepting-distance equality was asserted for every case before timing, 600-case
differential/property tests independently pass, and treatment mean latency is materially
lower. Limitation: this run used the performance governor but was not CPU-pinned; the effect
is substantially larger than ambient variance. State expansions are not uniformly lower
(158 vs 167 literal, 118 vs 57 alternation, 824 vs 371 branching), so no structural-pruning
claim is made.

## What did NOT work

_Nothing rejected (or no decisions yet)._

## Reproducibility

- git ref: `3f362b4`
- See each hypothesis's pre-registered criterion above; raw samples are retained in
  `experiment_samples`.

## Timeline

- 2026-07-31 20:08:36Z — **opened**: LanguageProduct cost-indexed frontier versus legacy
  cost-distinct frontier
- 2026-07-31 20:08:36Z — **criterion_locked**: Across the fixed Phase-0 pattern/input corpus,
  the canonical minimum-cost LanguageProduct frontier decreases mean per-input transition
  latency relative to the legacy cost-distinct frontier while returning the same accepting
  distance.
- 2026-07-31 20:08:52Z — **run**: treatment (treatment)
- 2026-07-31 20:08:52Z — **run**: control (control)
- 2026-07-31 20:09:02Z — **decided**: accepted on transition_latency_ns (welch_t)

---
_Rendered from the pgmcp experiment record (the structured source of truth). Edit the
experiment, not this file._
