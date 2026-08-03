---
title: affine-gap correctness, pruning, and resource gate
date: 2026-08-01
project: liblevenshtein-rust
kind: feature
status: complete
verdict: all_preregistered_correctness_formal_corpus_documentation_and_measurement_gates_passed
root_epic: extending-liblevenshtein-automaton-families-4bb97598
work_item: phase-7-affine-gap-gotoh-1982-doi-10-1016-0022-2836-82-90398-9-afe5c6b6
---

# Affine-gap correctness, pruning, and resource gate

## 1. Provenance

The expanded implementation plan pre-registered quadratic-Gotoh differential
agreement, exact scaling, B-1 through B-4 subsumption obligations, an
operation-derived window, layer-aware finish, symmetry/identity properties,
and the backend/unit/policy matrix before implementation. This ledger was
created after the first focused tests and proofs, so it does not relabel those
runs as newly pre-registered.

The benchmark protocol in Section 4 had not run when this file was created.
Its population and reporting fields are frozen before measurement.

## 2. Terms and hypotheses

An **affine gap** is a contiguous one-sided run whose length-`$`r`$` cost is
`$`g_o+r g_e`$`. **B-4** is the same-index layer-aware subsumption rule. The
**operation window** is the maximum number of extension-cost operations that
can fit in the remaining exact budget, plus the current diagonal lookahead.

| ID | Hypothesis or obligation | Frozen decision rule |
|---|---|---|
| `affine-001-reference` | The three-matrix DP implements the selected Gotoh convention. | Gap-run examples, empty boundaries, symmetry, and `$`g_o=0,g_e=s=1`$` Levenshtein degeneracy pass. |
| `affine-002-automaton` | The lazy automaton is extensionally equal to the DP. | Exact `(term, scaled_cost)` map equality over 2,000 generated dictionaries, queries, budgets, and parameter sets. |
| `affine-003-subsume` | Every enabled B-4 comparison preserves residual-language cost. | Rocq arbitrary-trace theorem; Dafny/Verus/SMT/TLC step preservation; 2,000 generated position-pair/suffix checks. |
| `affine-004-window` | Window width derives from operation budget, not raw scaled cost. | Formal quotient bound and 2,000 generated affordable runs; a scale-1,000 example produces width 5 rather than 1,001. |
| `affine-005-finish` | A trailing query run pays open exactly when not already in `$`I_x`$`. | Formal finish equations, focused unit examples, and complete DP differential equality. |
| `affine-006-laws` | Symmetry and positive-cost identity hold. | 2,000 cases each. No triangle-inequality claim is made. |
| `affine-007-genericity` | Units and policies do not change weighted operation semantics. | byte/char/`u64`, unrestricted/borrowed/owned/character policies, and applicable dictionaries pass. |
| `affine-008-performance` | Exact scaling and the three-layer frontier have bounded practical overhead. | Criterion reports Standard and affine medians for query lengths 8, 16, and 32 at two parameter sets; report ratios without a ship threshold. |

## 3. Correctness and formal results

| Evidence | Population | Result | Verdict |
|---|---:|---:|---|
| focused kernel/reference unit tests | 7 examples | 7 passed | accepted |
| automaton/reference differential | 2,000 generated cases | exact map and cost equality | accepted |
| budget monotonicity | 2,000 generated cases | every lower-budget result retained with identical cost | accepted |
| deterministic traversal | 2,000 generated cases | result vectors identical | accepted |
| symmetry and positive-cost identity | 2,000 generated cases | both properties pass | accepted |
| B-4 suffix dominance | 2,000 generated pairs/suffixes | every enabled comparison preserves completion inequality | accepted |
| uniform switch penalty | 2,000 generated layer/action configurations | `$`\Delta\le g_o`$` in every case | accepted |
| operation-window affordability | 2,000 generated budgets/runs | every affordable run lies inside the window | accepted |
| Birkbeck spelling corpus | 42,395 explicit pairs; 32,120 with reference cost at most 3 | every eligible correction found at its exact DP cost | accepted |
| Dafny | 14 proof obligations | 14 verified, 0 errors | accepted |
| Verus | 8 proof functions | 8 verified, 0 errors | accepted |
| Rocq | assumption-free theory | compiled with no admitted result or axiom | accepted |
| Z3 and cvc5 | 10 negated invariants per solver | 20 `unsat`, no `sat`/`unknown` | accepted |
| TLC | complete finite B-4 graph | 1,764 generated, 710 distinct, depth 4, no violation | accepted |

Captured evidence is stored under `/tmp/liblevenshtein_phase7_*` during the
active implementation run and is removed after final repository validation.

## 4. Frozen Criterion protocol

Use one `DoubleArrayTrie` containing exact, substitution-heavy, single-gap, and
multi-gap candidates. For query lengths 8, 16, and 32, measure:

1. `Algorithm::Standard` at edit budget 2;
2. affine with `$`g_o=0,g_e=s=1`$` at scaled budget 2;
3. affine with `$`g_o=2,g_e=1,s=2`$` at budget 6.

Record median estimates and affine/Standard ratios. Do not discard an arm and
do not infer asymptotic complexity from three lengths. The benchmark is a
cost report, not an acceptance threshold; correctness and resource bounds are
separate gates.

### 4.1 Recorded result

The registered benchmark ran with Criterion 0.8.2, 20 samples, a one-second
warm-up, and a one-second measurement window. Each arm queried the same
260-term `DoubleArrayTrie`; the corpus constructor asserts that exact unique
population so accidental generator collisions fail loudly.

| Query length | Standard `$`k=2`$` median | Affine `(0,1,1)`, `$`k=2`$` median / ratio | Affine `(2,1,2)`, `$`k=6`$` median / ratio |
|---:|---:|---:|---:|
| 8 | 148.97 µs | 171.47 µs / 1.15× | 303.86 µs / 2.04× |
| 16 | 158.62 µs | 183.89 µs / 1.16× | 332.15 µs / 2.09× |
| 32 | 158.51 µs | 207.74 µs / 1.31× | 341.59 µs / 2.16× |

Environment: AMD Ryzen Threadripper PRO 5975WX, Linux 7.1.5-arch1-2,
`rustc 1.97.1` with LLVM 22.1.6, release profile with link-time optimization.
The run was not CPU-pinned and frequency boost was enabled. Ratios are the
portable comparison within this run; absolute times and three-point trends are
descriptive, not a general asymptotic claim.

## 5. Protocol correction: B-5

The first 2,000-case differential run found and persisted seed
`d7a2f9a02a6170f32c7404adb71924e7e629a6891fa4f161ed1bd53d93b2321b`.
It minimized to query `ba`, dictionary term `a`, `$`g_o=0`$`, `$`g_e=s=1`$`,
and budget 1. Provisional B-5 cross-index pruning removed `(1,1,I_x)` beneath
`(0,0,M)`; the unfused successor inspected only the latter's current query
offset and missed the match.

The corrective decision is conservative and general: ship formally verified
B-4; keep all cross-index affine positions incomparable; require a fused
skip-and-consume transition and refinement proof before revisiting B-5. The
saved seed is committed in `tests/affine_gap.proptest-regressions`.

## 6. Repository gates

The final repository surface was checked after adding the corpus gate and
updating the source documentation. Command output remains under
`/tmp/liblevenshtein_phase7_*` until the complete multi-phase implementation
run finishes.

| Gate | Result |
|---|---|
| `cargo test --all-features` | passed; 3,839 library tests plus all enabled integration, property, and documentation tests |
| `cargo test --no-default-features` | passed; 1,072 library tests plus the enabled minimal integration, property, and documentation tests |
| affine backend-focused suites | 10 passed across default, PathMap, and persistent backends |
| Birkbeck affine corpus | 42,395 source pairs traversed; all 32,120 budget-eligible pairs passed in 2.29 seconds |
| `cargo run --example affine_gap` | passed with exact automaton/reference agreement |
| `cargo fmt --all -- --check` | passed |
| strict all-target, all-feature Clippy | passed with warnings denied |
| strict all-feature rustdoc | passed with warnings denied |
| `scripts/doc-mathlint.sh` | passed across 279 included documents |
| `docs/diagrams/render.sh --check` | all 60 rendered SVGs reproducible |
| `scripts/verify-formal.sh all` | Rocq, Dafny, Verus, Z3, cvc5, and every registered TLA+ model passed |
| `git diff --check` | passed |
| `pgmcp bug-gate` | passed; no open bug anchored to the 150 changed files |

## 7. Reference

O. Gotoh, “An improved algorithm for matching biological sequences,” *Journal
of Molecular Biology* 162(3), 705–708 (1982).
[DOI 10.1016/0022-2836(82)90398-9](https://doi.org/10.1016/0022-2836(82)90398-9).
