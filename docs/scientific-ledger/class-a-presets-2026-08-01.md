---
title: Class-A preset correctness and specialization gate
date: 2026-08-01
project: liblevenshtein-rust
kind: feature
status: complete
verdict: presets_accepted_dedicated_walkers_rejected
root_epic: extending-liblevenshtein-automaton-families-4bb97598
work_item: phase-8-class-a-presets-and-the-degenerate-walkers-816011c0
---

# Class-A preset correctness and specialization gate

## 1. Provenance

The expanded plan pre-registered the preset/oracle/reference three-way gate,
metric and boundary properties, complete Birkbeck corpus applicability, and the
compound degenerate-walker benchmark before Phase-8 implementation. This
ledger was created after the first core and property runs, so it reports those
results without relabeling them as newly pre-registered.

The walker measurement is not repeated here. Its append-only
[Phase-0 record](degenerate-metric-walkers-2026-07-31.md) is the source of the
frozen decision.

## 2. Hypotheses and decision rules

| ID | Hypothesis or obligation | Frozen rule |
|---|---|---|
| `class-a-001-hamming` | The preset and scalar mismatch count denote the same partial metric. | Preset = explicit operation set = reference on 2,000 generated pairs; fixed-length metric laws pass. |
| `class-a-002-indel` | The preset equals both the DP and the LCS identity. | Three-way equality on 2,000 pairs; symmetry, triangle, length, parity, and threshold laws pass. |
| `class-a-003-skip` | Bounded skip is exactly directional subsequence matching. | Three-way equality on 2,000 pairs, including empty and Unicode boundaries. |
| `class-a-004-validation` | Invalid or excessive operation sets fail before grid expansion. | Focused boundaries, 4,000 generated valid/non-progressing sets, and heterogeneous formal proofs pass. |
| `class-a-005-corpus` | Synthetic equality generalizes to spelling data. | Every explicit Birkbeck pair agrees with each applicable independent reference. |
| `class-a-006-walkers` | Dedicated dictionary walkers justify another public traversal path. | Ship only if runtime is at least two-times faster on the frozen arms and enumerated edges drop at least four-times. |

## 3. Results to date

| Evidence | Population | Result | Verdict |
|---|---:|---:|---|
| preset/manual/reference properties | 6,000 cases | exact equality | accepted |
| metric, threshold, and inter-metric properties | 10,000 cases | all invariants pass | accepted |
| resource-validation properties | 4,000 cases | valid sets accepted; cycles rejected | accepted |
| focused examples and compatibility suite | 1 runnable example; 9 compatibility tests | all pass | accepted |
| Birkbeck corpus | 42,395 pairs | every Hamming, indel, and skip result agrees | accepted |
| Hamming applicability | 13,297 equal-length corpus pairs | exact mismatch count agrees | accepted |
| bounded-skip applicability | 10,671 subsequence corpus pairs | exact skipped count agrees | accepted |
| dedicated-walker structural gate | 322,794 baseline edges / 262,656 candidate edges = 1.229 times | below required four-times reduction | rejected |
| formal toolbox | Rocq theorem suite; Dafny 16/16; Verus 10/10; Z3 and cvc5 13/13 unsatisfiable obligations; TLA+ 124 generated states, 72 distinct states, depth 7 | every registered obligation passes | accepted |
| repository gates | all-feature and no-default-feature tests; strict Clippy and rustdoc; documentation lint; 61-diagram reproducibility; diff check; pgmcp bug gate | every gate passes | accepted |

The generated threshold property found an affordable empty-side defect:
`indel_distance_bounded("a", "", 1)` returned `None`. The implementation now
handles either empty side before the interior band loop; the minimized seed
`7cf34e99389c421981029b46abaaa29f6948796f5bf3b2a44b8b8903010546de`
is committed in `tests/proptest_class_a_presets.proptest-regressions`.

Captured command evidence remains under `/tmp/liblevenshtein_phase8_*` until
the complete multi-phase implementation run finishes.

## 4. Repository-gate record

| Gate | Result |
|---|---|
| `cargo test --all-features` | accepted: 3,844 library tests plus all integration/property suites; 269 doctests passed and 364 were intentionally ignored |
| `cargo test --no-default-features` | accepted: 1,077 library tests plus all applicable integration/property suites; 173 doctests passed and 121 were intentionally ignored |
| Class-A generated properties | accepted: 20,000 cases across ten properties |
| complete Birkbeck validation | accepted: 42,395 source pairs, including every applicable Hamming and bounded-skip pair |
| `scripts/verify-formal.sh all` | accepted across the registered Rocq, Dafny, Verus, SMT, and TLA+ artifacts |
| strict Clippy and rustdoc | accepted with warnings denied |
| documentation and diagrams | accepted: 0 math-lint violations in 281 living documents; all 61 SVGs reproducible |
| `git diff --check` | accepted |
| `pgmcp bug-gate` | accepted: no open bugs anchored to 150 changed files |

The public deliverable is therefore the validated Class-A preset surface and
its independent reference distances. Dedicated Hamming and indel dictionary
walkers remain benchmark-only because their structural reduction missed the
pre-registered compound gate even though runtime improved.
