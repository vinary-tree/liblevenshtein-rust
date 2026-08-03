---
title: PositionKind seam zero-cost and compatibility gate
date: 2026-08-01
project: liblevenshtein-rust
kind: feature_refactor
status: decided
verdict: quantitative_zero_cost_accepted_exact_byte_identity_rejected
plan: /home/dylon/.claude/plans/liblevenshtein-currently-supports-levens-generic-sunrise.md
root_epic: extending-liblevenshtein-automaton-families-4bb97598
work_item: phase-5-the-positionkind-seam-seam-a-be391046
---

# `PositionKind` seam zero-cost and compatibility gate

## 1. Pre-registration provenance

This entry freezes Phase 5's hypotheses, measurement procedure, acceptance
rules, and reporting schema **before** the pre-change timing baseline is run.
The implementation plan pre-registered the quantitative thresholds on
2026-07-25. This ledger makes the aggregation rule explicit without weakening
those thresholds.

The intervention replaces per-position runtime `Algorithm` branching by one
runtime dispatch per dictionary edge and then monomorphized
`AutomatonVariant` calls. A **dictionary edge** is one labelled transition in
the indexed dictionary. A **position** is one dynamic-programming frontier
representative. `PositionKind` identifies the continuation language of a
representative; `aux` is an eight-bit variant payload reserved for later
history-dependent automata.

No post-change result exists when this section is written.

## 2. Hypotheses

Let `$`T_{s,b}^{\mathrm{before}}`$` and
`$`T_{s,b}^{\mathrm{after}}`$` be Criterion's mean time estimates for benchmark
case `$`b`$` in suite `$`s`$`. Define the relative change
`$`r_{s,b}`$` and the suite mean `$`\bar r_s`$` by

```math
r_{s,b}=\frac{T_{s,b}^{\mathrm{after}}}
              {T_{s,b}^{\mathrm{before}}}-1,
\qquad
\bar r_s=\frac{1}{|B_s|}\sum_{b\in B_s}r_{s,b}.
```

Criterion supplies a two-sided 95% confidence interval for every
`$`r_{s,b}`$`. Let `$`u_{s,b}`$` denote its upper endpoint. The conservative
suite upper bound is

```math
\bar u_s=\frac{1}{|B_s|}\sum_{b\in B_s}u_{s,b}.
```

This average of marginal upper bounds is deliberately conservative; it avoids
assuming that benchmark cases are independent merely to obtain a narrower
suite interval.

| ID | Pre-registered hypothesis | Acceptance rule |
|---|---|---|
| `position-kind-001-semantics` | The refactor preserves every existing `Standard`, OSA-transposition, and merge/split result, distance, result order, and public query surface. | All existing unit, differential, property, integration, corpus, example, and doctest gates pass without changing expected outputs. |
| `position-kind-002-layout` | Replacing `bool` plus padding by `PositionKind` and `aux` does not widen `Position`. | A compile-time assertion proves `size_of::<Position>() == 24` on the supported 64-bit release target; ordering distinguishes `(term_index, num_errors, kind, aux)`. |
| `position-kind-003-zero-cost` | Monomorphized variant dispatch is indistinguishable from the current `Algorithm::Standard` hot path. | For every one of the six suites below, `$`\bar r_s<0.015`$` and `$`\bar u_s<0.03`$`. |
| `position-kind-004-codegen` | The standard transition specialization introduces no dynamic variant branch inside the per-position kernel. | The normalized pre/post disassembly is byte-identical. The sole pre-authorized exception is one `cmov` caused by `saturating_add` replacing `add`; any other difference rejects the stronger claim and requires analysis. |
| `position-kind-005-laws` | Variant-specific pruning never changes the retained minimum continuation cost. | Formal decision-partition and subsumption obligations pass; the same invariants are exercised by property tests over generated positions and suffixes. |

The six suites `$`s`$` are `benchmarks`, `subsumption_benchmarks`,
`batch2a_subsumption_benchmarks`, `state_operations_benchmarks`,
`query_iterator_benchmarks`, and `transition_benchmarks`.

## 3. Frozen measurement protocol

1. Confirm that CPU 0 is in the process affinity set and every visible CPU uses
   the `performance` governor.
2. Use the same checkout, `rustc`, dependencies, feature set, and
   `RUSTFLAGS="-C target-cpu=native -C opt-level=3"` for both arms.
3. Pin every Criterion process to CPU 0 with `taskset -c 0`.
4. Run Criterion's normal warm-up, measurement, outlier analysis, bootstrap,
   and 95% confidence calculations; do not discard outliers manually.
5. Save the current arm as `phase5-before` and compare the changed arm as
   `phase5-after` using Criterion's persisted estimates.
6. Generate and compare a normalized disassembly witness for the exact standard
   transition probe with `scripts/check-unit-cost-zero-cost.sh`.
7. Record all commands, compiler identity, CPU identity, point estimates,
   confidence intervals, exclusions, and any environmental caveat here.

No timing threshold may be changed after step 3 starts. A noisy or interrupted
run is invalidated in full and rerun; individual inconvenient cases are never
selected out.

## 4. Literate benchmark algorithm

The procedure below is both an experimental narrative and executable design.
The prose states why each step exists; the pseudocode states its exact order.

```text
MEASURE-SEAM(before-tree, after-tree, suites):
    # Establish one thermally and scheduler-comparable execution lane.
    require cpu 0 is allowed
    require every visible governor equals "performance"

    for suite in suites:
        pin Criterion(before-tree, suite, save = "phase5-before") to cpu 0
        pin Criterion(after-tree, suite, compare = "phase5-before") to cpu 0

        # Preserve every registered case; aggregate only after Criterion has
        # produced its per-case bootstrap estimate and confidence interval.
        changes := read all case-relative estimates for suite
        require mean(point(changes)) < 0.015
        require mean(upper95(changes)) < 0.03

    require normalized_standard_disassembly(before-tree)
            == normalized_standard_disassembly(after-tree)
    return ACCEPT
```

## 5. Integrity, security, and reproducibility controls

- Benchmark names and persisted Criterion paths are treated as untrusted file
  names by analysis scripts: traversal components and missing JSON fields are
  rejected rather than interpolated into shell commands.
- The probe consumes bounded scalar inputs and constructs its own fixed-size
  characteristic vector; it accepts no raw pointer from an external caller.
- `PositionKind` and `aux` remain private fields. Constructors establish the
  invariant that normal positions carry `aux == 0`; accessors expose values
  without permitting invalid state mutation.
- Integer offset arithmetic remains checked. Overflow produces no successor,
  matching the current total semantics.
- Raw Criterion data and captured logs are evidence. Documentation reports may
  summarize them but may not replace or silently rewrite them.

## 6. Evidence table

The pre-registered protocol and thresholds above were not changed after the
baseline began. The table records the completed experiment rather than replacing
its raw Criterion, compiler, and formal-verification artifacts.

| Evidence | Before | After | Verdict |
|---|---:|---:|---|
| `size_of::<Position>()` on the 64-bit release target | 24 bytes | 24 bytes | accepted |
| semantic/property/integration suite | green | 3,817 library tests, integration suites, example targets, and 267 doctests green | accepted |
| six pinned Criterion suites | frozen `phase5-before` arm | 423 cases; all six aggregate gates pass | accepted |
| normalized standard disassembly | 1,073 bytes | 1,583 bytes | **rejected**: not byte-identical |
| Rocq / Verus / Z3 / cvc5 / TLA+ | n/a | every registered obligation green | accepted |
| docs math lint / diagram reproducibility | green | green | accepted |

## 7. Execution environment

The experiment ran on an AMD Ryzen Threadripper PRO 5975WX 32-Cores with one
hardware thread per core exposed to the benchmark process. CPU 0 was in the
affinity set and every visible CPU used the `performance` governor. Both arms
used `rustc 1.97.1 (8bab26f4f 2026-07-14)`, LLVM 22.1.6, Criterion 0.8.2, all
crate features, and
`RUSTFLAGS="-C target-cpu=native -C opt-level=3"`. Every Criterion command was
pinned to CPU 0 and used 100 samples.

The before arm was built from the frozen, exact pre-refactor source tree rather
than reconstructed from memory. The after arm used the same dependency trees,
compiler, flags, benchmark registrations, and CPU policy. All 423 registered
cases were retained; no outlier, suite, or inconvenient case was removed.

## 8. Quantitative results

Negative values are improvements. “Upper 95%” is `$`\bar u_s`$`, the
pre-registered conservative mean of Criterion's per-case upper confidence
endpoints, not a post-hoc interval over the suite mean.

| Suite | Cases | Mean change `$`\bar r_s`$` | Conservative upper 95% `$`\bar u_s`$` | Gate |
|---|---:|---:|---:|---|
| `benchmarks` | 36 | -0.499% | -0.020% | pass |
| `subsumption_benchmarks` | 144 | -4.639% | -4.313% | pass |
| `batch2a_subsumption_benchmarks` | 8 | -0.890% | -0.635% | pass |
| `state_operations_benchmarks` | 85 | -7.814% | -7.230% | pass |
| `query_iterator_benchmarks` | 43 | +0.175% | +0.641% | pass |
| `transition_benchmarks` | 107 | -6.017% | -5.196% | pass |

Thus every suite satisfies both `$`\bar r_s<0.015`$` and
`$`\bar u_s<0.03`$`. Hypothesis `position-kind-003-zero-cost` is accepted under
its pre-registered quantitative meaning.

### 8.1 Ownership-boundary amendment discovered by measurement

The thresholds did not change, but the first two implementations failed them:

1. assigning an owned successor vector into a caller-owned buffer regressed the
   transition suite by 32.125%;
2. filling a generic caller-owned buffer even for the one-position public API
   reduced that regression to 9.185%, but still failed;
3. retaining caller-owned buffers for repeated state loops while letting the
   one-position API return an owned aggregate produced the final -6.017% result.

The final boundary follows the actual ownership frequency. A whole-state edge
selects one static variant and reuses one empty `SmallVec` across position calls.
The public one-position operation performs its one closed runtime match and
returns an owned aggregate, allowing Rust's return-value optimization to avoid
the generic output-parameter penalty observed in the failed arms. All six suites
were measured again against the frozen baseline after this change.

### 8.2 Exact-byte hypothesis rejection

The normalized pre-change probe was 1,073 bytes with SHA-256
`cfaa1cc023ea1cc9cfdc4cd5050fa2e1041568a65032d074556b4da2ca36bd57`.
The final probe was 1,583 bytes with SHA-256
`167a7285c89812c3530145fcd44b31ff68021cfa9fde7859ed2aa8f9d4c1c61a`.
The difference is larger than the sole pre-authorized `cmov`, so
`position-kind-004-codegen` is rejected.

Analysis found that the exact-byte probe did not isolate runtime selector cost:
it also captured the intentional initialization of both `PositionKind` and
`aux` where the frozen representation initialized one Boolean, plus the final
owned-return boundary. Exact machine-code identity is therefore false and is
not claimed. The mismatch is tracked in pgmcp as root-epic bug
`phase-5-exact-byte-codegen-gate-conflates-dispatch-cost-with-the-intentional-position-payload-change-daeb31`.
The script remains strict so future runs continue to expose, rather than hide,
the rejected stronger hypothesis.

The resulting pgmcp bug was repaired without rewriting this historical result.
`scripts/check-unit-cost-zero-cost.sh audit LABEL` now emits optimized LLVM IR
for the same constant-Standard probe and checks the dispatch-specific facts the
byte comparison failed to isolate: the probe contains `transition_standard`
inlining provenance, contains no runtime selector or non-Standard leaf
provenance, and contains no surviving LLVM `switch`. This structural audit
passes. It is a new corroborating observation, not a retroactive pass for
`position-kind-004-codegen`.

This rejection does not override the distinct quantitative hypothesis: the
six-suite rule was the pre-registered acceptance criterion for runtime cost and
passed without exclusions. Rocq, Verus, SMT, and TLA+ separately establish the
selector and decision-partition invariants; they do not purport to prove ELF
byte identity.

## 9. Verification results and decision

The executable test mapping mirrors the formal invariants:

- Rocq proves representation validity, full-key injectivity, selector
  equivalence, continuation separation, and error-order obligations without
  admitted axioms.
- Verus proves the Rust-facing key, payload, selector, and variant contracts.
- Z3 and cvc5 independently find no counterexample to the same first-order
  obligations.
- TLC explores the 12-state closed selector model and satisfies all four
  invariants; deadlock checking is disabled only because the finite model has
  an intentional terminal state.
- Optimized LLVM IR contains only the Standard leaf for the constant-Standard
  probe and no selector `switch`; this isolates dispatch erasure while the
  historical byte mismatch remains visible.
- Proptest runs 2,000 generated cases per property for subsumption, ordering,
  deterministic transitions, budget preservation, and legal continuation
  kinds. Existing cross-validation properties retain all public-result checks.
- Unit, integration, example, and documentation tests pass with all features.

Accordingly, hypotheses `position-kind-001-semantics`,
`position-kind-002-layout`, `position-kind-003-zero-cost`, and
`position-kind-005-laws` are accepted. The stronger
`position-kind-004-codegen` hypothesis is rejected and retained as a visible
negative result. Phase 5 is accepted because its semantic, layout, formal, and
pre-registered quantitative gates pass; acceptance carries no claim of exact
machine-code identity.

## 10. References

- Schulz, K. U., and Mihov, S. “Fast string correction with Levenshtein
  automata.” *International Journal on Document Analysis and Recognition* 5,
  67–85 (2002). [DOI 10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8).
- Criterion.rs supplies the benchmark estimates, bootstrap confidence
  intervals, and persisted comparison artifacts used by this protocol.
  [Criterion.rs documentation](https://bheisler.github.io/criterion.rs/book/).
