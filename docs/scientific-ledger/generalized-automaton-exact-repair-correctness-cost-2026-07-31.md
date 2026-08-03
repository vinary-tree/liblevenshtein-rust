---
title: Generalized Automaton Exact-Repair Correctness Cost
date: 2026-07-31
pgmcp_experiment: generalized-automaton-exact-repair-correctness-cost
pgmcp_root: extending-liblevenshtein-automaton-families-4bb97598
scope: Phase 2 prime generalized automaton repair
---

# Generalized Automaton Exact-Repair Correctness Cost

This append-only ledger records the performance cost of replacing the
generalized automaton's truncating floating-point accumulator with exact scaled
integer arithmetic. It also records the operation-driven acceptance and
subsumption repairs that were measured with it. Correctness is the shipping
criterion; the experiment asks whether exactness imposes a material runtime
penalty and establishes an auditable baseline for future optimization.

The experiment and all 120 primary observations are stored in pgmcp under
`generalized-automaton-exact-repair-correctness-cost`. Experiment `154` was an
empty, superseded registration made before the criterion schema was known; it
contains no measurements. Experiment `155` is the preregistered record of
authority.

## Definitions and implementation under test

Let an operation weight be a finite decimal number `w`, let `S` be the least
common multiple of the weights' decimal denominators, and let `C` be the
accumulated scaled cost. The treatment implements:

```math
C' = C + \operatorname{round}(S w),
\qquad
c = \frac{C}{S}.
```

`CostScale` rejects non-finite, negative, inexact, and overflowing conversions.
Consequently, the equality and ordering used by the state antichain are exact
integer relations. The repair also charges suffix completion through operations
from the configured `OperationSet`; it does not synthesize unit insertion or
deletion costs. The standard Levenshtein subsumption offset theorem is enabled
only for the exact standard operation set. Other sets conservatively prune only
equal-control states with a dominated cost.

The following flow shows what differs between the control and treatment:

```text
Operation weight
    │
    ├── control: truncate `f64` to `usize` ──► accidental zero-cost edges
    │
    └── treatment: exact rational scaling ──► checked integer cost
                                                  │
                                                  ▼
                          operation-driven completion and antichain pruning
```

## Preregistered hypothesis and protocol

| Field | Preregistered value |
| --- | --- |
| Primary scenario | `state_transition/scenarios/one_error` |
| Control | Git archive of `3f362b44012184a222a64d7f95af541fac93f748` |
| Treatment | Phase 2-prime working tree |
| Expected direction | Treatment slower, because more exact states may survive |
| Statistical test | Welch independent-samples test, one-sided greater, `alpha = 0.05` |
| Minimum effect | Cohen's `d >= 0.5` |
| Practical envelope | Two one-sided equivalence tests within `-100%..+100%` |
| Samples | 60 observations per arm after 3 s warm-up |
| Measurement | Criterion, 8 s measurement, plotting disabled |
| Isolation | Distinct target directories and identical compiler flags |
| CPU | CPU 0 pinned with `taskset`; performance governor |

The two preregistered criteria form a conjunction: the expected slowdown must
be statistically and practically detectable, while still remaining inside a
two-times latency envelope. This unusual direction is deliberate: the plan
predicted a correctness-related regression, so observing a speedup must reject
that directional hypothesis rather than silently rewriting it after seeing the
data.

Both arms used `CARGO_NET_OFFLINE=true` and
`RUSTFLAGS="-C target-cpu=native -C opt-level=3"` on an AMD Ryzen Threadripper
PRO 5975WX under Linux 7.1.5. Repeated samples and explicit uncertainty follow
Kalibera and Jones's systems-benchmarking methodology
([DOI 10.1145/2464157.2464160](https://doi.org/10.1145/2464157.2464160)).

## Primary result

| Arm | Mean (ns) | Median (ns) | Standard deviation (ns) | Samples |
| --- | ---: | ---: | ---: | ---: |
| Control | 2,300.123 | 2,299.684 | 9.414 | 60 |
| Exact repair | 1,975.514 | 1,968.610 | 13.264 | 60 |

The treatment was **14.1% faster**, contrary to the preregistered expected
direction. pgmcp therefore correctly rejected the directional hypothesis:

| Statistic | pgmcp result |
| --- | ---: |
| Welch statistic | -154.5888 |
| One-sided greater `p` | 1.0 |
| Cohen's `d` | -28.2239 |
| Mean-difference 95% interval (ns) | [-328.7715, -320.4457] |
| Equivalence-envelope `p` | 0.0, passed |
| Mann-Whitney `p` | 0.0 |
| Cliff's delta | -1.0 |

The result rejects only the predicted *slowdown*. It does not reject the exact
repair, whose correctness obligations are independently established by tests
and formal proofs. The likely explanation is that checked scaled costs prevent
incorrect zero-cost paths from inflating the active-state set on this scenario;
this is an inference, not a separately isolated causal measurement.

## Exploratory breadth sweep

After the primary decision was frozen, an explicitly exploratory 32-case sweep
measured other benchmark groups. Each row reports treatment versus the archived
control. These observations are useful for profiling, but they do not change
the preregistered decision.

| Scenario | Control (ns) | Treatment (ns) | Change |
| --- | ---: | ---: | ---: |
| operation/delete | 424.286 | 707.695 | +66.80% |
| operation/insert | 563.690 | 892.919 | +58.41% |
| operation/match | 450.747 | 860.057 | +90.81% |
| operation/substitute | 516.318 | 762.780 | +47.73% |
| realistic/color-colour | 932.685 | 1,416.848 | +51.91% |
| realistic/definitely | 2,359.984 | 2,299.578 | -2.56% |
| realistic/gray | 801.112 | 1,058.935 | +32.18% |
| realistic/organize | 1,464.822 | 2,027.390 | +38.41% |
| realistic/theater | 1,306.418 | 1,772.089 | +35.64% |
| distance/0 | 459.288 | 602.044 | +31.08% |
| distance/1 | 1,043.419 | 1,208.420 | +15.81% |
| distance/2 | 2,304.109 | 1,970.359 | -14.49% |
| distance/3 | 3,192.897 | 2,748.498 | -13.92% |
| input-length/3 | 388.616 | 780.330 | +100.80% |
| input-length/5 | 788.428 | 1,313.303 | +66.57% |
| input-length/8 | 2,120.811 | 1,848.669 | -12.83% |
| input-length/12 | 2,352.340 | 2,016.021 | -14.30% |
| input-length/15 | 2,363.457 | 2,030.524 | -14.09% |
| scenario/exact | 1,411.056 | 2,261.877 | +60.30% |
| scenario/one-error | 2,376.436 | 2,004.999 | -15.63% |
| scenario/reject-three | 2,174.996 | 1,214.600 | -44.16% |
| scenario/two-errors | 2,660.179 | 1,775.358 | -33.26% |
| word-scale/5 | 649.176 | 1,232.354 | +89.83% |
| word-scale/10 | 1,459.252 | 2,455.737 | +68.29% |
| word-scale/15 | 2,291.403 | 3,679.278 | +60.57% |
| word-scale/20 | 3,093.568 | 4,752.799 | +53.63% |
| subsumption/exact-1 | 1,136.543 | 1,644.285 | +44.67% |
| subsumption/exact-2 | 1,588.408 | 2,479.197 | +56.08% |
| subsumption/exact-3 | 1,703.186 | 3,265.501 | +91.73% |
| maximum-state/1 | 344.831 | 571.657 | +65.78% |
| maximum-state/2 | 1,769.476 | 782.118 | -55.80% |
| maximum-state/3 | 5,092.801 | 1,047.738 | -79.43% |

Eleven cases improved and 21 regressed. The arithmetic mean change was
`+28.96%`; the geometric mean change was `+17.41%`. The best case improved by
79.43%, while the worst case was approximately 2.008 times the control. This
heterogeneity confirms that future optimization must target measured state-set
and operation-dispatch costs, not weaken exact arithmetic or operation-driven
semantics.

## Correctness and verification evidence

| Layer | Evidence |
| --- | --- |
| Example tests | Fractional boundary: six operations of weight `0.15` accepted under budget 1; seven rejected |
| Unit tests | Exact rescaling, finite/infinite empty-side rates, unsupported arities, operation-driven completion |
| Property tests | 2,000 cases each for operation-order minimum, antichain idempotence, empty-side rates, Hamming no-deletion behavior |
| Integration | Public generalized-state and operation-rate suites exercise the exported surface |
| Rocq | Completion charging, exact rescaling, conservative/certified subsumption, antichain and rate invariants |
| Verus | 16 verification conditions discharged |
| SMT | 12 negated obligations are `unsat` in both Z3 and cvc5 |

The formal invariants are intentionally mirrored as property tests. The proof
artifacts establish the bounded mathematical model; the randomized tests verify
that the Rust representation and operation ordering refine that model.

## Decision and retained work

| ID | Verdict | Justification |
| --- | --- | --- |
| `P2P-CORRECT` | retained | Fractional weights are no longer truncated and completion is derived from the operation set. |
| `P2P-PERF` | retained with profiling evidence | The primary treatment was faster and remained inside the preregistered envelope; exploratory regressions identify optimization targets without invalidating correctness. |
| `P2P-SUBSUME` | retained | Classical offset pruning is certified only for the standard set; conservative dominance is sound for arbitrary sets. |
| `P2P-ARITY` | retained | Unsupported operation geometries return a typed error instead of being silently misinterpreted. |

## Reproduction

The registered benchmark is `generalized_automaton_benchmarks`. A comparable
local run is:

```console
CARGO_NET_OFFLINE=true RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
  taskset -c 0 cargo bench --bench generalized_automaton_benchmarks -- \
  state_transition/scenarios/one_error --warm-up-time 3 \
  --measurement-time 8 --sample-size 60 --noplot
```

Raw primary observations, arm metadata, frozen criteria, and the server-side
decision remain in pgmcp. The repository retains only this compact scientific
record and the benchmark source; transient build trees and logs are not source
artifacts.
