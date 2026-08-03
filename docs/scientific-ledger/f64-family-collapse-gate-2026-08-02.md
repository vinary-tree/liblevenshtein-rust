# Phase 11: weighted-position collapse gate

Status: fallback implemented; full-suite verification pending.

## Pre-registered decision

The expanded plan made the `_f64` collapse optional. Its gate was defined
before this implementation work:

| ID | Hypothesis | Decision rule |
|---|---|---|
| F64-H1 | `Position` and `PositionF64` can share one representation without changing behavior or the integer hot-path layout. | Pilot only the two position modules. Continue into state, transition, pool, intersection, and query code only if the pilot preserves behavior under the complete suite. |
| F64-H2 | If F64-H1 fails, sharing only the subsumption relation removes meaningful drift at a small blast radius. | Retain both representations; factor the Standard/OSA/MergeSplit decision tree through one carrier-generic helper; prove and property-test equivalence to both legacy formulas. |

## Evidence

The pilot found four load-bearing differences:

| Dimension | Integer position | Weighted position | Consequence |
|---|---|---|---|
| continuation state | six typed `PositionKind` values plus an `aux` byte | one Boolean special flag | one representation would erase Damerau delta and affine-layer state or add unused weighted state |
| carrier/API | public `num_errors: usize` | public `accumulated_cost: f64` | a generic public field would be a breaking rename and would not preserve both APIs |
| memory contract | 24-byte compile-time assertion on 64-bit targets | floating carrier with different padding and ordering | a generic payload risks widening the integer transition hot path |
| semantics | exact integer comparison; true-Damerau and affine variants | epsilon comparison; weighted realignment; no true-Damerau state | generic storage does not imply one lawful transition or subsumption contract |

These are semantic differences, not duplicated syntax. F64-H1 is therefore
rejected before changing either representation.

The fallback adds `src/cost/subsumption.rs`. `subsumes_with<M>` owns every
shared variant-state branch. An internal `SubsumptionCost: CostMonoid`
extension retains the exact carrier-specific comparisons; subtraction and
index scaling do not pollute the public monoid contract.

## Verification results

| Evidence | Result |
|---|---|
| generated unit-cost equivalence property | passed |
| generated weighted epsilon-equivalence property | passed |
| focused Rust run | 2 passed, 0 failed |
| Rocq generic-carrier equivalence | passed |
| Verus Rust-shaped exact-unit model | 2 verified, 0 errors |
| Z3 unit-integer and weighted-real counterexample search | 2 `unsat` |
| cvc5 unit-integer and weighted-real counterexample search | 2 `unsat` |

## Verdict

| ID | Verdict | Justification |
|---|---|---|
| F64-H1 | rejected | The representations encode different state machines, public APIs, layouts, and lawful arithmetic. A whole-family collapse would be a redesign rather than deduplication. |
| F64-H2 | accepted, subject to the full-suite gate | The shared helper removes the structurally duplicated decision tree while property and formal models preserve the two machine formulas exactly. |

No state, transition, pool, intersection, or query family is collapsed. That
work remains outside the accepted gate unless future evidence supplies a
representation that preserves all four differences above.
