# Generalized-automaton repair

**Status:** implemented · **Behavior change:** fractional weights and missing
empty-side operations now affect acceptance exactly

This document specifies the repaired `GeneralizedAutomaton`. The public engine
is an exact, operation-driven automaton for runtime `OperationSet` values. It
binds one finite source and advances an unknown-length target stream through a
finite-lookback row ring; the former sparse alignment graph remains a
test-only independent oracle. It never asks a unit-cost universal-position
state whether a configured weighted alignment should accept.

![The generalized alignment graph contains only configured consuming operations; production evaluates it a target generation at a time through a finite-lookback row ring.](../diagrams/automata/generalized-operation-grid.svg)

*Each edge consumes the operation's declared source and target lengths. The
production row ring and independent sparse oracle compute the same least exact
scaled cost for every coordinate pair.*

## 1. Defect and required semantics

An operation $`t`$ is the tuple:

```math
t=\langle t^x,t^y,t^w\rangle,
```

where $`t^x`$ characters are consumed from the dictionary word, $`t^y`$
characters are consumed from the input, and $`t^w`$ is a non-negative weight.
A restricted operation additionally names the allowed source/target string
pairs.

The previous public acceptance path had two independent defects:

1. Empty input, trailing word characters, and long input used ordinary
   Levenshtein assumptions even when the operation set contained no deletion
   or insertion.
2. `weight.trunc()` made every weight in $`[0,1)`$ free. Seven `0.15`
   substitutions therefore cost zero instead of `1.05`.

These are correctness defects, not merely precision choices. For the Hamming
operation set—match plus substitution—there is no path between unequal-length
strings at any budget. For a phonetic operation of weight `0.15`, six uses fit
budget `1`, while seven do not.

## 2. Alignment graph and finite-lookback online evaluation

For word $`w`$ of length $`m`$ and input $`x`$ of length $`n`$, an **alignment
cell** $`(i,j)`$ means that the first $`i`$ source characters and first $`j`$
target characters have been consumed. Cell $`(0,0)`$ has cost zero. Operation
$`t`$ creates an edge:

```math
(i,j)\longrightarrow(i+t^x,j+t^y)
```

when the destination remains inside the grid and the corresponding string
slices satisfy the operation restriction. Its edge cost is the exact scaled
integer representation of $`t^w`$.

Every admitted operation must consume at least one character. Coordinates
therefore strictly increase in lexicographic order. Let $`r`$ be the maximum
target consumption of any configured operation. A cell in target generation
$`j`$ can read only generations $`j-r`$ through $`j`$. Production keeps
exactly $`r+1`$ committed source-width rows, one scratch row, and the last
$`r`$ target scalars. Generation tags distinguish a live predecessor row
from an overwritten ring slot.

Source-only operations have target consumption zero. The source coordinate is
evaluated in increasing order inside the scratch row, so those predecessors
are already final when read. Target-consuming operations read a tagged earlier
row. This is a topological schedule of the same acyclic alignment graph; no
heuristic acceptance completion or empty-string special case is needed.

The bounded recurrence is:

```math
D[i+t^x,j+t^y]
=\min\left(D[i+t^x,j+t^y],D[i,j]+\operatorname{scaled}(t^w)\right).
```

An update is retained only when its cost is at most the scaled public budget.
Acceptance is equivalent to reaching $`(m,n)`$. `advance` is transactional:
it builds the prospective target window and row in scratch, and commits the
generation only after validation and checked arithmetic succeed. An error
therefore leaves the previous observation intact.

The test configuration retains the earlier `BTreeMap<(usize, usize), usize>`
evaluator as an independent sparse-grid oracle. Exhaustive small Unicode
examples compare every production online result to that differently shaped
implementation.

## 3. Exact cost domain

One `CostScale` is derived from the complete operation set. The integer budget
is:

```math
K=\texttt{max\_distance}\cdot\texttt{scale.denominator()}.
```

Every operation cost is converted through the same scale before traversal.
NaN, infinity, an unrepresentable denominator, or arithmetic overflow is a
`GeneralizedAutomatonError`. `try_with_operations`, `cost_scale`,
`scaled_distance`, and `try_accepts` preserve that error. The legacy Boolean
`accepts` method fails closed by returning `false`.

The public budget remains an integer for compatibility. `scaled_distance`
exposes the exact numerator; callers recover a presentation value with the
automaton's `CostScale::from_scaled`.

## 4. Empty-side rates and why they are not a global length bound

For pure deletion and insertion operations, the useful per-character rates
are:

```math
\rho_{\mathrm{del}}
=\min_{t^y=0,t^x>0}\frac{t^w}{t^x},
\qquad
\rho_{\mathrm{ins}}
=\min_{t^x=0,t^y>0}\frac{t^w}{t^y}.
```

An absent set has infinite rate. This derivation explains why Hamming rejects
either non-empty/empty pairing and why trailing characters cannot be completed
with an operation that was never configured.

`OperationSet::rho_del` and `OperationSet::rho_ins` expose these quantities as
reduced exact rationals in `EmptySideRate::Finite`, or as the explicit
`EmptySideRate::Infinite` variant. `fits_budget` performs cross multiplication;
`max_consumable` implements the corresponding floor bound without floating
point arithmetic.

Those rates are safe lower bounds for *pure* empty-side completion; they are
not a complete preflight length rule for arbitrary generalized operations. A
$`\langle1,2,w\rangle`$ operation can expand the target without being a pure
insertion. Applying $`\rho_{\mathrm{ins}}`$ as a universal input-length ceiling
would reject a valid split. The repaired implementation therefore lets the
exact graph decide instead of relying on a lossy length heuristic.

## 5. Operation restrictions and Unicode

Consumption counts Unicode scalar values. `OperationType::can_apply_str`
checks scalar counts, then uses the existing UTF-8 restriction table for exact
string-pair lookup. The byte-oriented `can_apply` remains available for byte
callers. This distinction is required for a one-character rule such as
`"é" → "e"`: the source is one scalar but two UTF-8 bytes.

The built-in operation named `transpose` has adjacent-reversal semantics: its
two source scalars must equal the two target scalars in reverse order. Other
unrestricted non-zero operations accept any slices of their declared sizes;
restricted operations consult their pair set.

## 6. Compatibility boundary

`GeneralizedPosition`, `GeneralizedState`, and
`GeneralizedTransitionInput` remain public because earlier releases exposed
them. This bounded streaming representation supports single-scalar rules and
the historical $`\langle2,2\rangle`$, $`\langle2,1\rangle`$, and
$`\langle1,2\rangle`$ intermediates. It carries a `CostScale`, rescales the
complete state through a checked least common denominator, charges a
multi-scalar rule only after its complete target slice is known, and keeps the
least cost among all applicable rules independent of insertion order.

The classical offset/slack subsumption theorem is enabled only when the
complete operation set is exactly unrestricted standard Levenshtein. An
integer scale is not enough: a denominator-one operation may cost two units.
Every other operation lattice uses conservative dominance, which removes only
an identical control position at a strictly greater exact cost.

`GeneralizedState::try_transition` returns `UnsupportedOperationArity` for a
rule its finite intermediate-state vocabulary cannot encode. The infallible
`transition` wrapper fails closed. `GeneralizedAutomaton` is the
operation-complete API: its alignment graph accepts every non-zero source and
target consumption pair, so arbitrary arities are never silently ignored.

## 7. Complexity and resource policy

Let $`m`$ be the fixed source length, $`n`$ the consumed target length,
$`r`$ the maximum target consumption, and $`\lvert\mathcal{O}\rvert`$ the
number of operations. Excluding the cost of a restriction-table lookup, time is:

```math
\mathcal{O}\left(n(m+1)\lvert\mathcal{O}\rvert\right).
```

The DP ring and scratch row retain:

```math
(r+2)(m+1)
```

exact scaled-cost cells, plus the fixed source, compiled operations, Unicode
offsets, and at most $`r`$ target scalars. Memory is therefore independent of
$`n`$. `GeneralizedOnlineLimits::max_retained_cells` preflights the displayed
cell count, while `max_step_work_units` preflights
$`(m+1)\lvert\mathcal{O}\rvert`$ before the first target transition. Defaults
are one million retained cells and one hundred million relaxations per step.

Every coordinate and cost addition is checked. `try_reserve_exact` precedes
ring, scratch, operation, and lookback allocation. A resource or arithmetic
error is explicit, and a rejected target step does not commit its prospective
window or generation.

## 8. API

```rust
use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::{OperationSetBuilder, OperationType};

let operations = OperationSetBuilder::new()
    .with_match()
    .with_operation(OperationType::new(1, 1, 0.15, "cheap substitution"))
    .build();
let oracle = GeneralizedAutomaton::try_with_operations(1, operations)?;

assert!(oracle.accepts("aaaaaa", "bbbbbb"));
assert!(!oracle.accepts("aaaaaaa", "bbbbbbb"));
assert_eq!(oracle.cost_scale()?.denominator(), 20);

let mut online = oracle.online("aaaaaa")?;
for target in "bbbbbb".chars() {
    online.advance(target)?;
}
assert_eq!(online.observation().distance_within_budget, Some(18));
# Ok::<(), Box<dyn std::error::Error>>(())
```

## 9. Verification and tests

| Evidence | Invariant |
|---|---|
| Rocq `GeneralizedAutomatonRepair.v` | path folds, completion charging, exact rescaling, explicit-infinity and finite rates, certified/conservative subsumption, discovery guarding, operation-order independence, antichain set semantics, Hamming length preservation, absent-deletion rejection, coordinate progress |
| Verus `generalized_automaton.rs` | positive completion charging, exact rescaling, exact rate comparison, certified/conservative subsumption, discovery guarding, minimum-cost order independence, equal-position insertion idempotence, checked accumulation, `0.15` boundary, finite-lookback predecessor retention, and topological source-only ordering |
| Z3 + cvc5 `generalized_automaton.smt2` | thirteen bounded counterexample queries, including finite-lookback/topological scheduling, are UNSAT in both solvers |
| `proptest_generalized_automaton_repair.rs` | standard/reference, Hamming, indel/LCS, bounded-skip/subsequence, budget monotonicity, fractional, Unicode, invalid values |
| generalized unit suite | standard, transpose, merge/split, phonetic, empty-side, exact-weight examples, exhaustive online-versus-sparse Unicode correspondence, transactional limits, and a 100,000-scalar prefix-retention gate |

The generalized operation model follows Mitankin, Mihov, and Schulz,
“Deciding word neighborhood with universal neighborhood automata,”
*Theoretical Computer Science* 412(22), 2011,
[doi:10.1016/j.tcs.2011.01.013](https://doi.org/10.1016/j.tcs.2011.01.013).
