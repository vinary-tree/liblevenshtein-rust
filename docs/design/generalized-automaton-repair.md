# Generalized-automaton repair

**Status:** implemented · **Behavior change:** fractional weights and missing
empty-side operations now affect acceptance exactly

This document specifies the repaired `GeneralizedAutomaton`. The public engine
is an exact, operation-driven oracle for runtime `OperationSet` values. It no
longer asks a unit-cost universal-position state whether a configured weighted
alignment should accept.

![The generalized automaton explores only exact-cost alignment cells reachable within the scaled budget.](../diagrams/automata/generalized-operation-grid.svg)

*Each edge consumes the operation's declared source and target lengths. A cell
stores the least exact scaled cost seen for that coordinate pair.*

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

## 2. Alignment graph

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
therefore strictly increase in lexicographic order: a `BTreeMap<(usize,
usize), usize>` is both the sparse frontier and a topological work queue. When
a cell is popped, every predecessor has already contributed to its minimum.
There is no heuristic acceptance completion and no special case for either
empty string.

The bounded recurrence is:

```math
D[i+t^x,j+t^y]
=\min\left(D[i+t^x,j+t^y],D[i,j]+\operatorname{scaled}(t^w)\right).
```

An update is retained only when its cost is at most the scaled public budget.
Acceptance is equivalent to reaching $`(m,n)`$.

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

Let $`R`$ be the number of alignment cells reachable within budget and
$`\lvert\mathcal{O}\rvert`$ the number of operations. Time is:

```math
\mathcal{O}\left(R\lvert\mathcal{O}\rvert\log R\right)
```

and frontier memory is $`\mathcal{O}(R)`$. The sparse graph usually occupies a
narrow cost-bounded band. Zero-cost length-preserving operations remain on a
diagonal. Nevertheless, attacker-controlled operation sets and inputs can
increase $`R`$; evaluation stops before discovering cell
`MAX_GENERALIZED_ALIGNMENT_STATES + 1` and returns a resource error. The
initial cell counts as the first materialized cell. All coordinate, cost,
scale, and discovery-count additions are checked before indexing or insertion.

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
# Ok::<(), Box<dyn std::error::Error>>(())
```

## 9. Verification and tests

| Evidence | Invariant |
|---|---|
| Rocq `GeneralizedAutomatonRepair.v` | path folds, completion charging, exact rescaling, explicit-infinity and finite rates, certified/conservative subsumption, discovery guarding, operation-order independence, antichain set semantics, Hamming length preservation, absent-deletion rejection, coordinate progress |
| Verus `generalized_automaton.rs` | positive completion charging, exact rescaling, exact rate comparison, certified/conservative subsumption, discovery guarding, minimum-cost order independence, equal-position insertion idempotence, checked accumulation, `0.15` boundary |
| Z3 + cvc5 `generalized_automaton.smt2` | twelve bounded counterexample queries are UNSAT in both solvers |
| `proptest_generalized_automaton_repair.rs` | standard/reference, Hamming, indel/LCS, bounded-skip/subsequence, budget monotonicity, fractional, Unicode, invalid values |
| generalized unit suite | standard, transpose, merge/split, phonetic, empty-side, and changed exact-weight examples |

The generalized operation model follows Mitankin, Mihov, and Schulz,
“Deciding word neighborhood with universal neighborhood automata,”
*Theoretical Computer Science* 412(22), 2011,
[doi:10.1016/j.tcs.2011.01.013](https://doi.org/10.1016/j.tcs.2011.01.013).
