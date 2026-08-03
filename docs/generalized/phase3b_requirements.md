# Phonetic split and two-for-two operation requirements

**Status:** implemented and verified · **Supersedes:** the 2025 Phase 3b
planning draft · **Current oracle:** `GeneralizedAutomaton`

This document records the closed requirements for restricted
$`\langle1,2\rangle`$ split and $`\langle2,2\rangle`$ two-for-two operations.
The original draft proposed charging on speculative entry and subtracting on
completion. That design was unsafe for fractional weights and for competing
operations. The shipped design charges exactly once, after the complete source
and target slices are known.

![Restricted and unrestricted operations advance through the exact generalized alignment grid.](../diagrams/automata/generalized-operation-grid.svg)

## 1. Terminology and notation

An operation $`t`$ has the form

```math
t=\langle t^x,t^y,t^w,t^r\rangle,
```

where $`t^x`$ and $`t^y`$ are source and target Unicode-scalar consumption,
$`t^w`$ is a non-negative cost, and $`t^r`$ is an optional set of allowed
source/target string pairs. A **split** consumes one source scalar and two
target scalars. A **two-for-two operation** consumes two scalars on each side;
the built-in unrestricted operation named `transpose` specifically requires
adjacent reversal.

A **completion charge** is the exact scaled cost added only when every scalar
needed to validate $`t^r`$ is available. A **control position** is the variant,
offset, and pending-entry character that determine a streaming state's future.

## 2. Required semantics

For source slice $`u`$ and target slice $`v`$, an operation contributes an
alignment edge exactly when:

```math
|u|=t^x \land |v|=t^y \land t^r(u,v),
```

where lengths count Unicode scalar values. The edge adds the fixed-point
integer $`\operatorname{scaled}(t^w)`$. No cast, truncation, epsilon comparison,
or provisional cost is permitted.

Examples:

| Rule | Consumption | Exact cost | Valid pair |
|---|---:|---:|---|
| `k` to `ch` | $`\langle1,2\rangle`$ | `0.15` | `("k", "ch")` |
| `qu` to `kw` | $`\langle2,2\rangle`$ | `0.15` | `("qu", "kw")` |
| adjacent transpose | $`\langle2,2\rangle`$ | `1.0` | `("ab", "ba")` |

At integer budget `1`, six rules of cost `0.15` cost `0.90` and fit; seven
cost `1.05` and do not.

## 3. Operation-complete alignment oracle

`GeneralizedAutomaton` evaluates a sparse, topologically ordered alignment
grid. It supports every operation for which $`t^x+t^y>0`$, including arities
larger than two. Restricted operations call `OperationType::applies_to_slices`
on exact UTF-8 slices whose scalar counts match the declared dimensions.

```pseudocode
for each reachable cell (source_index, target_index) do
    for each configured operation t do
        destination <- checked coordinate advance by (t.x, t.y)
        if destination is in bounds and t applies to both exact slices then
            candidate <- checked_add(current_cost, scaled(t.weight))
            retain the minimum in-budget candidate at destination
        end if
    end for
end for
```

The minimum is taken across every applicable operation. Consequently,
acceptance and distance are independent of `OperationSet` insertion order.

## 4. Bounded streaming compatibility state

`GeneralizedState` retains the historical universal-position API for streamed
transitions. Its finite intermediate vocabulary supports one-scalar rules plus
$`\langle2,2\rangle`$, $`\langle2,1\rangle`$, and
$`\langle1,2\rangle`$ rules.

The state carries:

- the exact common `CostScale` and scaled budget;
- an antichain of exact-`usize` position costs;
- the preceding input scalar for two-step two-for-two validation;
- the first target scalar in each split position.

Entry into a split or two-for-two intermediate costs zero. Completion rebuilds
the exact two-scalar target slice, checks every applicable operation, and adds
its scaled weight once. Distinct split entry characters are distinct control
positions and cannot subsume one another.

If a rule has an arity outside the finite vocabulary,
`GeneralizedState::try_transition` returns
`GeneralizedStateError::UnsupportedOperationArity`. The infallible wrapper
maps that error to no transition. This boundary is explicit; no operation is
silently ignored.

## 5. Subsumption requirement

The classical Levenshtein rule uses cost slack to cover offset slack:

```math
i\mathbin{\preceq}j
\quad\Longleftrightarrow\quad
e<f \land |j-i|\le f-e.
```

That implication assumes the complete unrestricted unit-cost Levenshtein
lattice. A denominator of one does not prove the assumption because an integer
operation may cost two. Therefore the implementation certifies the classical
branch only for exactly one unrestricted match, substitution, insertion, and
deletion with costs `0`, `1`, `1`, and `1`. Every other operation set uses the
conservative rule:

```math
p\mathbin{\preceq}q
\quad\Longleftrightarrow\quad
\operatorname{control}(p)=\operatorname{control}(q)
\land \operatorname{cost}(p)<\operatorname{cost}(q).
```

This can retain more positions, but it cannot discard a distinct future.

## 6. Resource and failure requirements

The alignment oracle counts unique cells when they are discovered, including
the initial cell. It checks `MAX_GENERALIZED_ALIGNMENT_STATES` before inserting
a vacant frontier entry. Coordinates, costs, rescaling, budgets, and discovery
counts use checked arithmetic. Fallible APIs distinguish invalid weights,
overflow, and resource exhaustion; Boolean acceptance fails closed.

Unicode consumption is defined in scalar values, not bytes or grapheme
clusters. Restriction storage remains UTF-8 byte based only after exact scalar
slices have been selected.

## 7. Verification and executable evidence

The high-risk fractional-completion item is closed by mutually reinforcing
evidence:

| Layer | Checked invariant |
|---|---|
| Rocq | positive completion is not free; rescaling preserves rational value; explicit-infinity and finite rates are exact; uncertified subsumption requires identical coordinates; discovery guard precedes materialization; minimum is commutative; equal insertion is idempotent and preserves duplicate-freedom |
| Verus | Rust-facing arithmetic versions of completion, rescaling, rate comparison, subsumption, discovery, minimum-cost, and equal-insertion obligations |
| Z3 and cvc5 | twelve bounded negated obligations are independently UNSAT |
| example tests | `0.15`, operation-order, Unicode, invalid-weight, unsupported-arity, empty-side, transpose, merge, and split boundaries |
| property tests | standard/reference equivalence, Hamming length, indel/LCS identity, bounded-skip/subsequence identity, budget monotonicity, exact decimals, and subsumption laws |

The proof sources and trust status are listed in the
[formal-verification manifest](../verification/FORMAL_VERIFICATION_MANIFEST.tsv).
The operation-complete algorithm is developed in literate form in the
[generalized-operation grid chapter](../algorithms/14-generalized-operation-grid/README.md),
and deployment limits are covered by the
[resource-exhaustion guide](../security/resource-exhaustion.md).

The theoretical operation model follows Mitankin, Mihov, and Schulz,
“Deciding word neighborhood with universal neighborhood automata,”
*Theoretical Computer Science* 412(22), 2011,
[doi:10.1016/j.tcs.2011.01.013](https://doi.org/10.1016/j.tcs.2011.01.013).
