# Exact affine-gap automaton design

**Status:** implemented · **Audience:** maintainers and verifier authors · **Primary source:** [Gotoh 1982](https://doi.org/10.1016/0022-2836(82)90398-9)

## 1. Problem and constraints

Affine gaps need history: the next gap symbol costs `$`g_e`$` when the same gap
is open and `$`g_o+g_e`$` otherwise. A scalar `(index, cost)` position cannot
distinguish those cases. The design must preserve byte/character/token
genericity, exact comparison, pooled traversal, and the 24-byte position seam.

The selected design maps Gotoh's three matrices to `PositionKind` and supplies
typed parameters through `AutomatonVariant::Params`. It does not add an
`Algorithm` variant because `Algorithm` is a parameter-free unit-cost selector.

## 2. Component map

| Component | Responsibility |
|---|---|
| `AffineGapParams` | exact decimal scaling, getters, budget conversion |
| `AffineV` | successors, epsilon closure, B-4 subsumption, finish, window |
| `QueryVariant::Affine` | carry typed parameters through the existing iterator |
| `AffineQueryIterator` | convert scaled candidate costs for presentation |
| `affine_gap_distance_units` | independent quadratic oracle |
| formal tree | prove arithmetic and bounded-transition invariants |

The ordinary and affine iterators share traversal, path reconstruction,
dictionary units, substitution policies, and state pooling. Only the five
variant operations differ.

## 3. Rejected alternatives

### 3.1 Reuse the `_f64` engine

Rejected because exact decimal input is already representable through
`CostScale`. Floating comparison would reintroduce epsilon policy, NaN branches,
and platform-sensitive ordering while duplicating 3,637 lines of weighted
state machinery.

### 3.2 Put costs on `Algorithm`

Rejected because an enum value cannot carry the parameter set without changing
every serialized, FFI, and product surface. `VariantSpec::AffineGap` is a seam
marker; typed entry points dispatch directly to `AffineV`.

### 3.3 Store three full DP rows per trie node

Rejected because it abandons canonical position frontiers and state pooling.
The `PositionKind` byte is sufficient history.

### 3.4 Enable B-5 immediately

Rejected by a generated counterexample. Cross-index pruning changes which
epsilon representatives survive closure. It is safe only after the transition
kernel can fuse the priced query skip with consumption of the current
dictionary edge. The root-epic task records that precondition.

## 4. Safety contracts

- Decimal conversion is exact or returns `ScaleError`.
- Every transition cost uses `checked_add`; every run uses checked
  multiplication where multiplication occurs.
- A route exceeding the exact integer domain is absent, never wrapped.
- B-4 compares only positions at the same query index.
- `$`g_e=0`$` is exact but uses the full remaining-query window.
- Suffix dictionaries retain their documented substring completion semantics.

## 5. Verification architecture

The proof stack is intentionally redundant:

1. Rocq proves B-4 for arbitrary action traces over mathematical integers.
2. Dafny automatically verifies the imperative-style arithmetic contracts.
3. Verus mirrors the Rust-facing natural/integer guards.
4. Z3 and cvc5 independently search for counterexamples to ten negated claims.
5. TLC explores all bounded B-4 layer/cost traces in the configured model.
6. Proptest instantiates the same claims over concrete suffix DPs and the real
   dictionary automaton.

The [formal manifest](../verification/FORMAL_VERIFICATION_MANIFEST.tsv) is the
single registration point. Trusted proof files contain no admitted theorem,
axiom, external body, or assumed function.

## 6. Maintenance rule

Any change to gap convention, `gap_step`, layer order, finish cost, or window
must update all of the following in one change: reference DP, automaton kernel,
Rocq, Dafny, Verus, SMT, TLA+, direct properties, literate chapter, and diagram.
This duplication is deliberate cross-validation, not production-code drift.
