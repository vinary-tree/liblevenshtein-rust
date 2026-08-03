# True-Damerau streaming design

**Status:** implemented and evidence-gated · **Owner:** Phase 6 under root epic
`extending-liblevenshtein-automaton-families-4bb97598`

## Decision

Implement budget-bounded unrestricted Damerau–Levenshtein as a static
`AutomatonVariant` whose `DamerauPending` positions carry one endpoint delta.
Keep `State` and public query iterator types concrete. Select the variant once
per dictionary edge through `with_variant!`.

This design preserves every Standard transition, adds exact Lowrance–Wagner
macro chains, keeps `Position` at 24 bytes on 64-bit targets, and avoids
alphabet-sized history in a state.

## Constraints that shaped the design

1. `Algorithm` is a runtime, serde, FFI, WASM, and REPL value. Parameterizing
   the public iterator by a variant would multiply public types and force
   boxing or a breaking signature change.
2. `OperationSet` represents alignments. True Damerau is history-dependent and
   cannot be recovered from a local transposition tuple.
3. The Lowrance–Wagner budget jointly bounds both macro interiors, so one
   positive delta plus accumulated cost is sufficient for bounded search.
4. State pruning must compare residual languages, not merely coordinates.

## Ownership and data flow

`Algorithm::DamerauLevenshtein` maps to `VariantSpec::Damerau`, which selects
`DamerauV`. `transition_state_inner` computes a characteristic vector at each
position's origin. `DamerauV::successors` delegates to
`transition_damerau_into`. The shared state insertion kernel then applies the
variant's kind-aware subsumption rule.

The [state-flow diagram](../diagrams/automata/true-damerau-macro-chain.svg)
shows the complete lifecycle. The detailed recurrence and literate pseudocode
live in the [algorithm chapter](../algorithms/11-true-damerau/README.md).

## Representation invariant

For every reachable position:

- `Normal` implies `aux == 0`.
- `DamerauPending` implies `$`1\le\mathrm{aux}\le255`$`.
- pending accumulated cost already includes one transposition and all skipped
  query endpoints;
- pending positions never take an epsilon deletion;
- a pending resolution preserves cost and advances exactly `aux + 1` query
  units from the stored origin.

The public transition boundary rejects a true-Damerau budget above 255 before
constructing state. This converts a former silent loss of macros into an
explicit contract failure without changing the Phase 5 layout/code-generation
decision.

## Subsumption rationale

Normal/normal uses the classic diagonal rule. Pending/pending dominance
requires equal origin and delta plus non-greater cost. All cross-kind pairs are
incomparable. This is deliberately conservative: a normal path and a pending
path consume different future languages even if their indices and costs match.

## Rejected alternatives

### Full last-occurrence map per state

It is direct but makes state size depend on the alphabet and would copy or
share a map on each dictionary edge. The bounded macro needs only the one
endpoint currently owed.

### `Position<V>` and `State<V>`

This pushes variant type parameters through every iterator and backend. The
closed, one-dispatch-per-edge seam already gives static hot loops without that
API multiplication.

### Projecting through `OperationSet`

The projection computes OSA because every tuple consumes its symbols once.
The public conversion remains for local-repertoire compatibility and is
documented as lossy.

### Pretending weighted/product engines support it

`PositionF64` and the phonetic products have no delta/history carrier. They now
fail explicitly rather than silently label OSA or Standard behavior as true
Damerau.

## Verification architecture

Rocq proves arithmetic and transition invariants without axioms or admissions.
Verus checks a Rust-shaped executable model. Z3 and cvc5 independently show
the negations of five bounded laws are unsatisfiable. TLA+ explores every
state of the `$`k=3`$` entry/extension/resolution protocol and checks budget,
validity, and no-pending-epsilon invariants. Each law is mirrored in focused or
generated Rust tests so the proof is connected to implementation behavior.

## Operational observations

The repeated-unit release profile measured maximum state sizes 2, 4, and 7 at
budgets 1, 2, and 3. The one-position successor buffer did not spill. The
Birkbeck release gate observed 136 true/OSA separators among 42,395 explicit
pairs and no direction violation. Its restored `DoubleArrayTrie` arm verifies
the exact inventory before checking 37,472 budget-eligible corrections. Full
results and protocol amendments are in
the [scientific ledger](../scientific-ledger/true-damerau-2026-08-01.md).

## Source

The design refines the last-occurrence recurrence of Lowrance and Wagner,
[DOI 10.1145/321879.321880](https://doi.org/10.1145/321879.321880).
